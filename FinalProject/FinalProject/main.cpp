#include<SDL.h>
#include<stdio.h>
#include<iostream>
#include<Eigen/Dense>
#include<vector>
#include "Edge.h"

using namespace Eigen;

// Simulation Parameters 
const int SCREEN_WIDTH = 640; // The horizontal dimension of the screen, in pixels
const int SCREEN_HEIGHT = 480; // The vertical dimension of the screen, in pixels

const int M = 640; // The number of grid spaces per dimension that will represent the scene
int N; // The number of edges in the scene, determined dynamically in init()
const int G_SIZE = SCREEN_WIDTH / M; // The dimension of each individual grid space, calculated to fit exactly on width of screen.
const int EDGE_CHUNK_LENGTH = 64; // The maximum length an edge is allowed to have before it is subdivided
const float RAY_SAMPLE_SPACING = 32.0f; // The length along edges between ray samples when checking for visibility between edges.
const int ATTENUATION_DIST = 1000; // The square-root distance (in screen-pixels) at which a sampled point recieves half the intensity of the light source.

SDL_Window *g_window = NULL; // SDL structure representing the screen window
SDL_Renderer *g_renderer = NULL; // SDL structure representing a device-agnostic real-time renderer

bool init(); // Initialization function where all lighting calculations are made.
void close(); // Clean-up function where SDL structures are disposed of

// Given two pairs of points representing two line segments a and b, returns
// the fraction along a as a float between 0 and 1 where the two segments intersect, if they do
// Returns INFINITY if the two segments do not intersect
float line_line_intersection(const Vector2f &a0, const Vector2f &a1, const Vector2f &b0, const Vector2f &b1);
// Given two edge indices i and j, casts rays between the two checking to see
// if obstacles exist between them. Returns the fraction of rays that successfully
// traverse from one edge to the other
float edge_ray_factor(int i, int j);
// Given an edge i and a gridspace (x, y), returns the fraction of rays cast from uniformly
// sampled points along i that successfully reach (x, y)
float grid_ray_factor(int i, int x, int y);

struct Graph {
	Vector2f *V; // Our set of vertices
	Edge *E; // Our set of edges
} G; // The graph representing the obstacles in the scene

VectorXf R; // Our N-length reflectivity vector
VectorXf S; // Our N-length emissivity vector
VectorXf B; // Our N-length radiosity vector
MatrixXf F; // Our N x N visibility factor matrix for edges to edges
MatrixXf H; // Our N x M^2 visibility factor matrix for edges to grid spaces

struct GridSpace {
	Uint8 r, g, b; // Our 3 color factors, with values of 0-255
	float illum; // The illumination factor of this grid space
};

GridSpace L[M * M]; // Our level, represented by a matrix of grid spaces

int main(int argc, char **args) {
	if (!init()) return -1;
	
	// Clear Screen
	SDL_SetRenderDrawColor(g_renderer, 0x0, 0x0, 0x0, SDL_ALPHA_OPAQUE);
	SDL_RenderClear(g_renderer);

	// BEGIN Draw Level
	// For each grid space in L, draw that grid according to its color components
	// scaled by its illumination factor.
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < M; ++j) {
			GridSpace l = L[i * M + j];
			// Prevent any color component as rendered from going above
			// 0xFF (255) to prevent color blowout. This can happen when
			// Edges have very high emissivities/radiosities.
			Uint8 r = (Uint8)std::fminf(l.illum * l.r, 0xFF);
			Uint8 g = (Uint8)std::fminf(l.illum * l.g, 0xFF);
			Uint8 b = (Uint8)std::fminf(l.illum * l.b, 0xFF);
			SDL_SetRenderDrawColor(g_renderer, r, g, b, SDL_ALPHA_OPAQUE);
			// We draw a rectangle of our grid unit size for each grid space,
			// colored accordingly.
			SDL_Rect grid_rect = { i * G_SIZE, j * G_SIZE, G_SIZE, G_SIZE };
			SDL_RenderFillRect(g_renderer, &grid_rect);
		}
	// END Draw Level

	// BEGIN Draw graph edges
	SDL_SetRenderDrawColor(g_renderer, 0xFF, 0xFF, 0xFF, SDL_ALPHA_OPAQUE);
	for (int i = 0; i < N; ++i) {
		Vector2f *u = G.E[i].u; // Starting point of edge
		Vector2f *v = G.E[i].v; // End point of edge
		// We shade edges roughly corresponding to the magnitude of their radiosity
		// This is just as a helpful visual cue
		int x = int(0xFF * fminf(B[i], 1.0f));
		SDL_SetRenderDrawColor(g_renderer, x, x, x, SDL_ALPHA_OPAQUE);

		SDL_RenderDrawLine(g_renderer, u->x(), u->y(), v->x(), v->y());

		Vector2f c = G.E[i].center();
		// Calculate elongated, offset normal vector, draw it
		// We do this to indicate which side of the is the 'real' one;
		// the side which emits and reflects light
		Vector2f n = 10.0f * G.E[i].normal() + c;
		SDL_RenderDrawLine(g_renderer, c.x(), c.y(), n.x(), n.y());
	}
	// END Draw Graph Edges

	// Render the commands pushed onto the renderer to the window
	SDL_RenderPresent(g_renderer);

	// SDL Loop
	// This is used exclusively for detecting the user ending the program.
	SDL_Event e;
	bool running = true;

	while (running) {
		// Event handling loop
		while (SDL_PollEvent(&e) != 0) {
			if (e.type == SDL_QUIT)	running = false;
			// Escape also ends the program
			else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
				running = false;
		}
	}
	// Close releases resources used by SDL.
	close();
	return 0;
}

bool init() {
	Uint32 time = 0, start_time = 0;

	// ============= SDL INITIALIZATION ================
	start_time = time = SDL_GetTicks();
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	g_window = SDL_CreateWindow("Dumas Radiosity Solution", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
	if (g_window == NULL) {
		printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	g_renderer = SDL_CreateRenderer(g_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (g_renderer == NULL) {
		printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
		return false;
	}
	printf("SDL Initialization \ttook %d milliseconds.\n", SDL_GetTicks() - time);
	// ============= LEVEL INITIALIZATION ================
	time = SDL_GetTicks();
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < M; ++j)
			L[i + M * j] = GridSpace{ 0x33, 0x33, 0x33, 0.0f };
	printf("Level Initialization \ttook %d milliseconds.\n", SDL_GetTicks() - time);
	// ============= GRAPH INITIALIZATION ================
	time = SDL_GetTicks();
	G = Graph{};
	
	const int NUM_V = 8;
	Vector2f vs[NUM_V];
	vs[0] = Vector2f{ 360, 360 };
	vs[1] = Vector2f{ 300, 400 };

	vs[2] = Vector2f{ 10, 200 };
	vs[3] = Vector2f{ 200, 10 };

	vs[4] = Vector2f{ 200, 200 };
	vs[5] = Vector2f{ 100, 326 };

	vs[6] = Vector2f{ 280, 250 };
	vs[7] = Vector2f{ 220, 200 };

	//vs[8] = Vector2f{ 10, 10 };
	//vs[9] = Vector2f{ SCREEN_WIDTH - 10, 10 };

	//vs[10] = Vector2f{ 10, SCREEN_HEIGHT - 10 };
	//vs[11] = Vector2f{ 10, 10 };

	//vs[12] = Vector2f{ SCREEN_WIDTH - 10, 10 };
	//vs[13] = Vector2f{ SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10 };

	//vs[14] = Vector2f{ SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10 };
	//vs[15] = Vector2f{ 10, SCREEN_HEIGHT - 10 };
	//G.V = vs;

	// Now that we have our initial set of vertex pairs, we need to subdivide them into edges based on our EDGE_CHUNK_LENGTH.
	// The new set of subdivided edges will be what we use in our calculations.
	N = 0;
	std::vector<Edge> es{};

	for (int i = 0; i < NUM_V - 1; i += 2) {
		Vector2f u = vs[i], v = vs[i + 1]; // Vertex pair representing an edge
		Edge e = { &u, &v };
		// Track from u to v, creating a new vertex and edge every time we surpass EDGE_CHUNK_LENGTH;
		Vector2f x = u, prev_x = u;
		int j = 0;
		while (x.x() != e.v->x() && x.y() != e.v->y()) { // We're done if we've reached v.
			x = e.track(EDGE_CHUNK_LENGTH, ++j); // Find our next point along the line segment.
			es.push_back(Edge{ new Vector2f{ x }, new Vector2f{ prev_x } });
			prev_x = x;
		}
	}
	N = es.size();
	G.E = new Edge[N];
	std::copy(es.cbegin(), es.cend(), G.E);

	printf("Graph Initialization \ttook %d milliseconds.\n", SDL_GetTicks() - time);
	// ============= VECTOR INITIALIZATION ===============
	time = SDL_GetTicks();
	R = VectorXf{ N };
	S = VectorXf{ N };
	B = VectorXf{ N };

	for (int i = 0; i < N; ++i) {
		R[i] = 0.5f;
		S[i] = 0.0f;
	}

	S[5] = 10.0f; // Create a base light-source.
	printf("Vector Initialization \ttook %d milliseconds.\n", SDL_GetTicks() - time);
	// ======= EDGE-EDGE VIEW-FACTOR CALCULATION =========
	time = SDL_GetTicks();
	// For each edge in E, calculate the visiblity of that edge to all other
	// edges. Use the equation defined in the proposal. Note that F is symmetric;
	// calculating F_ij also calculates F_ji.
	F = MatrixXf{ N, N };

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < i; ++j) {
			Edge e_i = G.E[i], e_j = G.E[j];

			//Angle-finding method for perpendicularity
			Edge connector = { &e_i.center(), &e_j.center() };
			float theta = connector.angle();
			// We add PI to the casting angle to correctly show its relationship to the connector
			float theta_i = e_i.normal_angle() - theta + M_PI;
			float theta_j = e_j.normal_angle() - theta;
			float cos_product = cosf(theta_i) * cosf(theta_j);
			if (cos_product <= 0) F(i, j) = F(j, i) = 0.0f;
			else {
				float r2 = connector.w.squaredNorm();
				F(i, j) = F(j, i) = (ATTENUATION_DIST / (ATTENUATION_DIST + r2)) * cos_product * edge_ray_factor(i, j);
			}
		}

	for (int i = 0; i < N; ++i) F(i, i) = 1.0f; // We ensure all edges have full visibility to themselves
	printf("Edge-Edge View-Factor \ttook %d milliseconds.\n", SDL_GetTicks() - time);
	// ============= RADIOSITY CALCULATION ===============
	time = SDL_GetTicks();
	// Setting up the linear system
	MatrixXf rF(N, N);
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			rF(i, j) = R(i) * F(i, j);

	MatrixXf Q = MatrixXf::Identity(N, N) - rF;
	
	B = Q.ldlt().solve(S);

	printf("Radiosity Calculation \ttook %d milliseconds.\n", SDL_GetTicks() - time);
	// ===== EDGE-GRIDSPACE VIEW-FACTOR CALCULATION ======
	time = SDL_GetTicks();
	H = MatrixXf{ N, M * M };
	for (int i = 0; i < N; ++i) {
		Vector2f edge_loc = G.E[i].center();
		for (int x = 0; x < M; ++x)
			for (int y = 0; y < M; ++y) {
				int j = x * M + y;
				Vector2f grid_loc = Vector2f{ x * G_SIZE, y * G_SIZE };

				Edge connector = { &edge_loc, &grid_loc };
				float theta = G.E[i].normal_angle() - connector.angle();
				float cos = cosf(theta);
				if (cos < 0.0f) H(i, j) = 0.0f;
				else {
					float r2 = (edge_loc - grid_loc).squaredNorm();
					H(i, j) = (ATTENUATION_DIST / (ATTENUATION_DIST + r2)) * cos * grid_ray_factor(i, x, y);
				}
			}
	}
	printf("Edge-Grid View-Factor \ttook %d milliseconds.\n", SDL_GetTicks() - time);
	// ======= GRIDSPACE ILLUMINATION CALCULATION ========
	time = SDL_GetTicks();
	for (int i = 0; i < N; ++i)
		for (int x = 0; x < M; ++x)
			for (int y = 0; y < M; ++y) {
				int j = x * M + y;
				// Each grid space holds the sum of all of the edges' radiosities 
				// scaled by that grid-space's view-factor to that edge.
				L[j].illum += H(i, j) * B[i];
			}
	printf("Gridspace Illumination \ttook %d milliseconds.\n", SDL_GetTicks() - time);

	printf("Total Lighting Calc \ttook %d milliseconds.\n", SDL_GetTicks() - start_time);
	return true;
}

void close() {
	SDL_DestroyRenderer(g_renderer);
	g_renderer = NULL;
	SDL_DestroyWindow(g_window);
	g_window = NULL;

	SDL_Quit();
}

float line_line_intersection(const Vector2f &a0, const Vector2f &a1,
	const Vector2f &b0, const Vector2f &b1)
{
	float d;
	Vector2f a = a1 - a0; // Line segment represented by the first two vectors
	Vector2f b = b1 - b0; // Line segment represented by the second two vectors

	// Make sure the lines aren't parallel by comparing their slopes
	if (a.y() / a.x() != b.y() / b.x())
	{	// 2D Cross Product!
		d = a.x() * b.y() - a.y() * b.x();
		if (d != 0)
		{
			Vector2f ab = a0 - b0;
			// These coefficients represent what percentage along each vector the intersection occurs
			// It should probably be verified that r and s are in between 0 and 1!
			float r = (ab.y() * b.x() - ab.x() * b.y()) / d; // The coefficient of intersection for a
			float s = (ab.y() * a.x() - ab.x() * a.y()) / d; // The coefficient of intersection for b
			if (0.0f < r && r < 1.0f && 0.0f < s && s < 1.0f) // This guarantees that the collision occurs on the line
				// Checking only one of these coefficients only checks one-dimensionally
				return r;
			else return HUGE_VALF;
		}
	}
	return false;
}

float edge_ray_factor(int i, int j) {
	Edge e_i = G.E[i], e_j = G.E[j];

	int n_samples_i = int(ceilf(e_i.w.norm() / RAY_SAMPLE_SPACING));
	int n_samples_j = int(ceilf(e_j.w.norm() / RAY_SAMPLE_SPACING));
	int n_samples;
	if (n_samples_i < n_samples_j) n_samples = n_samples_i;
	else n_samples = n_samples_j; // To save time an energy, we just want to use as many rays as the shortest segment needs

	int hits = 0;
	for (int k = 0; k <= n_samples; ++k) {
		// We uniformly sample n_samples points along each edge
		Vector2f sample_i = e_i.interpolate(float(k) / n_samples);
		Vector2f sample_j = e_j.interpolate(float(k) / n_samples);
		bool ray_hit = true; // We assume that there is no obstacle until proven otherwise

		for (int u = 0; u < N; ++u) // We check every edge
			if (u != i && u != j) { // That is not either of our sample edges
				// We test them to see if they intercept the line connecting our two sampled points
				// If there is an intersection along the segment from sample_i to sample_j,
				// r is > 0 and < 1, which indicates how far along that segment the other edge
				// intercepts our segment. If this is the case then that edge blocks the ray going from
				// sample_i to sample_j before it can reach sample_j.
				float r = line_line_intersection(sample_i, sample_j, *G.E[u].u, *G.E[u].v);
				if (r > 0.0f && r < 1.0f) {
					ray_hit = false;
					break;
				}
			}

		// If the ray DID hit sample_j, me make note of it
		if (ray_hit) ++hits;
	}

	// We return the fraction of rays that successfully reached their target
	return float(hits) / n_samples;
}

// From edge E_i to grid-space L_j
float grid_ray_factor(int i, int x, int y)
{
	Vector2f grid_loc = Vector2f{ x * G_SIZE, y * G_SIZE };
	// We just want to cast a ray from each endpoint of edge i to the location of 
	// our gridspace. Return proportion of rays that hit that grid-space.
	int hits = 0;
	Edge e_i = G.E[i];
	int n_samples = 6; // int(ceilf(e_i.w.norm() / RAY_SAMPLE_SPACING));

	for (int k = 0; k <= n_samples; ++k) {
		// We uniformly sample n_samples points along each edge
		Vector2f sample_i = e_i.interpolate(float(k) / n_samples);
		bool ray_hit = true; // We assume that there is no obstacle until proven otherwise
		for (int u = 0; u < N; ++u) // We check every edge
			if (u != i) { // That is not either of our sample edges
				// We test them to see if they intercept the line connecting our two sampled points
				// If there is an intersection along the segment from sample_i to sample_j,
				// r is > 0 and < 1, which indicates how far along that segment the other edge
				// intercepts our segment. If this is the case then that edge blocks the ray going from
				// sample_i to sample_j before it can reach sample_j.
				float r = line_line_intersection(sample_i, grid_loc, *G.E[u].u, *G.E[u].v);
				if (r > 0.0f && r < 1.0f) {
					ray_hit = false;
					break;
				}
			}

		// If the ray DID hit sample_j, me make not of it
		if (ray_hit) ++hits;
	}

	// We return the fraction of rays that successfully reached their target
	return float(hits) / n_samples;
}