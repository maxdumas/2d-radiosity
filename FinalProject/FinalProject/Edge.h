#pragma once
#include<Eigen/Dense>

class Edge
{
public:
	Edge();
	Edge(Eigen::Vector2f *u, Eigen::Vector2f *v);

	~Edge();

	float angle();
	Eigen::Vector2f normal();
	float normal_angle();
	Eigen::Vector2f center();
	// Standard linear interpolation, a = 0 return u, a = 1 returns v
	Eigen::Vector2f interpolate(float a);
	// Performs a stepping track along the edge from u to v, returning step
	// n, which is along delta * n of (u - v). Returns Zero if n or delta is
	// such that tracking goes past v.
	// VERIFY THAT THIS WORKS CORRECTLY.
	Eigen::Vector2f track(float delta, int n);

	Eigen::Vector2f *u;
	Eigen::Vector2f *v;

	// The vector representing the line segment connecting u to v.
	Eigen::Vector2f w;
};

