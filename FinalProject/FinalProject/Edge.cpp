#include "Edge.h"


Edge::Edge()
{
}

Edge::Edge(Eigen::Vector2f *u, Eigen::Vector2f *v)
{
	this->u = u;
	this->v = v;
	this->w = *v - *u;
}

Edge::~Edge()
{
}

float Edge::angle()
{
	return atan2f(w.y(), w.x());
}

Eigen::Vector2f Edge::normal()
{
	float theta = normal_angle();
	return Eigen::Vector2f{ cosf(theta), sinf(theta) };
}

float Edge::normal_angle()
{
	return angle() - M_PI / 2.0f;
}

Eigen::Vector2f Edge::center()
{
	return (*u + *v) / 2.0f;
}

Eigen::Vector2f Edge::interpolate(float a)
{
	return (1.0f - a) * *u + a * *v;
}

Eigen::Vector2f Edge::track(float delta, int i)
{
	float length = w.norm();
	if (delta * i > length) return *v;

	Eigen::Vector2f dir = w / length;
	return *u + dir * delta * i;
}