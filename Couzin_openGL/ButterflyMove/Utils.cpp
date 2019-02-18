#include "Utils.h"
#include <cmath>
#include <iostream>

using namespace std;

double Utils::PI = 3.141592654;

float Utils::degreeToRadian(float degree)
{
	return degree / 180.0f * PI;
}

float Utils::length(float v1, float v2, float v3)
{
	float l = sqrt(v1*v1 + v2 * v2 + v3 * v3);
	return l;
}


float Utils::lengthRound(float v1, float v2, float v3)
{
	float l = sqrt(v1*v1 + v2 * v2 + v3 * v3);
	if (l < 1E-8) {
		cout << "length for zero vector" << endl;
		l = 1E-8;
	}
	return l;
}

void Utils::normalize(float &v1, float &v2, float &v3) {
	float l = sqrt(v1*v1 + v2 * v2 + v3 * v3);
	if (l < 1E-8) {
		//cout << "normalize a zero vector" << endl;
		return;
	}

	v1 /= l; v2 /= l; v3 /= l;
}

float Utils::includedAngle(float v1, float v2, float v3, float t1, float t2, float t3)
{
	float vlen = sqrt(v1*v1 + v2 * v2 + v3 * v3);
	float tlen = sqrt(t1*t1 + t2 * t2 + t3 * t3);

	if (vlen < 1E-8 || tlen < 1E-8) {
		cout << "Included a zero vector" << endl;
	}

	vlen = (vlen < 1E-8) ? 1E-8 : vlen;
	tlen = (tlen < 1E-8) ? 1E-8 : tlen;

	float r = v1 * t1 + v2 * t2 + v3 * t3;

	float result = r / (vlen * tlen);
	return acos(result);
}

bool Utils::zeroVector(vector<float> vec)
{
	if (vec.size() != 3) return false;
	if (vec[0] != 0 || vec[1] != 0 || vec[2] != 0) return false;
	return true;
}

// suppose d is the source of rotation
vector<float> Utils::rotate(float angle, float d0, float d1, float d2, float p0, float p1, float p2)
{
	// (b1c2 - b2c1, c1a2 - a1c2, a1b2 - a2b1)
	// a1 b1 c1	a2 b2 c2
	// d0 d1 d2 p0 p1 p2
	float k0 = d1 * p2 - p1 * d2;
	float k1 = d2 * p0 - d0 * p2;
	float k2 = d0 * p1 - p0 * d1;

	float kv0 = p1 * k2 - k1 * p2;
	float kv1 = p2 * k0 - p0 * k2;
	float kv2 = p0 * k1 - k0 * p1;

	float cosA = cos(angle);
	float sinA = sin(angle);
	//cout << "sinaA " << sinA << " cosA " << cosA << endl;

	vector<float> result(3, 0);
	result[0] = d0 * cosA + kv0 * sinA;
	result[1] = d1 * cosA + kv1 * sinA;
	result[2] = d2 * cosA + kv2 * sinA;

	return result;
}

void Utils::copy3(vector<float> &dst, int index, const vector<float> &src, int srcIdx)
{
	if (dst.size() < index + 3) throw runtime_error("Copy3 size error!");
	if (src.size() < srcIdx + 3) throw runtime_error("Copy3 src size error!");

	for (int i = 0; i < 3; i++) dst[index + i] = src[srcIdx + i];
}

void Utils::shade(float &color0, float &color1, float &color2, float f0, float f1, float f2, int sp)
{
	if (sp == 0) {
		color0 = f0 / 2.0;
		color1 = 0.75 + f1 / 4.0;
		color2 = f2 / 2.0;
	}
	else {
		color0 = f0 / 4.0 + 0.75;
		color1 = f1 / 2.0;
		color2 = f2 / 2.0;
	}
}