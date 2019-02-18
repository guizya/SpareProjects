#pragma once
#include <vector>
using namespace std;

class Utils {
public:
	static void normalize(float &v1, float &v2, float &v3);
	static float includedAngle(float v1, float v2, float v3, float t1, float t2, float t3);
	static float degreeToRadian(float degree);
	static float length(float v1, float v2, float v3);
	static float lengthRound(float v1, float v2, float v3);
	static bool zeroVector(vector<float> vec);
	static vector<float> rotate(float angle, float d0, float d1, float d2, float p0, float p1, float p2);
	static void copy3(vector<float> &dst, int index, const vector<float> &src, int srcIdx);
	static void shade(float &color0, float &color1, float &color2, float f0, float f1, float f2, int sp);

	static double PI;
};