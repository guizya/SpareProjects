#pragma once
#include "LssParallel.h"

extern "C"
void eulerKernal(float *descriptor1, float *descriptor2, float *d_points1, float *d_points2,
	unsigned int pointsNum1, unsigned int pointsNum2, unsigned int *d_results,
	unsigned int gridX1, unsigned int gridX2);

struct EulerDistanceStruct {
public:
	EulerDistanceStruct(vector<KeyPoint>& points1, vector<KeyPoint>& points2, LssParallelStruct &descriptor1, LssParallelStruct &descriptor2) {
		pointsNum1 = points1.size();
		pointsNum2 = points2.size();
		eulerPreparation(points1, points2, descriptor1, descriptor2);
	}
	~EulerDistanceStruct() { eulerRelease(); };
	void eulerPreparation(vector<KeyPoint>& points1, vector<KeyPoint>& points2, LssParallelStruct &descriptor1, LssParallelStruct &descriptor2);
	void eulerRelease();
	void eulerParallel(vector<pair<KeyPoint, KeyPoint> >& matches1, vector<pair<KeyPoint, KeyPoint> > &matches2,
		vector<KeyPoint>& points1, vector<KeyPoint>& points2, LssParallelStruct &descriptor1, LssParallelStruct &descriptor2);
	void dumpResult(vector<pair<KeyPoint, KeyPoint> >& points, string filename);
	float *h_points1, *h_points2, *d_points1, *d_points2;
	unsigned int *h_results, *d_results;
	unsigned int pointsNum1;
	unsigned int pointsNum2;
	unsigned int processor;
};