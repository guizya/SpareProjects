#pragma once

#include "opencv2/opencv.hpp"  
#include <vector>

#define BLOCK_POINT_SIZE_X 32
#define BLOCK_POINT_SIZE_Y 32

using namespace std;
using namespace cv;

extern "C"
void euler();

extern float* h_points1;
extern float* h_points2;
extern float* d_points1;
extern float* d_points2;
extern float* h_descriptor1;
extern float* h_descriptor2;
extern float* d_descriptor1;
extern float* d_descriptor2;
extern unsigned pointsCount1;
extern unsigned pointsCount2;
extern float *d_EulerResults;
extern unsigned int *h_results1;
extern unsigned int *h_results2;
extern unsigned int *d_results1;
extern unsigned int *d_results2;

void eulerStart(vector<KeyPoint>& points1, vector<KeyPoint>& points2, Mat descriptor1, Mat descriptor2);

void eulerExecute(vector<pair<KeyPoint, KeyPoint> >& matches1, vector<pair<KeyPoint, KeyPoint> > &matches2,
	vector<KeyPoint>& points1, vector<KeyPoint>& points2);
void eulerEnd();
void dumpResult(vector<pair<KeyPoint, KeyPoint> >& points, string filename);