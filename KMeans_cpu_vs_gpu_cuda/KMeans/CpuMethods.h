#pragma once
#include <vector>

using namespace std;

extern vector<float> points;
extern vector<int>	clusters;
extern vector<float> kmeans;
extern int K;
extern int N;
extern int D;

class CPU {
public:
	static float calcLength(int pnt, int cluster);
	static void kmeansCPU();
};