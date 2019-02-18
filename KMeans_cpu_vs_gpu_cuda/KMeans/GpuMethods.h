#pragma once
#include <vector>

using namespace std;

#define MAXLOOP 10000

extern vector<float> points;
extern vector<int>	clusters;
extern vector<float> kmeans;
extern int K;
extern int N;
extern int D;

class GPU {
public:
	static void initialize();
	static void free();
	static void analyze();
	static void kmeansGPU();
	static void initMem();
	static void freeMem();

	static int SM;
	static int TotalThreads;
	static int ParallelThreads;
	static int ThreadsPerSM;
	static int SHM;
	static float *d_points;
	static float *d_kmeans;
	static int *d_clusters;
	static int *d_counts;
	static int *d_flags;
	static float *d_debug;
};