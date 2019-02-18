#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include "CpuMethods.h"
#include "GpuMethods.h"

using namespace std;

vector<float> pointsRandom;
vector<float> kmeansRandom;
vector<float> points;
vector<int>	clusters;
vector<float> kmeans;

int K = 0;	// k clusters
int N = 0;	// n points
int D = 0;	// d dimensions

void inputRandom() {
	pointsRandom.clear(); kmeansRandom.clear();

	pointsRandom.resize(N * D, 0);
	kmeansRandom.resize(K * D, 0);

	float tude = 10.0f;
	for (int pnt = 0; pnt < N * D; pnt++) {
		pointsRandom[pnt] = ((float)rand() / (float)RAND_MAX) * 2.0f * tude - tude; // in range [-tude, tude]
	}

	// NOTE that the assumption is N >= K;
	if (N < K)
		throw runtime_error("N is supposed to be larger than K");

	vector<bool> flags;
	flags.resize(N, false);
	for (int cluster = 0; cluster < K; cluster++) {

		int idx = -1;
		while(idx == -1 || flags[idx]) idx = rand() % N;
		flags[idx] = true;

		for (int dim = 0; dim < D; dim++)
			kmeansRandom[cluster * D + dim] = pointsRandom[idx * D + dim];
	}
}

void prepare() {
	clusters.clear();
	clusters.resize(N, 0);
	points = pointsRandom;
	kmeans = kmeansRandom;
}

void outputResults(string filename) {
	ofstream file(filename);
	for (int cluster = 0; cluster < K; cluster++) {
		file << "Cluster " << cluster << " :" << std::endl;
		for (int i = 0; i < N; i++) {
			if (clusters[i] == cluster) {
				for (int dim = 0; dim < D; dim++) {
					file << points[i * D + dim] << " ";
				}
				file << std::endl;
			}
		}
	}
}

void outputKmeans(string filename)
{
	ofstream file(filename);
	for (int cluster = 0; cluster < K; cluster++) {
		file << "Cluster " << cluster << " :" << std::endl;
		for (int dim = 0; dim < D; dim++) {
			file << kmeans[cluster * D + dim] << " ";
		}
		file << std::endl;
	}

	file << "Clusters ";
	for (int cluster = 0; cluster < N; cluster++) {
		file << clusters[cluster] << " ";
	}
	file << endl;
}

void main() {
	N = 6144;
	D = 3;

	GPU::analyze();
	//GPU::initMem();

	for (int cluster = 1; cluster < 9; cluster++) {
		K = (1 << cluster);

		inputRandom();
		prepare();
		clock_t startTime = clock();
		CPU::kmeansCPU();
		clock_t endTime = clock();
		clock_t cpuTime = (endTime - startTime);

		vector<int> cpuCluster = clusters;
		ostringstream out;
		out << "CPUkmeans" << K << ".txt";
		//outputKmeans(out.str());
		outputResults(out.str());

		prepare();
		GPU::initialize();
		startTime = clock();
		GPU::kmeansGPU();
		endTime = clock();
		GPU::free();
		clock_t gpuTime = (endTime - startTime);

		vector<int> gpuCluster = clusters;

		ostringstream out1;
		out1 << "GPUkmeans" << K << ".txt";
		//outputKmeans(out1.str());
		outputResults(out1.str());

		//if (cpuCluster.size() != gpuCluster.size()) cout << "!!!!!!!!!!!!!!!!!!!!!Size Mismatch!" << endl;
		//for (int i = 0; i < cpuCluster.size(); i++) if (cpuCluster[i] != gpuCluster[i]) cout << i << "th element mismatch!!!!!!!!!!!!!!!!!!!!!!" << endl;

		std::cout << "k = " << K << "  Kmeans CPU runtime = " << (float)cpuTime / CLOCKS_PER_SEC << " seconds  Kmeans GPU runtime = " << (float)gpuTime / CLOCKS_PER_SEC 
				<< "  speedup = " << (float)cpuTime / (float)gpuTime << endl;
	}

	//GPU::freeMem();
	system("pause");
	return;
}