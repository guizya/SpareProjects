#include "Euler.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include <ctime>

float* h_points1;
float* h_points2;
float* d_points1;
float* d_points2;
float* h_descriptor1;
float* h_descriptor2;
float* d_descriptor1;
float* d_descriptor2;
unsigned pointsCount1;
unsigned pointsCount2;
float *d_EulerResults;
unsigned int *h_results1;
unsigned int *h_results2;
unsigned int *d_results1;
unsigned int *d_results2;

// 准备数据，并将数据传输到GPU
void eulerStart(vector<KeyPoint>& points1, vector<KeyPoint>& points2, Mat descriptor1, Mat descriptor2)
{
	pointsCount1 = points1.size();
	pointsCount2 = points2.size();

	h_points1 = new float[pointsCount1 * 2]();
	h_points2 = new float[pointsCount2 * 2]();

	for (int i = 0; i < pointsCount1; i++) {
		h_points1[i * 2] = points1[i].pt.x;
		h_points1[i * 2 + 1] = points1[i].pt.y;
	}

	for (int i = 0; i < pointsCount2; i++) {
		h_points2[i * 2] = points2[i].pt.x;
		h_points2[i * 2 + 1] = points2[i].pt.y;
	}

	h_descriptor1 = new float[pointsCount1 * 80]();
	h_descriptor2 = new float[pointsCount2 * 80]();

	for (int i = 0; i < pointsCount1; i++) {
		for (int j = 0; j < 80; j++)
			h_descriptor1[i * 80 + j] = descriptor1.at<float>(i, j);
	}

	for (int i = 0; i < pointsCount2; i++) {
		for (int j = 0; j < 80; j++)
			h_descriptor2[i * 80 + j] = descriptor2.at<float>(i, j);
	}

	h_results1 = new unsigned int[pointsCount1]();
	h_results2 = new unsigned int[pointsCount2]();

	checkCudaErrors(cudaMalloc((void **)&d_points1, pointsCount1 * 2 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_points2, pointsCount2 * 2 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_descriptor1, pointsCount1 * 80 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_descriptor2, pointsCount2 * 80 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_EulerResults, pointsCount1 * pointsCount2 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_results1, pointsCount1 * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void **)&d_results2, pointsCount2 * sizeof(unsigned int)));

	checkCudaErrors(cudaMemset(d_EulerResults, 0x0, pointsCount1 * pointsCount2 * sizeof(float)));
	checkCudaErrors(cudaMemset(d_results1, 0xf, pointsCount1 * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_results2, 0xf, pointsCount2 * sizeof(unsigned int)));

	checkCudaErrors(cudaMemcpy(d_points1, h_points1, pointsCount1 * 2 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_points2, h_points2, pointsCount2 * 2 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_descriptor1, h_descriptor1, pointsCount1 * 80 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_descriptor2, h_descriptor2, pointsCount2 * 80 * sizeof(float), cudaMemcpyHostToDevice));
}

void eulerExecute(vector<pair<KeyPoint, KeyPoint> >& matches1, vector<pair<KeyPoint, KeyPoint> > &matches2,
	vector<KeyPoint>& points1, vector<KeyPoint>& points2)
{
	clock_t start = clock();
	euler();
	clock_t end = clock();
	std::cout << "Euler time: " << (float)(end - start) / CLOCKS_PER_SEC << std::endl;

	checkCudaErrors(cudaMemcpy(h_results1, d_results1, pointsCount1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_results2, d_results2, pointsCount2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// 将结果转换成为特征点对
	for (int i = 0; i < pointsCount1; i++) {
		if (h_results1[i] >= pointsCount2) continue;
		matches1.push_back(std::make_pair(points1[i], points2[h_results1[i]]));
	}

	for (int i = 0; i < pointsCount2; i++) {
		if (h_results2[i] >= pointsCount1) continue;
		matches2.push_back(std::make_pair(points2[i], points1[h_results2[i]]));
	}
}

void eulerEnd()
{
	checkCudaErrors(cudaFree(d_points1));
	checkCudaErrors(cudaFree(d_points2));
	checkCudaErrors(cudaFree(d_descriptor1));
	checkCudaErrors(cudaFree(d_descriptor2));
	checkCudaErrors(cudaFree(d_EulerResults));
	checkCudaErrors(cudaFree(d_results1));
	checkCudaErrors(cudaFree(d_results2));

	delete[]h_points1;
	delete[]h_points2;
	delete[]h_descriptor1;
	delete[]h_descriptor2;
	delete[]h_results1;
	delete[]h_results2;
}

void dumpResult(vector<pair<KeyPoint, KeyPoint> >& points, string filename)
{
	ofstream out(filename);
	for (int i = 0; i < points.size(); i++) {
		out << "Index " << i << " " << points[i].first.pt.x << " " << points[i].first.pt.y << " Second " << points[i].second.pt.x << " " << points[i].second.pt.y << std::endl;
	}
}