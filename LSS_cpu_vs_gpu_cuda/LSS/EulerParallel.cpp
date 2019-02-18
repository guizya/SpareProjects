#include "EulerParallel.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include <ctime>

void EulerDistanceStruct::eulerPreparation(vector<KeyPoint>& points1, vector<KeyPoint>& points2, LssParallelStruct &descriptor1, LssParallelStruct &descriptor2)
{
	h_points1 = new float[pointsNum1 * 3]();
	h_points2 = new float[pointsNum2 * 3]();

	unsigned int num[100];
	memset(num, 0, 100 * sizeof(unsigned int));
	// 将特征点对数据重新组织，并在之后传输进GPU端
	for (int i = 0; i < pointsNum1; i++) {
		int x = points1[i].pt.y + 0.5;
		int y = points1[i].pt.x + 0.5;

		int cta_id_x = (x - RADIUS) / BLOCK_PIXEL_X;
		int cta_id_y = (y - RADIUS) / BLOCK_PIXEL_Y;
		int cta_id = cta_id_y * descriptor1.gridX + cta_id_x;

		int temp = num[cta_id] ++;

		h_points1[i * 3] = points1[i].pt.y;
		h_points1[i * 3 + 1] = points1[i].pt.x;
		h_points1[i * 3 + 2] = temp;
	}

	memset(num, 0, 100 * sizeof(unsigned int));
	// 将特征点对数据重新组织，并在之后传输进GPU端
	for (int i = 0; i < pointsNum2; i++) {
		int x = points2[i].pt.y + 0.5;
		int y = points2[i].pt.x + 0.5;

		int cta_id_x = (x - RADIUS) / BLOCK_PIXEL_X;
		int cta_id_y = (y - RADIUS) / BLOCK_PIXEL_Y;
		int cta_id = cta_id_y * descriptor2.gridX + cta_id_x;

		int temp = num[cta_id] ++;

		h_points2[i * 3] = points2[i].pt.y;
		h_points2[i * 3 + 1] = points2[i].pt.x;
		h_points2[i * 3 + 2] = temp;
	}

	// 初始化数组用于存储最后的计算结果
	// 计算结果为特征点对的索引(i, j)的数组，每组(i, j)中，i表示第一系列特征点数组中的第i个特征点，j表示第二系列特征点数组中的第j个特征点
	h_results = new unsigned int[(pointsNum1 + pointsNum2) * 2]();

	// 将特征点的数据传输进GPU
	checkCudaErrors(cudaMalloc((void **)&d_points1, pointsNum1 * 3 * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void **)&d_points2, pointsNum2 * 3 * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void **)&d_results, (pointsNum1 + pointsNum2) * 2 * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_results, 0xf, (pointsNum1 + pointsNum2) * 2 * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(d_points1, h_points1, pointsNum1 * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_points2, h_points2, pointsNum2 * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void EulerDistanceStruct::eulerRelease()
{
	checkCudaErrors(cudaFree(d_points1));
	checkCudaErrors(cudaFree(d_points2));
	checkCudaErrors(cudaFree(d_results));

	delete[]h_points1;
	delete[]h_points2;
	delete[]h_results;
}

void EulerDistanceStruct::eulerParallel(vector<pair<KeyPoint, KeyPoint> >& matches1, vector<pair<KeyPoint, KeyPoint> > &matches2,
	vector<KeyPoint>& points1, vector<KeyPoint>& points2, LssParallelStruct &descriptor1, LssParallelStruct &descriptor2)
{
	//printf("pointNum1 %d, pointNum2 %d\n", pointsNum1, pointsNum2);
	// kernel的设计
	// 每个线程处理d_points1中的一个特征点
	// 每个线程遍历d_points2中的所有特征点
	// 计算特征点之间的欧拉距离，并进行处理，最后保存结果
	eulerKernal(descriptor1.d_results, descriptor2.d_results, d_points1, d_points2, pointsNum1, pointsNum2, d_results, descriptor1.gridX, descriptor2.gridX);


	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(h_results, d_results, (pointsNum1 + pointsNum2) * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// 将结果从GPU上拷出，并重新组织特征点对，最终返回结果
	for (int i = 0; i < pointsNum1; i++) {
		unsigned int point1 = h_results[i * 2];
		unsigned int point2 = h_results[i * 2 + 1];
		if (point1 < pointsNum1)
			matches1.push_back(std::make_pair(points1[point1], points2[point2]));
	}

	for (int i = 0; i < pointsNum2; i++) {
		unsigned int point2 = h_results[pointsNum1 * 2 + i * 2];
		unsigned int point1 = h_results[pointsNum1 * 2 + i * 2 + 1];
		if (point2 < pointsNum2)
			matches2.push_back(std::make_pair(points2[point2], points1[point1]));
	}
}

void EulerDistanceStruct::dumpResult(vector<pair<KeyPoint, KeyPoint> >& points, string filename)
{
	ofstream out(filename);
	for (int i = 0; i < points.size(); i++) {
		out << "Index " << i << " " << points[i].first.pt.x << " " << points[i].first.pt.y << " Second " << points[i].second.pt.x << " " << points[i].second.pt.y << std::endl;
	}
}