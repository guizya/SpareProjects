#ifndef _EULERPARALLEL_CU_
#define _EULERPARALLEL_CU_

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "helper_cuda.h"
#include "EulerParallel.h"

__global__ void eulerKernalLaunch(
	float *descriptor1,
	float *descriptor2,
	float *d_points1,
	float *d_points2,
	unsigned int pointsNum1,
	unsigned int pointsNum2,
	unsigned int gridX1,
	unsigned int gridX2,
	unsigned int *d_results)
{
	float x, y;
	unsigned int pointIdx;
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	float des[80];

	// 加载当前线程的特征点的描述符
	if (globalId < pointsNum1) {
		x = d_points1[globalId * 3];
		y = d_points1[globalId * 3 + 1];
		int xx = x + 0.5;
		int yy = y + 0.5;
		pointIdx = d_points1[globalId * 3 + 2];

		unsigned int cta_id_x = (xx - RADIUS) / BLOCK_PIXEL_X;
		unsigned int cta_id_y = (yy - RADIUS) / BLOCK_PIXEL_Y;
		unsigned int cta_id = cta_id_y * gridX1 + cta_id_x;

		unsigned offset = cta_id * POINTS_IN_BLOCK * 80 + pointIdx * 80;
		for (unsigned i = 0; i < 80; i++)
			des[i] = descriptor1[offset + i];
	}

	// 加载第二幅图片的所有特征点的位置
	__shared__ float pointsInfo[MAX_POINTS_NUM * 3];
	int iteration = pointsNum2 / blockDim.x;
	iteration += (pointsNum2 % blockDim.x == 0 ? 0 : 1);
	for (int iter = 0; iter < iteration; iter++) {
		int index = threadIdx.x * iteration + iter;
		if (index >= pointsNum2) continue;

		for (int i = 0; i < 3; i++)
			pointsInfo[index * 3 + i] = d_points2[index * 3 + i];
	}
	__syncthreads();

	// Shared memory存储d_points2的一部分特征点描述符
	__shared__ float shmDes[DESCRIPTOR_NUM * 80];
	int base = 0;
	iteration = DESCRIPTOR_NUM / blockDim.x;
	iteration += (DESCRIPTOR_NUM % blockDim.x == 0 ? 0 : 1);

	int matchingIndex = -1;
	float minDist = 1e10;
	float secondDist = 1e10;
	int outerIter = pointsNum2 / DESCRIPTOR_NUM;
	outerIter += (pointsNum2 % DESCRIPTOR_NUM == 0 ? 0 : 1);

	float d_temp = 0;

	for (int outer = 0; outer < outerIter; outer++) {
		// 循环为shared memory加载特征点描述符
		for (int iter = 0; iter < iteration; iter++) {
			int shmIndex = threadIdx.x * iteration + iter;
			int index = shmIndex + base;

			if (index >= pointsNum2) continue;
			if (shmIndex >= DESCRIPTOR_NUM) continue;

			int xx = pointsInfo[index * 3] + 0.5;
			int yy = pointsInfo[index * 3 + 1] + 0.5;

			unsigned int cta_id_x = (xx - RADIUS) / BLOCK_PIXEL_X;
			unsigned int cta_id_y = (yy - RADIUS) / BLOCK_PIXEL_Y;
			unsigned int cta_id = cta_id_y * gridX2 + cta_id_x;
			unsigned int num = pointsInfo[index * 3 + 2];

			for (int component = 0; component < 80; component++) {
				shmDes[shmIndex * 80 + component] = descriptor2[cta_id * POINTS_IN_BLOCK * 80 + num * 80 + component];
			}
		}

		__syncthreads();

		if (globalId < pointsNum1) {
			// calcualte the distance
			for (int point = 0; point < DESCRIPTOR_NUM; point++) {
				if (base + point >= pointsNum2) continue;

				float xxx = pointsInfo[(base + point) * 3];
				float yyy = pointsInfo[(base + point) * 3 + 1];

				int xx = xxx + 0.5;
				int yy = yyy + 0.5;
				unsigned int cta_id_x = (xx - RADIUS) / BLOCK_PIXEL_X;
				unsigned int cta_id_y = (yy - RADIUS) / BLOCK_PIXEL_Y;
				unsigned int cta_id = cta_id_y * gridX2 + cta_id_x;
				unsigned int num = pointsInfo[(base + point) * 3 + 2];

				float temp1 = x - xxx;
				float temp2 = y - yyy;
				int size = 20;

				if (temp1 >= size || temp1 <= -size || temp2 >= size || temp2 <= -size) continue;

				float dist = 0;
				for (int component = 0; component < 80; component++) {
					float temp = shmDes[point * 80 + component];
					dist += (des[component] - temp) * (des[component] - temp);
				}

				dist = sqrt(dist);

				if (base + point == 211) {
					d_temp = dist;
				}

				if (dist < minDist) {
					secondDist = minDist;
					minDist = dist;
					matchingIndex = point + base;
				}
				else if (dist < secondDist) {
					secondDist = dist;
				}
			}
		}

		base += DESCRIPTOR_NUM;
		__syncthreads();
	}

	// 将最终结果写进内存
	if (globalId < pointsNum1) {
		if (minDist / secondDist < 0.95) {
			d_results[globalId * 2] = globalId;
			d_results[globalId * 2 + 1] = matchingIndex;
		}
	}
}


__global__ void eulerKernalLaunchNoshm(
	float *descriptor1,
	float *descriptor2,
	float *d_points1,
	float *d_points2,
	unsigned int pointsNum1,
	unsigned int pointsNum2,
	unsigned int gridX1,
	unsigned int gridX2,
	unsigned int *d_results)
{

	// 加载第二幅图片的所有特征点的位置
	__shared__ float pointsInfo[MAX_POINTS_NUM * 3];
	int iteration = pointsNum2 / blockDim.x;
	iteration += (pointsNum2 % blockDim.x == 0 ? 0 : 1);
	for (int iter = 0; iter < iteration; iter++) {
		int index = threadIdx.x * iteration + iter;
		if (index >= pointsNum2) continue;

		for (int i = 0; i < 3; i++)
			pointsInfo[index * 3 + i] = d_points2[index * 3 + i];
	}
	__syncthreads();

	float x, y;
	unsigned int pointIdx;
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	float des[80];

	// 加载当前线程的特征点的描述符
	if (globalId < pointsNum1) {
		x = d_points1[globalId * 3];
		y = d_points1[globalId * 3 + 1];
		int xx = x + 0.5;
		int yy = y + 0.5;
		pointIdx = d_points1[globalId * 3 + 2];

		unsigned int cta_id_x = (xx - RADIUS) / BLOCK_PIXEL_X;
		unsigned int cta_id_y = (yy - RADIUS) / BLOCK_PIXEL_Y;
		unsigned int cta_id = cta_id_y * gridX1 + cta_id_x;

		unsigned offset = cta_id * POINTS_IN_BLOCK * 80 + pointIdx * 80;
		for (unsigned i = 0; i < 80; i++)
			des[i] = descriptor1[offset + i];
	}

	int matchingIndex = -1;
	float minDist = 1e10;
	float secondDist = 1e10;

	if (globalId < pointsNum1) {
		// calcualte the distance
		for (int point = 0; point < pointsNum2; point++) {
			float xxx = pointsInfo[point * 3];
			float yyy = pointsInfo[point * 3 + 1];

			int xx = xxx + 0.5;
			int yy = yyy + 0.5;
			unsigned int cta_id_x = (xx - RADIUS) / BLOCK_PIXEL_X;
			unsigned int cta_id_y = (yy - RADIUS) / BLOCK_PIXEL_Y;
			unsigned int cta_id = cta_id_y * gridX2 + cta_id_x;
			unsigned int num = pointsInfo[point * 3 + 2];

			float temp1 = x - xxx;
			float temp2 = y - yyy;
			int size = 20;
			if (temp1 >= size || temp1 <= -size || temp2 >= size || temp2 <= -size) continue;

			float dist = 0;
			for (int component = 0; component < 80; component++) {
				float temp = descriptor2[cta_id * POINTS_IN_BLOCK * 80 + num * 80 + component];
				dist += (des[component] - temp) * (des[component] - temp);
			}

			dist = sqrt(dist);

			if (dist < minDist) {
				secondDist = minDist;
				minDist = dist;
				matchingIndex = point;
			}
			else if (dist < secondDist) {
				secondDist = dist;
			}
		}
	}

	// 将最终结果写进内存
	if (globalId < pointsNum1) {
		if (minDist / secondDist < 0.95) {
			d_results[globalId * 2] = globalId;
			d_results[globalId * 2 + 1] = matchingIndex;
		}
	}
}


extern "C"
void eulerKernal(float *descriptor1, float *descriptor2, float *d_points1, float *d_points2,
	unsigned int pointsNum1, unsigned int pointsNum2, unsigned int *d_results,
	unsigned int gridX1, unsigned int gridX2)
{
	cudaError_t error = cudaSuccess;

	unsigned int threadNum1 = 128;
	unsigned int threadNum2 = 128;

	dim3 block1(threadNum1, 1, 1);
	dim3 block2(threadNum2, 1, 1);

	unsigned int blockNum1 = ceil((float)pointsNum1 / threadNum1);
	unsigned int blockNum2 = ceil((float)pointsNum2 / threadNum2);

	dim3 grid1(blockNum1, 1, 1);
	dim3 grid2(blockNum2, 1, 1);

	// 每个线程处理d_points1中的一个特征点，遍历d_points2中的所有特征点
	//eulerKernalLaunch << <grid1, block1 >> >(descriptor1, descriptor2, d_points1, d_points2, pointsNum1, pointsNum2, gridX1, gridX2, d_results);
	// 与上述正好相反，每个线程处理d_points2中的一个特征点，遍历d_points1中的所有特征点
	//eulerKernalLaunch << <grid2, block2 >> >(descriptor2, descriptor1, d_points2, d_points1, pointsNum2, pointsNum1, gridX2, gridX1, d_results + (pointsNum1 * 2));

	eulerKernalLaunchNoshm << <grid1, block1 >> >(descriptor1, descriptor2, d_points1, d_points2, pointsNum1, pointsNum2, gridX1, gridX2, d_results);
	eulerKernalLaunchNoshm << <grid2, block2 >> >(descriptor2, descriptor1, d_points2, d_points1, pointsNum2, pointsNum1, gridX2, gridX1, d_results + (pointsNum1 * 2));

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("kernel() failed to launch error = %d\n", error);
	}
}

#endif