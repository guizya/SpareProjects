#ifndef _LSSPARALLEL_CU_
#define _LSSPARALLEL_CU_

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "helper_cuda.h"
#include "LssParallel.h"

__global__ void kernel(
	unsigned char *d_image,
	float *d_points,
	unsigned char *d_mapping,
	float *d_results,
	unsigned int *d_pointsNum,
	unsigned int imageX,
	unsigned int imageY
)
{
	int block_id = gridDim.x * blockIdx.y + blockIdx.x;
	// N 是属于这个block的特征点的个数
	int N = d_pointsNum[block_id];
	__syncthreads();
	// 如果属于这个block的特征点个数为0，属于当前block的线程直接退出
	// REQUIRE: N <= POINTS_IN_BLOCK
	if (N == 0) return;

	// 计算当前block的图像区域的起始位置
	int imageBaseX = BLOCK_PIXEL_X * blockIdx.x;
	int imageBaseY = BLOCK_PIXEL_Y * blockIdx.y;

	// 声明用于加载几何坐标到极坐标转换矩阵的shared memory
	__shared__ unsigned char mappings[LARGE_SIZE][LARGE_SIZE];

	// 将几何坐标到极坐标的转换矩阵加载进shared memory
	int iteration = (LARGE_SIZE * LARGE_SIZE) / blockDim.x;
	iteration += ((LARGE_SIZE * LARGE_SIZE) % blockDim.x) == 0 ? 0 : 1;
	for (int i = 0; i < iteration; i++) {
		int index = threadIdx.x * iteration + i;
		int shmIdxX = index % LARGE_SIZE;
		int shmIdxY = index / LARGE_SIZE;

		if (shmIdxY >= LARGE_SIZE) continue;

		int mappingIdx = shmIdxY * LARGE_SIZE + shmIdxX;
		mappings[shmIdxY][shmIdxX] = d_mapping[mappingIdx];
	}
	__syncthreads();

	// 如果当前线程的id超过了当前block特征点的个数，线程退出
	int tId = threadIdx.x;
	if (tId >= N) return;

	// FIXME: what if directly use the global memory?
	float descriptor[80];
	// 初始化descriptor的初始值，后面用于求最小值操作
	for (int i = 0; i < 80; i++) descriptor[i] = 255 * 255 * SMALL_SIZE * SMALL_SIZE;

	// 加载当前线程的特征点的位置
	int pointX = d_points[POINTS_IN_BLOCK * block_id * 2 + tId * 2]     - imageBaseX + 0.5;
	int pointY = d_points[POINTS_IN_BLOCK * block_id * 2 + tId * 2 + 1] - imageBaseY + 0.5;

	// 计算当前线程特征点的ssd
	//float var_noise = 0;
	float var_noise = 1000.f;
	int NUM = 0;
	for (int j = -LARGE_RADIUS; j <= LARGE_RADIUS; j++) for (int i = -LARGE_RADIUS; i <= LARGE_RADIUS; i++) {
		float ssd = 0;
		// 得到极坐标中的索引
		int index = mappings[LARGE_RADIUS + j][LARGE_RADIUS + i];

		for (int jj = -SMALL_RADIUS; jj <= SMALL_RADIUS; jj++) for (int ii = -SMALL_RADIUS; ii <= SMALL_RADIUS; ii++) {
			int pImageX = pointX + ii;
			int pImageY = pointY + jj;
			int pointImage = d_image[(imageBaseY + pImageY) * imageX + imageBaseX + pImageX];

			int shmImageX = pImageX + i;
			int shmImageY = pImageY + j;
			int shmImage = d_image[(imageBaseY + shmImageY) * imageX + imageBaseX + shmImageX];

			int temp = (int)shmImage - (int)pointImage;
			ssd += temp * temp;
		}

		// 计算ssd中的最小值
		descriptor[index] = descriptor[index] < ssd ? descriptor[index] : ssd;

		// 计算var_noise
		if (j >= -1 && j <= 1 && i >= -1 && i <= 1) {
		//if (j == 1 && i == 1) {
			var_noise = var_noise > ssd ? var_noise : ssd;
		}
	}

	// 计算exp，并记录极小值的个数
	var_noise = -1.0f / var_noise;
	int lowerNum = 0;
	for (int i = 0; i < 80; i++) {
		float result = expf(descriptor[i] * var_noise);
		if (result < 0.01) lowerNum++;

		//if (lowerNum > 24) break;
		descriptor[i] = result;
	}

	// 如果极小值过多，则忽略当前特征点的descriptor
	if (lowerNum > 24) return;

	// 写回属于当前线程特征点的descriptor
	int base = block_id * POINTS_IN_BLOCK * 80 + tId * 80;
	for (int i = 0; i < 80; i++) d_results[base + i] = descriptor[i];

	return;
}

extern "C"
void lssKernal(unsigned int imageX, unsigned int imageY, unsigned int gridX, unsigned int gridY,
	unsigned char *d_image,
	float *d_points,
	unsigned char *d_mapping,
	float *d_results,
	unsigned int *d_pointsNum)
{
	cudaError_t error = cudaSuccess;

	dim3 block(POINTS_IN_BLOCK, 1, 1);
	dim3 grid(gridX, gridY, 1);

	kernel << <grid, block >> >(d_image, d_points, d_mapping, d_results, d_pointsNum, imageX, imageY);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("kernel() failed to launch error = %d\n", error);
	}
}

#endif