#include "Euler.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include <ctime>

__global__ void eulerKernel(
	float *descriptor1,
	float *descriptor2,
	float *d_points1,
	float *d_points2,
	unsigned int pointsNum1,
	unsigned int pointsNum2,
	float *results)
{
	__shared__ float shmDesX[BLOCK_POINT_SIZE_X][80];
	__shared__ float shmDesY[BLOCK_POINT_SIZE_Y][80];

	if (blockIdx.x * BLOCK_POINT_SIZE_X + threadIdx.x >= pointsNum1) return;
	if (blockIdx.y * BLOCK_POINT_SIZE_Y + threadIdx.y >= pointsNum2) return;

	int xIndex = blockIdx.x * BLOCK_POINT_SIZE_X + threadIdx.x;
	int yIndex = blockIdx.y * BLOCK_POINT_SIZE_Y + threadIdx.y;
	unsigned int offset = xIndex * pointsNum2 + yIndex;

	// block has 8 * 8 = 64 threads
	// 属于第一幅图片的特征点的描述符只有8组
	// 只需要8个线程做加载的事情
	// 第二幅图片类似

	// 将第一幅图片的描述符信息加载进shared memory
	if (threadIdx.y == 0) {
		for (int j = 0; j < 80; j++) {
			shmDesX[threadIdx.x][j] = descriptor1[xIndex * 80 + j];
		}
	}

	// 将第二幅图片的描述符信息加载进shared memory
	if (threadIdx.x == 0) {
		for (int j = 0; j < 80; j++) {
			shmDesY[threadIdx.y][j] = descriptor2[yIndex * 80 + j];
		}
	}
	__syncthreads();

	float x = d_points1[xIndex * 2];
	float y = d_points1[xIndex * 2 + 1];

	float xx = d_points2[yIndex * 2];
	float yy = d_points2[yIndex * 2 + 1];

	float xxx = x - xx;
	float yyy = y - yy;
	int size = 20;
	if (xxx >= size || xxx <= -size || yyy >= size || yyy <= -size) {
		results[offset] = 1e10;
		return;
	}

	// 计算欧式距离，并将结果写进一个pointsNum1 * pointsNum2的二维数组
	float dist = 0;
	for (int i = 0; i < 80; i++) {
		dist += (shmDesX[threadIdx.x][i] - shmDesY[threadIdx.y][i]) * (shmDesX[threadIdx.x][i] - shmDesY[threadIdx.y][i]);
	}
	results[offset] = sqrt(dist);
}

__global__ void eulerKernel1(
	float *eulerResults,
	unsigned int pointsNum,
	bool vertical,
	unsigned int pointsNum1,
	unsigned int pointsNum2,
	unsigned int *results)
{
	unsigned globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if (globalId >= pointsNum) return;

	float minDist = 1e10;
	float secondDist = 1e10;
	int matchingIndex = -1;
	if (!vertical) {
		// 为第一幅图片计算最近和次近欧氏距离，横向遍历欧氏距离二维数组
		for (int i = 0; i < pointsNum2; i++) {
			if (eulerResults[globalId * pointsNum2 + i] < minDist) {
				secondDist = minDist;
				minDist = eulerResults[globalId * pointsNum2 + i];
				matchingIndex = i;
			}
			else if (eulerResults[globalId * pointsNum2 + i] < secondDist) {
				secondDist = eulerResults[globalId * pointsNum2 + i];
			}
		}
	}
	else {
		// 为第二幅图片计算最近和次近欧氏距离，纵向遍历欧氏距离二维数组
		for (int i = 0; i < pointsNum1; i++) {
			if (eulerResults[i * pointsNum2 + globalId] < minDist) {
				secondDist = minDist;
				minDist = eulerResults[i * pointsNum2 + globalId];
				matchingIndex = i;
			}
			else if (eulerResults[i * pointsNum2 + globalId] < secondDist) {
				secondDist = eulerResults[i * pointsNum2 + globalId];
			}
		}
	}

	// 将匹配的特征点索引写进memory
	if ((minDist / secondDist < 0.95) && matchingIndex != -1) results[globalId] = matchingIndex;
}

__global__ void eulerKernelNoShm(
	float *descriptor1,
	float *descriptor2,
	float *d_points1,
	float *d_points2,
	unsigned int pointsNum1,
	unsigned int pointsNum2,
	float *results)
{
	int xIndex = blockIdx.x * BLOCK_POINT_SIZE_X + threadIdx.x;
	int yIndex = blockIdx.y * BLOCK_POINT_SIZE_Y + threadIdx.y;

	if (xIndex >= pointsNum1) return;
	if (yIndex >= pointsNum2) return;

	unsigned int offset = xIndex * pointsNum2 + yIndex;

	float x = d_points1[xIndex * 2];
	float y = d_points1[xIndex * 2 + 1];

	float xx = d_points2[yIndex * 2];
	float yy = d_points2[yIndex * 2 + 1];

	float xxx = x - xx;
	float yyy = y - yy;
	int size = 20;
	if (xxx >= size || xxx <= -size || yyy >= size || yyy <= -size) {
		results[offset] = 1e10;
		return;
	}

	// 计算欧式距离，并将结果写进一个pointsNum1 * pointsNum2的二维数组
	float dist = 0;
	for (int i = 0; i < 80; i++) {
		dist += (descriptor1[xIndex * 80 + i] - descriptor2[yIndex * 80 + i]) * (descriptor1[xIndex * 80 + i] - descriptor2[yIndex * 80 + i]);
	}
	results[offset] = sqrt(dist);
}

extern "C"
void euler()
{
	cudaError_t error = cudaSuccess;

	dim3 block1(BLOCK_POINT_SIZE_X, BLOCK_POINT_SIZE_Y, 1);
	unsigned int blockNum1 = ceil((float)pointsCount1 / BLOCK_POINT_SIZE_X);
	unsigned int blockNum2 = ceil((float)pointsCount2 / BLOCK_POINT_SIZE_Y);

	dim3 grid1(blockNum1, blockNum2, 1);

	dim3 block2(64, 1, 1);
	unsigned int blockNum3 = ceil((float)pointsCount1 / block2.x);
	dim3 grid2(blockNum3, 1, 1);

	unsigned int blockNum4 = ceil((float)pointsCount2 / block2.x);
	dim3 grid3(blockNum4, 1, 1);

	//eulerKernel << <grid1, block1 >> >(d_descriptor1, d_descriptor2, d_points1, d_points2, pointsCount1, pointsCount2, d_EulerResults);
	eulerKernelNoShm << <grid1, block1 >> >(d_descriptor1, d_descriptor2, d_points1, d_points2, pointsCount1, pointsCount2, d_EulerResults);
	checkCudaErrors(cudaDeviceSynchronize());

	eulerKernel1 << <grid2, block2 >> >(d_EulerResults, pointsCount1, false, pointsCount1, pointsCount2, d_results1);
	eulerKernel1 << <grid3, block2 >> >(d_EulerResults, pointsCount2, true, pointsCount1, pointsCount2, d_results2);
	checkCudaErrors(cudaDeviceSynchronize());
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("kernel() failed to launch error = %d\n", error);
	}
}