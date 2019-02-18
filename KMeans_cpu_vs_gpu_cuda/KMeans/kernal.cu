#ifndef _KERNEL_CU_
#define _KERNEL_CU_

#include "GpuMethods.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

__device__ void syncAllThreads(int *flags, int threads)
{
	__syncthreads();
	while (true) {
		//__threadfence_system();
		__threadfence();
		if (flags[0] == threads) break;
	}
}

__global__ void kernel(
	float *d_Points,
	float *d_Kmeans,
	int *d_Clusters,
	int *d_Counts,
	int *d_Flags,
	unsigned int iter,
	unsigned int iter1,
	int d_N,
	int d_K,
	int d_D,
	int d_totalThreads  //,
//	float *d_debug
)
{
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	
	// suppose this is only one iteration first
	if (threadId >= d_N) return;

	// REQUIREMENT: (K * D) < (48KB (shared memory size in chip) / 4 (sizeof float)/ 2 (blocks)) = 6000, K < ParallelThreads
	__shared__ float meansSHM[256 * 4];

	// here for the algorithm loop
	for (int loop = 0; loop < MAXLOOP; loop ++) {

		for (int i = 0; i < iter1; i++) {
			int tempTid = threadIdx.x * iter1 + i;
			if (tempTid < d_K)
				for (int dim = 0; dim < d_D; dim++) {
					meansSHM[tempTid * d_D + dim] = d_Kmeans[tempTid * d_D + dim];
				}
		}

		// SYNC ALL THREADS
		__syncthreads();
		atomicAdd(d_Flags + loop * 4 + 0, 1);
		syncAllThreads(d_Flags + loop * 4 + 0, d_totalThreads);

		for (int i = 0; i < iter; i++) {
			int tempTid = threadId * iter + i;
			if (tempTid >= d_N) break;

			// REQUIREMENT: D <= 4
			float data[4];
			// fetch point data for this thread from global memory
			for (int dim = 0; dim < d_D; dim++)
				data[dim] = d_Points[tempTid * d_D + dim];

			// find the minimal cluster for this point
			float minimal = 1E8;
			int minIdx = -1;
			for (int cluster = 0; cluster < d_K; cluster++) {
				float length = 0;
				for (int dim = 0; dim < d_D; dim++) {
					float t = (data[dim] - meansSHM[cluster * d_D + dim]);
					length += t * t;
				}

				if (length < minimal) {
					minimal = length;
					minIdx = cluster;
				}
			}
			
			if (d_Clusters[tempTid] != minIdx) {
				d_Clusters[tempTid] = minIdx;
				atomicAdd(d_Flags + loop * 4 + 1, 1);
			}

			// write results to global memory for the next iteration
			for (int dim = 0; dim < d_D; dim++)
				atomicAdd(d_Kmeans + minIdx * d_D + dim, data[dim]);
			atomicAdd(d_Counts + minIdx, 1);
		}

		// SYNC ALL THREADS again
		__syncthreads();
		atomicAdd(d_Flags + loop * 4 + 2, 1);
		syncAllThreads(d_Flags + loop * 4 + 2, d_totalThreads);

		// after syncing all threads here
		// check if we need to iterate again
		if (d_Flags[loop * 4 + 1] == 0) break;

		// update the cluster data in global memory
		if (threadId < d_K) {
			int count = d_Counts[threadId];
			float value1 = (count != 0) ? (1.0f / (float)count) : 0;
			for (int dim = 0; dim < d_D; dim++) {
				float value = d_Kmeans[threadId * d_D + dim] - meansSHM[threadId * d_D + dim];
				d_Kmeans[threadId * d_D + dim] = value * value1;
			}
			d_Counts[threadId] = 0;
		}

		// SYNC ALL THREADS again
		__syncthreads();
		atomicAdd(d_Flags + loop * 4 + 3, 1);
		syncAllThreads(d_Flags + loop * 4 + 3, d_totalThreads);
	}
}

extern "C"
void kmeansKernal(unsigned int gridX, unsigned int blockX, unsigned iter)
{
	cudaError_t error = cudaSuccess;

	dim3 block(blockX, 1, 1);
	dim3 grid(gridX, 1, 1);
	//cout << "blockX " << blockX << " gridX " << gridX << " iter " << iter << endl;

	int iter1 = (K % blockX == 0) ? (K / blockX) : (K / blockX + 1);
	iter1 = (iter1 == 0 ? 1 : iter1);

	//kernel << <grid, block >> >(GPU::d_points, GPU::d_kmeans, GPU::d_clusters, GPU::d_counts, GPU::d_flags, iter, iter1, N, K, D, GPU::ParallelThreads, GPU::d_debug);
	kernel << <grid, block >> >(GPU::d_points, GPU::d_kmeans, GPU::d_clusters, GPU::d_counts, GPU::d_flags, iter, iter1, N, K, D, GPU::ParallelThreads);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("kernel() failed to launch error = %d\n", error);
	}
}

#endif _KERNEL_CU_