#include "GpuMethods.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include <iostream>

using namespace std;

int GPU::SM = 0;
int GPU::SHM = 0;
int GPU::ParallelThreads = 0;
int GPU::ThreadsPerSM = 0;
int GPU::TotalThreads = 0;
float* GPU::d_points = 0;
float* GPU::d_kmeans = 0;
int* GPU::d_clusters = 0;
int* GPU::d_counts = 0;
int* GPU::d_flags = 0;
float* GPU::d_debug = 0;

extern "C"
void kmeansKernal(unsigned int gridX, unsigned int blockX, unsigned iter);

void GPU::analyze() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driver_version);
		printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
		printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
		printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
		printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
		printf("Warp size:                                      %d\n", deviceProp.warpSize);
		printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
		printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
		printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
		printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
		printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);

		SM = deviceProp.multiProcessorCount;
		SHM = deviceProp.sharedMemPerBlock;
		TotalThreads = deviceProp.maxThreadsPerMultiProcessor * SM;
	}
}

void GPU::initialize()
{
	ParallelThreads = TotalThreads > N ? N : TotalThreads;
	ThreadsPerSM = (ParallelThreads % SM == 0) ? (ParallelThreads / SM) : (ParallelThreads / SM + 1);
	ThreadsPerSM = (ThreadsPerSM < 32) ? 32 : ThreadsPerSM;

	checkCudaErrors(cudaMalloc((void **)&d_points, N * D * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_kmeans, K * D * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_clusters, N * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_counts, K * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_flags, 4 * MAXLOOP * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_debug, 1000 * sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_points, &points[0], N * D * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_kmeans, &kmeans[0], K * D * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_clusters, 0, N * sizeof(int)));
	checkCudaErrors(cudaMemset(d_counts, 0, K * sizeof(int)));
	checkCudaErrors(cudaMemset(d_flags, 0, 4 * MAXLOOP * sizeof(int)));
	checkCudaErrors(cudaMemset(d_debug, 0, 1000 * sizeof(float)));
}

void GPU::initMem()
{
	//int KK = 256;
	//checkCudaErrors(cudaMalloc((void **)&d_points, N * D * sizeof(float)));
	//checkCudaErrors(cudaMalloc((void **)&d_kmeans, KK * D * sizeof(float)));
	//checkCudaErrors(cudaMalloc((void **)&d_clusters, N * sizeof(int)));
	//checkCudaErrors(cudaMalloc((void **)&d_counts, KK * sizeof(int)));
	//checkCudaErrors(cudaMalloc((void **)&d_flags, 4 * MAXLOOP * sizeof(int)));
	//checkCudaErrors(cudaMalloc((void **)&d_debug, 1000 * sizeof(float)));
}

void GPU::freeMem()
{

}

void GPU::free() 
{
	checkCudaErrors(cudaMemcpy(&kmeans[0], d_kmeans, K * D * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&clusters[0], d_clusters, N * sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_points));
	checkCudaErrors(cudaFree(d_kmeans));
	checkCudaErrors(cudaFree(d_clusters));
	checkCudaErrors(cudaFree(d_counts));
	checkCudaErrors(cudaFree(d_flags));
	checkCudaErrors(cudaFree(d_debug));
}

void GPU::kmeansGPU()
{
	unsigned int blockX = ThreadsPerSM > 1024 ? 1024 : ThreadsPerSM;
	unsigned int gridX = (ParallelThreads % blockX == 0) ? (ParallelThreads / blockX) : (ParallelThreads / blockX + 1);
	unsigned int iterationCount = (N % ParallelThreads == 0) ? (N / ParallelThreads) : (N / ParallelThreads + 1);
	
	//std::cout << "TotalThreads " << TotalThreads << " ParallelThreads " << ParallelThreads << " ThreadsPerSM " << ThreadsPerSM << endl;
	//cout << "blockX " << blockX << " gridX " << gridX << " iter " << iterationCount << endl;

	kmeansKernal(gridX, blockX, iterationCount);

	checkCudaErrors(cudaDeviceSynchronize());
	//float number[1000];
	//checkCudaErrors(cudaMemcpy(&number, d_debug, 1000 * sizeof(float), cudaMemcpyDeviceToHost));
	//cout << "Number " << number[0] << endl;
	//cout << "Numbers ";
	//for (int i = 0; i < 100; i++)
	//	cout << number[i] << " ";
	//cout << endl;
}