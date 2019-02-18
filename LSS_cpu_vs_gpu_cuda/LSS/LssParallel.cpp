
// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "LssParallel.h"
#include "helper_cuda.h"
#include <ctime>


void LssParallelStruct::lssPreparation(unsigned int imageX, unsigned int imageY, Mat& image, vector<KeyPoint>& points)
{
	SelfSimDescriptor sd(SMALL_SIZE, LARGE_SIZE);
	sd.computeLogPolarMapping(mappingMask);

	// 将图像切分为多个 BLOCK_PIXEL_X x BLOCK_PIXEL_Y 大小的图像区域
	// 每个block处理一个 BLOCK_PIXEL_X x BLOCK_PIXEL_Y 大小的图像区域
	// 统计属于每个block的特征点的位置和数目
	// block的每个线程都处理一个特征点
	// 每个block的线程数目最大限制为 POINTS_IN_BLOCK
	// 这样的话，每个block需要加载的图像大小为 (BLOCK_PIXEL_Y + 2 * RADIUS) x (BLOCK_PIXEL_X + 2 * RADIUS)
	int realImageX = imageX - 2 * RADIUS;
	int realImageY = imageY - 2 * RADIUS;
	gridX = ceil((float)realImageX / BLOCK_PIXEL_X);
	gridY = ceil((float)realImageY / BLOCK_PIXEL_Y);

	// 用于重新排列特征点的数据
	// 每个block在数组里面占据 (POINTS_IN_BLOCK * 2)
	// 后面将每个特征点置于属于它的block的区间内
	h_points = new float[gridX * gridY * POINTS_IN_BLOCK * 2]();
	// 用于统计属于每个block 特征点的个数
	h_pointsNum = new unsigned int[gridX * gridY]();
	// 用于存放最后结果
	// 每个block的特征点的descriptor在数组里面占据 (POINTS_IN_BLOCK * 80)
	h_results = new float[gridX * gridY * POINTS_IN_BLOCK * 80]();

	// 将特征点的位置重新排列，并统计属于每个block的特征点个数
	// REQUIRE: points in the border have already been filtered out
	int maxnum = 0;
	int size = points.size();
	for (int i = 0; i < size; i++) {
		int x = points[i].pt.y + 0.5;
		int y = points[i].pt.x + 0.5;

		int cta_id_x = (x - RADIUS) / BLOCK_PIXEL_X;
		int cta_id_y = (y - RADIUS) / BLOCK_PIXEL_Y;
		int cta_id = cta_id_y * gridX + cta_id_x;

		int num = h_pointsNum[cta_id] ++;
		maxnum = maxnum > h_pointsNum[cta_id] ? maxnum : h_pointsNum[cta_id];
		if (num >= POINTS_IN_BLOCK) {
			throw std::runtime_error("the number of key points in each cta pixel region should not exceed POINTS_IN_BLOCK");
		}

		h_points[cta_id * POINTS_IN_BLOCK * 2 + num * 2] = points[i].pt.y;
		h_points[cta_id * POINTS_IN_BLOCK * 2 + num * 2 + 1] = points[i].pt.x;
	}
	//std::cout << "Max num is " << maxnum << std::endl;

	// 将图像的像素信息重新排列
	h_image = new unsigned char[imageX * imageY]();
	for (int i = 0; i < imageX; i++) for (int j = 0; j < imageY; j++) {
		int index = j * imageX + i;
		h_image[index] = image.at<unsigned char>(i, j);
	}
	// 将几何坐标到极坐标的转换矩阵重新排列
	h_mapping = new unsigned char[LARGE_SIZE * LARGE_SIZE]();
	for (int i = 0; i < LARGE_SIZE; i++) for (int j = 0; j < LARGE_SIZE; j++) {
		int index = j * LARGE_SIZE + i;
		h_mapping[index] = mappingMask.at<unsigned char>(i, j);
	}

	// 初始化 GPU 上的内存空间
	checkCudaErrors(cudaMalloc((void **)&d_points, gridX * gridY * POINTS_IN_BLOCK * 2 * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void **)&d_pointsNum, gridX * gridY * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void **)&d_results, gridX * gridY * POINTS_IN_BLOCK * 80 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_image, imageX * imageY * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void **)&d_mapping, LARGE_SIZE * LARGE_SIZE * sizeof(unsigned char)));

	// 将上述准备的数据传输到 GPU
	checkCudaErrors(cudaMemset(d_results, 0, gridX * gridY * POINTS_IN_BLOCK * 80 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_points, h_points, gridX * gridY * POINTS_IN_BLOCK * 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_pointsNum, h_pointsNum, gridX * gridY * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_image, h_image, imageX * imageY * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mapping, h_mapping, LARGE_SIZE * LARGE_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));
}

Mat LssParallelStruct::lssParallel(unsigned int imageX, unsigned int imageY, vector<KeyPoint>& points)
{
	// 调用并行内核
	lssKernal(imageX, imageY, gridX, gridY, d_image, d_points, d_mapping, d_results, d_pointsNum);

	checkCudaErrors(cudaDeviceSynchronize());
	// 将结果从 GPU 传输到 CPU
	checkCudaErrors(cudaMemcpy(h_results, d_results, gridX * gridY * POINTS_IN_BLOCK * 80 * sizeof(float), cudaMemcpyDeviceToHost));

	// 分析结果，并将结果重排
	// 最后得到属于每个特征点的descriptor
	memset(h_pointsNum, 0, gridX * gridY * sizeof(unsigned int));
	Mat finalResult;
	for (int i = 0; i < points.size(); i++) {
		int x = points[i].pt.y + 0.5;
		int y = points[i].pt.x + 0.5;

		int cta_id_x = (x - RADIUS) / BLOCK_PIXEL_X;
		int cta_id_y = (y - RADIUS) / BLOCK_PIXEL_Y;
		int cta_id = cta_id_y * gridX + cta_id_x;

		int num = h_pointsNum[cta_id] ++;
		int offset = cta_id * POINTS_IN_BLOCK * 80 + num * 80;

		Mat temp(1, 80, CV_32F);
		for (int j = 0; j < 80; j++) temp.at<float>(0, j) = h_results[offset + j];
		finalResult.push_back(temp);
	}

	return finalResult;
}

void LssParallelStruct::lssRelease()
{
	// 析构掉GPU上的内存空间
	checkCudaErrors(cudaFree(d_points));
	checkCudaErrors(cudaFree(d_pointsNum));
	checkCudaErrors(cudaFree(d_results));
	checkCudaErrors(cudaFree(d_image));
	checkCudaErrors(cudaFree(d_mapping));

	// 析构掉CPU上的内存空间
	delete[]h_points;
	delete[]h_pointsNum;
	delete[]h_results;
	delete[]h_image;
	delete[]h_mapping;
}

void LssParallelStruct::dumpResult(Mat results, vector<KeyPoint> keyPoints, ofstream& out) {
	for (int i = 0; i < results.rows; i++) {
		out << "Key point " << i << " coordinates: " << keyPoints[i].pt.x << " " << keyPoints[i].pt.y << std::endl;
		for (int j = 0; j < results.cols; j++) {
			out << results.at<float>(i, j) << " ";
		}
		out << std::endl;
	}
}

double LssParallelStruct::calculateDiff(Mat results1, Mat results2)
{
	double result = 0;
	for (int i = 0; i < results1.rows; i++) {
		for (int j = 0; j < results1.cols; j++) {
			double temp = results1.at<float>(i, j) - results2.at<float>(i, j);
			temp = temp * temp;
			if (temp > 0.9) std::cout << "Error happens at " << i << " " << temp << std::endl;
			result += temp;
		}
	}
	return result;
}

