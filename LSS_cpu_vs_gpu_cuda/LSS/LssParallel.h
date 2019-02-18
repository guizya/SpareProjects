#ifndef _LSSPARALLEL_h_
#define _LSSPARALLEL_h_
#include <fstream>
#include "opencv2/opencv.hpp"  


#define BLOCK_PIXEL_X 250
#define BLOCK_PIXEL_Y 100
#define	BLOCK_OFFSET (BLOCK_PIXEL_X * BLOCK_PIXEL_Y * 2)
#define POINTS_IN_BLOCK 256
#define PIXELS_IN_BLOCK ((BLOCK_PIXEL_X + 2 * RADIUS) * (BLOCK_PIXEL_Y + 2 * RADIUS))
#define SMALL_SIZE 5
#define LARGE_SIZE 41
#define SMALL_RADIUS 2
#define LARGE_RADIUS 20
#define RADIUS 22
#define DESCRIPTOR_NUM 50
#define MAX_POINTS_NUM 1000

using namespace cv;
using namespace std;

extern "C"
void lssKernal(unsigned int imageX, unsigned int imageY, unsigned int gridX, unsigned int gridY,
	unsigned char *d_image,
	float *d_points,
	unsigned char *d_mapping,
	float *d_results,
	unsigned int *d_pointsNum);

struct LssParallelStruct {
public:
	LssParallelStruct(unsigned int imageX, unsigned int imageY, Mat& image, vector<KeyPoint>& points) {
		lssPreparation(imageX, imageY, image, points);
	};
	~LssParallelStruct() { lssRelease(); };
	void lssPreparation(unsigned int imageX, unsigned int imageY, Mat& image, vector<KeyPoint>& points);
	Mat lssParallel(unsigned int imageX, unsigned int imageY, vector<KeyPoint>& points);
	void lssRelease();
	void dumpResult(Mat results, vector<KeyPoint> keyPoints, ofstream& out);
	double calculateDiff(Mat results1, Mat results2);

	float * d_results;
	unsigned char *d_image, *d_mapping;
	unsigned int  *d_pointsNum;

	unsigned char *h_image;
	unsigned char *h_mapping;
	unsigned int  *h_pointsNum;
	float * h_points, *d_points;
	float *h_results;
	Mat mappingMask;
	unsigned gridX;
	unsigned gridY;
};

#endif