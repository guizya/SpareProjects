#ifndef _mosaic_h_
#define _mosaic_h_

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

using namespace std;
using namespace cv;

/*�ú����γ�����ͼ�������ͼ�������ں�����������ͼ��*/
void mosaic_map(const Mat &image_1, const Mat &image_2, Mat &chessboard_1, Mat &chessboard_2, Mat &mosaic_image, int width);

#endif