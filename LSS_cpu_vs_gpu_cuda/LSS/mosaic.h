#ifndef _mosaic_h_
#define _mosaic_h_

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

using namespace std;
using namespace cv;

/*该函数形成两幅图像的棋盘图，并且融合这两幅棋盘图像*/
void mosaic_map(const Mat &image_1, const Mat &image_2, Mat &chessboard_1, Mat &chessboard_2, Mat &mosaic_image, int width);

#endif