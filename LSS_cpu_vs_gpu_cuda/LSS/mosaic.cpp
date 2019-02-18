#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include "mosaic.h"

using namespace std;
using namespace cv;


/********************该函数生成两幅图的棋盘网格图*************************/
/*image_1表示参考图像
 image_2表示配准后的待叠加图像
 chessboard_1表示image_1的棋盘图像
 chessboard_2表示image_2的棋盘图像
 mosaic_image表示image_1和image_2的镶嵌图像
 width表示棋盘网格大小
 */
void mosaic_map(const Mat &image_1, const Mat &image_2, Mat &chessboard_1, Mat &chessboard_2, Mat &mosaic_image, int width)
{
	if (image_1.size != image_2.size)
		CV_Error(CV_StsBadArg, "mosaic_map模块输入两幅图大小必须一致！");

	//生成image_1的棋盘网格图
	chessboard_1 = image_1.clone();
	int rows_1 = chessboard_1.rows;
	int cols_1 = chessboard_1.cols;

	int row_grids_1 = cvFloor((double)rows_1 / width);//行方向网格个数
	int col_grids_1 = cvFloor((double)cols_1 / width);//列方向网格个数

	for (int i = 0; i < row_grids_1; i = i + 2){
		for (int j = 1; j < col_grids_1; j = j + 2){
			Range range_x(j*width, (j + 1)*width);
			Range range_y(i*width, (i + 1)*width);
			chessboard_1(range_y, range_x) = 0;
		}
	}

	for (int i = 1; i < row_grids_1; i = i + 2){
		for (int j = 0; j < col_grids_1; j = j + 2){
			Range range_x(j*width, (j + 1)*width);
			Range range_y(i*width, (i + 1)*width);
			chessboard_1(range_y, range_x) = 0;
		}
	}

	//生成image_2的棋盘网格图
	chessboard_2 = image_2.clone();
	int rows_2 = chessboard_2.rows;
	int cols_2 = chessboard_2.cols;

	int row_grids_2 = cvFloor((double)rows_2 / width);//行方向网格个数
	int col_grids_2 = cvFloor((double)cols_2 / width);//列方向网格个数

	for (int i = 0; i < row_grids_2; i = i + 2){
		for (int j = 0; j < col_grids_2; j = j + 2){
			Range range_x(j*width, (j + 1)*width);
			Range range_y(i*width, (i + 1)*width);
			chessboard_2(range_y, range_x) = 0;
		}
	}

	for (int i = 1; i < row_grids_2; i = i + 2){
		for (int j = 1; j < col_grids_2; j = j + 2){
			Range range_x(j*width, (j + 1)*width);
			Range range_y(i*width, (i + 1)*width);
			chessboard_2(range_y, range_x) = 0;
		}
	}
	mosaic_image = chessboard_1 + chessboard_2;
}