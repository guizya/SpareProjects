#ifndef _match_h_
#define _match_h_

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

const double dis_ratio = 0.95;//最近邻和次近邻距离比阈值

/*该函数用于图像之间的特征点描述子匹配*/
vector<pair<KeyPoint, KeyPoint> > MatchDes(const Mat Des1,const Mat Des2,vector<KeyPoint>keypoints1,vector<KeyPoint>keypoints2);

/*该函数根据最小均方误差原则，计算变换矩阵*/
static Mat LMS(const Mat&points_1, const Mat &points_2, string model, float &rmse);

/*该函数使用ransac算法删除错误匹配点对*/
Mat ransac(const vector<Point2f>&points_1, const vector<Point2f> &points_2, string model, float threshold, vector<bool> &inliers, float &rmse);

/*该函数把两幅配准后的图像进行融合镶嵌*/
void image_fusion(const Mat &image_1, const Mat &image_2, const Mat T, Mat &fusion_image);

#endif
