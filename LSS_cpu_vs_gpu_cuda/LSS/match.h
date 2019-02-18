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

const double dis_ratio = 0.95;//����ںʹν��ھ������ֵ

/*�ú�������ͼ��֮���������������ƥ��*/
vector<pair<KeyPoint, KeyPoint> > MatchDes(const Mat Des1,const Mat Des2,vector<KeyPoint>keypoints1,vector<KeyPoint>keypoints2);

/*�ú���������С�������ԭ�򣬼���任����*/
static Mat LMS(const Mat&points_1, const Mat &points_2, string model, float &rmse);

/*�ú���ʹ��ransac�㷨ɾ������ƥ����*/
Mat ransac(const vector<Point2f>&points_1, const vector<Point2f> &points_2, string model, float threshold, vector<bool> &inliers, float &rmse);

/*�ú�����������׼���ͼ������ں���Ƕ*/
void image_fusion(const Mat &image_1, const Mat &image_2, const Mat T, Mat &fusion_image);

#endif
