#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <algorithm>
#include <vector>
#include <cmath>
#include "match.h"

using namespace std;
using namespace cv;

/********�ú�������������������ŷʽ���룬�����ͼ��֮���ƥ����********/
/*Des1��ʾ�ο�ͼ�������������Ӽ���,
 Des2��ʾ����׼ͼ�������������Ӽ���
 keypoints1��ʾ�ο�ͼ�������㼯��
 keypoints2��ʾ����׼ͼ�������㼯��
 ����ֵƥ��������Լ���
 */
vector<pair<KeyPoint, KeyPoint> > MatchDes(const Mat Des1,const Mat Des2,vector<KeyPoint>keypoints1,vector<KeyPoint>keypoints2){
	  float d1,d2;
	  vector<pair<KeyPoint, KeyPoint> > matches;
	  KeyPoint *match = 0;
	  int maximal = 0;
	  matches.clear();
      for(int i=0;i!=keypoints1.size();++i)
	  {
		 d1 = d2 = FLT_MAX;
	     for(int j=0;j!=keypoints2.size();++j)
		 { 
		   float disparity_x=abs(keypoints1[i].pt.x-keypoints2[j].pt.x);
		   float disparity_y=abs(keypoints1[i].pt.y-keypoints2[j].pt.y);
		   float dist = 0;
		   if(disparity_x<20&&disparity_y<20)
			{
		      for(int k=0;k!=Des1.cols;++k)
			  { 
				 dist=dist+(Des1.at<float>(i,k)-Des2.at<float>(j,k))*(Des1.at<float>(i,k)-Des2.at<float>(j,k));
			  }
			  dist=sqrt(dist);
			  if(dist<d1) 
              {
                  d2 = d1;
                  d1 = dist;
                  match = &keypoints2[j];
				  maximal = j;
              }
              else if(dist<d2)
              {
                 d2 = dist;
              }
		   }
		 }

		 if(d1/d2 < dis_ratio)
         { 
             matches.push_back(std::make_pair(keypoints1[i], *match));
	 	 }
    } 
	return matches;
};
/********����С�����㷨����������ȷ��ƥ���ԣ������ͼ��֮��ı任��ϵ********/
/*match1_xy��ʾ�ο�ͼ�����������꼯��,[M x 2]����M��ʾ�����ĸ���
 match2_xy��ʾ����׼ͼ�������㼯�ϣ�[M x 2]����M��ʾ�����㼯��
 model��ʾ�任���ͣ������Ʊ任��,"����任","͸�ӱ任"
 rmse��ʾ���������
 ����ֵΪ����õ���3 x 3�������
 */
static Mat LMS(const Mat&match1_xy, const Mat &match2_xy, string model, float &rmse)
{

	if (match1_xy.rows != match2_xy.rows)
		cout<<"LMSģ������������Ը�����һ�£�"<<endl;

	if (!(model == string("affine") || model == string("similarity") || 
		model ==string("perspective")))
		cout<<"LMSģ��ͼ��任�����������"<<endl;

	const int N = match1_xy.rows;//���������
	
	Mat match2_xy_trans, match1_xy_trans;//����ת�ã�ÿ�д洢һ��
	transpose(match1_xy, match1_xy_trans);
	transpose(match2_xy, match2_xy_trans);

	Mat change = Mat::zeros(3, 3, CV_32FC1);

	//A*X=B,���������ַ���任��͸�ӱ任һ��,��������������M����A=[2*M,6]����
	//A=[x1,y1,0,0,1,0;0,0,x1,y1,0,1;.....xn,yn,0,0,1,0;0,0,xn,yn,0,1]
	Mat A = Mat::zeros(2*N,6,CV_32FC1);
	for (int i = 0; i < N; ++i)
	{
		A.at<float>(2 * i, 0) = match2_xy.at<float>(i, 0);//x
		A.at<float>(2 * i, 1) = match2_xy.at<float>(i, 1);//y
		A.at<float>(2*i, 4) = 1.f;

		A.at<float>(2 * i + 1, 2) = match2_xy.at<float>(i, 0);
		A.at<float>(2 * i + 1, 3) = match2_xy.at<float>(i, 1);
		A.at<float>(2 * i+1, 5) = 1.f;
	}

	//��������������M,�Ǹ�B=[2*M,1]����
	//B=[u1,v1,u2,v2,.....,un,vn]
	Mat B;
	B.create(2 * N, 1, CV_32FC1);
	for (int i = 0; i < N; ++i)
	{
		B.at<float>(2 * i, 0) = match1_xy.at<float>(i, 0);//x
		B.at<float>(2 * i + 1, 0) = match1_xy.at<float>(i, 1);//y
	}

	//����Ƿ���任
	if (model == string("affine"))
	{
		Vec6f values;
		solve(A, B, values, DECOMP_QR);
		change = (Mat_<float>(3,3)<<values(0), values(1), values(4),
			values(2), values(3), values(5),
			+0.0f, +0.0f, 1.0f);

		Mat temp_1 = change(Range(0, 2), Range(0, 2));//�߶Ⱥ���ת��
		Mat temp_2 = change(Range(0, 2), Range(2, 3));//ƽ����
		
		Mat match2_xy_change = temp_1 * match2_xy_trans + repeat(temp_2, 1, N);
		Mat diff = match2_xy_change - match1_xy_trans;//���
		//cout<<"ӳ�����"<<diff<<endl;
		pow(diff,2.f,diff);
		rmse = (float)sqrt(sum(diff)(0)*1.0/N);//sum����Ǹ���ͨ���ĺ�
	}
	//�����͸�ӱ任
	else if (model == string("perspective"))
	{
		/*͸�ӱ任ģ��
		[u'*w,v'*w, w]'=[u,v,w]' = [a1, a2, a5;
		                            a3, a4, a6;
		                            a7, a8, 1] * [x, y, 1]'
		[u',v']'=[x,y,0,0,1,0,-u'x, -u'y;
		         0, 0, x, y, 0, 1, -v'x,-v'y] * [a1, a2, a3, a4, a5, a6, a7, a8]'
		����Y = A*X     */

		Mat A2;
		A2.create(2 * N, 2, CV_32FC1);
		for (int i = 0; i < N; ++i)
		{
			A2.at<float>(2*i, 0) = match1_xy.at<float>(i, 0)*match2_xy.at<float>(i, 0)*(-1.f);
			A2.at<float>(2*i, 1) = match1_xy.at<float>(i, 0)*match2_xy.at<float>(i, 1)*(-1.f);

			A2.at<float>(2 * i + 1, 0) = match1_xy.at<float>(i, 1)*match2_xy.at<float>(i, 0)*(-1.f);
			A2.at<float>(2 * i + 1, 1) = match1_xy.at<float>(i, 1)*match2_xy.at<float>(i, 1)*(-1.f);
		}

		Mat A1;
		A1.create(2 * N, 8, CV_32FC1);
		A.copyTo(A1(Range::all(), Range(0, 6)));
		A2.copyTo(A1(Range::all(), Range(6, 8)));

		Mat values;
		solve(A1, B, values, DECOMP_QR);
		change.at<float>(0, 0) = values.at<float>(0);
		change.at<float>(0, 1) = values.at<float>(1);
		change.at<float>(0, 2) = values.at<float>(4);
		change.at<float>(1, 0) = values.at<float>(2);
		change.at<float>(1, 1) = values.at<float>(3);
		change.at<float>(1, 2) = values.at<float>(5);
		change.at<float>(2, 0) = values.at<float>(6);
		change.at<float>(2, 1) = values.at<float>(7);
		change.at<float>(2, 2) = 1.f;

		Mat temp1 = Mat::ones(1, N, CV_32FC1);
		Mat temp2;
		temp2.create(3, N, CV_32FC1);
		match2_xy_trans.copyTo(temp2(Range(0, 2), Range::all()));
		temp1.copyTo(temp2(Range(2, 3), Range::all()));

		Mat match2_xy_change = change * temp2;
		Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());
		float *temp_ptr = match2_xy_change.ptr<float>(2);
		for (int i = 0; i < N; ++i)
		{
			float div_temp = temp_ptr[i];
			match2_xy_change_12.at<float>(0, i) = match2_xy_change_12.at<float>(0, i) / div_temp;
			match2_xy_change_12.at<float>(1, i) = match2_xy_change_12.at<float>(1, i) / div_temp;
		}
		Mat diff = match2_xy_change_12 - match1_xy_trans;
		pow(diff, 2, diff);
		rmse = (float)sqrt(sum(diff)(0)*1.0/ N);//sum����Ǹ���ͨ���ĺ�
	}
	//��������Ʊ任
	else if (model == string("similarity"))
	{
		/*[x, y, 1, 0;
		  y, -x, 0, 1] * [a, b, c, d]'=[u,v]*/

		Mat A3;
		A3.create(2 * N, 4, CV_32FC1);
		for (int i = 0; i < N; ++i)
		{
			A3.at<float>(2 * i, 0) = match2_xy.at<float>(i, 0);
			A3.at<float>(2 * i, 1) = match2_xy.at<float>(i, 1);
			A3.at<float>(2 * i, 2) = 1.f;
			A3.at<float>(2 * i, 3) = 0.f;

			A3.at<float>(2 * i+1, 0) = match2_xy.at<float>(i, 1);
			A3.at<float>(2 * i+1, 1) = match2_xy.at<float>(i, 0)*(-1.f);
			A3.at<float>(2 * i+1, 2) = 0.f;
			A3.at<float>(2 * i + 1, 3) = 1.f;
		}

		Vec4f values;
		solve(A3, B, values, DECOMP_QR);
		change = (Mat_<float>(3, 3) << values(0), values(1), values(2),
			values(1)*(-1.0f), values(0), values(3),
			+0.f, +0.f, 1.f);

		Mat temp_1 = change(Range(0, 2), Range(0, 2));//�߶Ⱥ���ת��
		Mat temp_2 = change(Range(0, 2), Range(2, 3));//ƽ����

		Mat match2_xy_change = temp_1 * match2_xy_trans + repeat(temp_2, 1, N);
		Mat diff = match2_xy_change - match1_xy_trans;//���
		pow(diff, 2, diff);
		rmse = (float)sqrt(sum(diff)(0)*1.0 / N);//sum����Ǹ���ͨ���ĺ�
	}

	return change;
}

/*********************�ú���ɾ�������ƥ����****************************/
/*points_1��ʾ�ο�ͼ����ƥ���������λ��
 points_2��ʾ����׼ͼ���ϵ�������λ�ü���
 model��ʾ�任ģ�ͣ���similarity��,"affine"����perspective��
 threshold��ʾ�ڵ���ֵ
 inliers��ʾpoints_1��points_2�ж�Ӧ�ĵ���Ƿ�����ȷƥ�䣬����ǣ���ӦԪ��ֵΪ1������Ϊ0
 rmse��ʾ���������ȷƥ���Լ�����������
 ����һ��3 x 3���󣬱�ʾ����׼ͼ�񵽲ο�ͼ��ı任����
 */
Mat ransac(const vector<Point2f> &points_1, const vector<Point2f> &points_2, string model, float threshold, vector<bool> &inliers, float &rmse)
{
	if (points_1.size() != points_2.size())
		cout<<"ransacģ������������������һ�£�"<<endl;

	if (!(model == string("affine") || model == string("similarity") ||
		model == string("perspective")))
		cout<<"ransacģ��ͼ��任�����������"<<endl;

	const size_t N = points_1.size();//���������
	int n;
	size_t max_iteration, iterations;
	if (model == string("similarity")){
		n = 2;
		max_iteration = N*(N - 1) / 2;
	}
	else if (model == string("affine")){
		n = 3; 
		max_iteration = N*(N - 1)*(N - 2) / (2 * 3);
	}
	else if (model == string("perspective")){
		n = 4;
		max_iteration = N*(N - 1)*(N - 2) / (2 * 3);
	}

	if (max_iteration > 800)
		iterations = 800;
	else
		iterations = max_iteration;

	//ȡ��������points_1��points_2�еĵ����꣬������Mat�����У����㴦��
	Mat arr_1, arr_2;//arr_1,��arr_2��һ��[3 x N]�ľ���ÿһ�б�ʾһ��������,������ȫ��1
	arr_1.create(3, (int)N, CV_32FC1);
	arr_2.create(3, (int)N, CV_32FC1);
	float *p10 = arr_1.ptr<float>(0), *p11 = arr_1.ptr<float>(1),*p12 = arr_1.ptr<float>(2);
	float *p20 = arr_2.ptr<float>(0), *p21 = arr_2.ptr<float>(1), *p22 = arr_2.ptr<float>(2);
	for (size_t i = 0; i < N; ++i)
	{
		p10[i] = points_1[i].x;
		p11[i] = points_1[i].y;
		p12[i] = 1.f;

		p20[i] = points_2[i].x;
		p21[i] = points_2[i].y;
		p22[i] = 1.f;
	}

	Mat rand_mat;
	rand_mat.create(1, n, CV_32SC1);
	int *p = rand_mat.ptr<int>(0);
	Mat sub_arr1, sub_arr2;
	sub_arr1.create(n, 2, CV_32FC1);
	sub_arr2.create(n, 2, CV_32FC1);
	Mat T;//����׼ͼ�񵽲ο�ͼ��ı任����
	int most_consensus_num = 0;//��ǰ����һ�¼�������ʼ��Ϊ0
	vector<bool> right;
	right.resize(N);
	inliers.resize(N);

	for (size_t i = 0; i < iterations;++i)
	{
		//���ѡ��n����ͬ�ĵ��
		while (1)
		{
			randu(rand_mat, 0, double(N-1));//�������n����Χ��[0,N-1]֮�����

			//��֤��n�������겻��ͬ
			if (n == 2 && p[0] != p[1] && 
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]))
				break;

			if (n == 3 && p[0] != p[1] && p[0] != p[2] && p[1] != p[2] &&
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p10[p[0]] != p10[p[2]] || p11[p[0]] != p11[p[2]]) &&
				(p10[p[1]] != p10[p[2]] || p11[p[1]] != p11[p[2]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]) &&
				(p20[p[0]] != p20[p[2]] || p21[p[0]] != p21[p[2]]) &&
				(p20[p[1]] != p20[p[2]] || p21[p[1]] != p21[p[2]]))
				break;

			if (n == 4 && p[0] != p[1] && p[0] != p[2] && p[0] != p[3] &&
				p[1] != p[2] && p[1] != p[3] && p[2] != p[3] &&
				(p10[p[0]] != p10[p[1]] || p11[p[0]] != p11[p[1]]) &&
				(p10[p[0]] != p10[p[2]] || p11[p[0]] != p11[p[2]]) &&
				(p10[p[0]] != p10[p[3]] || p11[p[0]] != p11[p[3]]) &&
				(p10[p[1]] != p10[p[2]] || p11[p[1]] != p11[p[2]]) &&
				(p10[p[1]] != p10[p[3]] || p11[p[1]] != p11[p[3]]) &&
				(p10[p[2]] != p10[p[3]] || p11[p[2]] != p11[p[3]]) &&
				(p20[p[0]] != p20[p[1]] || p21[p[0]] != p21[p[1]]) &&
				(p20[p[0]] != p20[p[2]] || p21[p[0]] != p21[p[2]]) &&
				(p20[p[0]] != p20[p[3]] || p21[p[0]] != p21[p[3]]) &&
				(p20[p[1]] != p20[p[2]] || p21[p[1]] != p21[p[2]]) &&
				(p20[p[1]] != p20[p[3]] || p21[p[1]] != p21[p[3]]) &&
				(p20[p[2]] != p20[p[3]] || p21[p[2]] != p21[p[3]]))
				break;
		}

		//��ȡ��n�����
		for (int i = 0; i < n; ++i)
		{
			sub_arr1.at<float>(i, 0) = p10[p[i]];
			sub_arr1.at<float>(i, 1) = p11[p[i]];

			sub_arr2.at<float>(i, 0) = p20[p[i]];
			sub_arr2.at<float>(i, 1) = p21[p[i]];
		}

		//������n����ԣ�����任����T
		T = LMS(sub_arr1, sub_arr2, model, rmse);
		
		int consensus_num = 0;//��ǰһ�¼�����
		if(model == string("perspective"))
		{

			Mat match2_xy_change = T * arr_2;
			Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());
			float *temp_ptr = match2_xy_change.ptr<float>(2);
			for (size_t i = 0; i < N; ++i)
			{
				float div_temp = temp_ptr[i];
				match2_xy_change_12.at<float>(0, (int)i) = match2_xy_change_12.at<float>(0, (int)i) / div_temp;
				match2_xy_change_12.at<float>(1, (int)i) = match2_xy_change_12.at<float>(1, (int)i) / div_temp;
			}
			Mat diff = match2_xy_change_12 - arr_1(Range(0,2),Range::all());
			pow(diff, 2, diff);

			//��һ�к͵ڶ������
			
			Mat add = diff(Range(0, 1), Range::all()) + diff(Range(1, 2), Range::all());
			float *p_add = add.ptr<float>(0);
			for (size_t i = 0; i < N; ++i)
			{
				if (p_add[i] < threshold){//���С����ֵ
					right[i] = true;
					++consensus_num;
				}
				else
					right[i] = false;
			}
		}

		else if (model == string("affine") || model == string("similarity"))
		{
			Mat match2_xy_change = T * arr_2;
			Mat diff = match2_xy_change - arr_1;
			pow(diff, 2, diff);

			//��һ�к͵ڶ������
			Mat add = diff(Range(0, 1), Range::all()) + diff(Range(1, 2), Range::all());
			float *p_add = add.ptr<float>(0);
			for (size_t i = 0; i < N; ++i)
			{
				if (p_add[i] < threshold){//���С����ֵ
					right[i] = true;
					++consensus_num;
				}
				else
					right[i] = false;
			}
		}

		//�жϵ�ǰһ�¼��Ƿ�������֮ǰ����һ�¼�
		if (consensus_num>most_consensus_num){
				most_consensus_num = consensus_num;
				for (size_t i = 0; i < N; ++i)
					inliers[i] = right[i];
			}//���µ�ǰ����һ�¼�����

	}

	//ɾ���ظ����
	for (size_t i = 0; i < N-1; ++i)
	{
		for (size_t j = i + 1; j < N; ++j)
		{
			if (inliers[i] && inliers[j])
			{
				if (p10[i] == p10[j] && p11[i] == p11[j] && p20[i] == p20[j] && p21[i] == p21[j])
				{
					inliers[j] = false;
					--most_consensus_num;
				}
			}
		}
	}

	//�����������������һ�¼��ϣ�������Щ����һ�¼��ϼ�������յı任��ϵT
	Mat consensus_arr1, consensus_arr2;
	consensus_arr1.create(most_consensus_num, 2, CV_32FC1);//ÿ�д洢һ���ڵ�
	consensus_arr2.create(most_consensus_num, 2, CV_32FC1);
	int k = 0;
	for (size_t i = 0; i < N; ++i)
	{
		if (inliers[i])
		{
			consensus_arr1.at<float>(k, 0) = p10[i];
			consensus_arr1.at<float>(k, 1) = p11[i];

			consensus_arr2.at<float>(k, 0) = p20[i];
			consensus_arr2.at<float>(k, 1) = p21[i];
			++k;
		}
	}
	T = LMS(consensus_arr1, consensus_arr2, model, rmse);
	return T;
}


void image_fusion(const Mat &image_1, const Mat &image_2, const Mat T, Mat &fusion_image)
{
	if (!(image_1.depth() == CV_8U && image_2.depth() == CV_8U))
		cout<<"image_fusionģ���֧��uchar����ͼ��"<<endl;

	int rows_1 = image_1.rows, cols_1 = image_1.cols;
	int rows_2 = image_2.rows, cols_2 = image_2.cols;
	int channel_1 = image_1.channels();
	int channel_2 = image_2.channels();

	Mat image_1_temp, image_2_temp;
	if (channel_1 == 3 && channel_2 == 3){
		image_1_temp = image_1;
		image_2_temp = image_2;
	}
	else if (channel_1 == 1 && channel_2 == 3){
		image_1_temp = image_1;
		cvtColor(image_2, image_2_temp, CV_RGB2GRAY);
	}
	else if (channel_1 == 3 && channel_2 == 1){
		cvtColor(image_1, image_1_temp, CV_RGB2GRAY);
		image_2_temp = image_2;
	}
	else if (channel_1 == 1 && channel_2 == 1){
		image_1_temp = image_1;
		image_2_temp = image_2;
	}
		
	Mat T_temp = (Mat_<float>(3,3)<<1, 0, cols_1,
		                           0, 1, rows_1,
		                                0, 0, 1);
	Mat T_1 = T_temp*T;

	//�Բο�ͼ��ʹ���׼ͼ����б任
	Mat trans_1,trans_2;//same type as image_2_temp 
	trans_1=Mat::zeros(3 * rows_1, 3 * cols_1, image_1_temp.type());
	image_1_temp.copyTo(trans_1(Range(rows_1,2*rows_1), Range(cols_1,2*cols_1)));
	warpPerspective(image_2_temp, trans_2, T_1, Size(3 * cols_1, 3 * rows_1), INTER_LINEAR);

	Mat trans = trans_2.clone();
	int nRows = rows_1;
	int nCols = cols_1*image_1_temp.channels();
	int len = nCols;
	for (int i = 0; i < nRows; ++i)
	{
		uchar *ptr_1 = trans_1.ptr<uchar>(i+rows_1);
		uchar *ptr = trans.ptr<uchar>(i+rows_1);
		for (int j = 0; j < nCols; ++j)
		{
			if (ptr[j + len] == 0)//���ڷ��غ�����
				ptr[j + len] = ptr_1[j + len];
			else//�����غ�����
				ptr[j + len] = saturate_cast<uchar>(((float)ptr[j + len] + (float)ptr_1[j + len]) / 2);
		}
	}

	//ɾ�����������
	Mat left_up = T_1*(Mat_<float>(3, 1) << 0, 0, 1);//���Ͻ�
	Mat left_down = T_1*(Mat_<float>(3, 1) << 0, rows_2 - 1, 1);//���½�
	Mat right_up = T_1*(Mat_<float>(3, 1) << cols_2 - 1, 0, 1);//���Ͻ�
	Mat right_down = T_1*(Mat_<float>(3, 1) << cols_2 - 1, rows_2 - 1, 1);//���½�

	//����͸�ӱ任����Ҫ����һ������
	left_up = left_up / left_up.at<float>(2, 0);
	left_down = left_down / left_down.at<float>(2, 0);
	right_up = right_up / right_up.at<float>(2, 0);
	right_down = right_down / right_down.at<float>(2, 0);

	//����x,y����ķ�Χ
	float temp_1 = min(left_up.at<float>(0, 0), left_down.at<float>(0, 0));
	float temp_2 = min(right_up.at<float>(0, 0), right_down.at<float>(0, 0));
	float min_x = min(temp_1, temp_2);

	temp_1 = max(left_up.at<float>(0, 0), left_down.at<float>(0, 0));
	temp_2 = max(right_up.at<float>(0, 0), right_down.at<float>(0, 0));
	float max_x = max(temp_1, temp_2);

	temp_1 = min(left_up.at<float>(1, 0), left_down.at<float>(1, 0));
	temp_2 = min(right_up.at<float>(1, 0), right_down.at<float>(1, 0));
	float min_y = min(temp_1, temp_2);

	temp_1 = max(left_up.at<float>(1, 0), left_down.at<float>(1, 0));
	temp_2 = max(right_up.at<float>(1, 0), right_down.at<float>(1, 0));
	float max_y = max(temp_1, temp_2);

	int X_min = max(cvFloor(min_x), 0);
	int X_max = min(cvCeil(max_x), 3 * cols_1-1);
	int Y_min = max(cvFloor(min_y), 0);
	int Y_max = min(cvCeil(max_y), 3 * rows_1-1);


	if (X_min>cols_1)
		X_min = cols_1;
	if (X_max<2 * cols_1-1)
		X_max = 2 * cols_1 - 1;
	if (Y_min>rows_1)
		Y_min = rows_1;
	if (Y_max<2 * rows_1-1)
		Y_max = 2 * rows_1 - 1;

	Range Y_range(Y_min, Y_max+1);
	Range X_range(X_min, X_max+1);
	fusion_image = trans(Y_range, X_range);
	Mat matched_image = trans_2(Y_range, X_range);

	imwrite(".\\image_save\\��׼��Ĵ���׼ͼ��.jpg", matched_image);
}

	  