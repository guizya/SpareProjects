#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <iomanip> 
#include <fstream>
#include "LSS.h"
#include "match.h"
#include "mosaic.h"
#include "LssParallel.h"
#include "EulerParallel.h"
#include "Euler.h"
#include <ctime>

using namespace cv;
using namespace std;

ofstream surfDes("surf_des.txt");
ofstream surfIrDes("surfIrDes.txt");
ofstream Visdes("Vis_des.txt");
ofstream VisIrdes("VisIrdes.txt");
ofstream parallelVisDes("VisParallel.txt");
ofstream parallelThermalDes("ThermalParallel.txt");

Mat mergeCols(const Mat A,const Mat B);  // 按列合并矩阵
Mat getSobel(Mat src);
vector<pair<KeyPoint, KeyPoint> > slopeHist(vector<pair<KeyPoint, KeyPoint> >matches);
void opposImg(Mat &Img);
static void showDifference(const Mat& image1, const Mat& image2, const char* title);
vector<float> CalRMSE(vector<pair<KeyPoint, KeyPoint> > goodMatches,const Mat &warp);

int main()
{
	Mat thermal = imread("PB_2.jpg", 0 );
	Mat visible = imread("PB_1.jpg", 0 );
	if( !thermal.data || !visible.data )
	{ printf("读取图片错误 \n"); return false; } 

	//canny检测
	//Mat therCanny,visCanny;
	//Canny(thermal,therCanny,10,30,3);
	//Canny(visible,visCanny,10,30,3);
	//imshow("thermal",therCanny);
	//imshow("visible",visCanny);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	//SURF
	int minHessian = 900,minHessian1=1200;
	SurfFeatureDetector detector( minHessian1 ),detector1(minHessian);
	
	detector.detect( visible, keypoints_1 );
	detector1.detect(thermal, keypoints_2 );

	//drawKeypoints(visible, keypoints_1, visible, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);  
   // imshow("SURF visible feature", visible); 
	//drawKeypoints(thermal, keypoints_1, thermal, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG); 
	//imshow("SURF thermal feature", thermal); 


	//surf descriptor
	SurfDescriptorExtractor SurfDescriptor;    
    Mat SurfDesc1,SurfDesc2;    
    SurfDescriptor.compute(visible, keypoints_1,SurfDesc1);    
    SurfDescriptor.compute(thermal, keypoints_2,SurfDesc2);
	//surfDes<<SurfDesc1<<endl;
	//surfIrDes<<SurfDesc2<<endl;

    
	/*剔除边界特征点及生成特征点的自相似性描述子*/
	int border=23; //用于剔除边界特征点；
    SelfSimilarity sim,sim1;
	vector<KeyPoint> keypoints1, keypoints2;
	Mat VisSimDes,IrSimDes;
	vector<Mat> tempResult, tempResult2;
	unsigned int number = 0;
	clock_t start = clock();
    for(int i=0;i!=keypoints_1.size();++i)
     {
	    if( keypoints_1[i].pt.x > border && keypoints_1[i].pt.x <= visible.cols - border &&
                keypoints_1[i].pt.y > border && keypoints_1[i].pt.y <= visible.rows - border )
				{
					number++;
					keypoints1.push_back(keypoints_1[i]);
					 //Visdes<<keypoints_1[i].pt<<": ";
					Mat temp = sim.ComputeDescriptor(visible, 0, 0, keypoints_1[i]);
					VisSimDes.push_back(temp);
					tempResult.push_back(temp);
					 //Visdes<<sim.ComputeDescriptor(visible,0,0,keypoints_1[i])<<endl;
		        }
      }
	clock_t end = clock();
	std::cout << "Number " << number << " " << (double)(end - start) /  CLOCKS_PER_SEC << endl;
	
	number = 0;
	start = clock();
      for(int j=0;j!=keypoints_2.size();++j)
	   {
	    if( keypoints_2[j].pt.x > border && keypoints_2[j].pt.x <= thermal.cols - border &&
                keypoints_2[j].pt.y > border && keypoints_2[j].pt.y <= thermal.rows - border )
			      {
					number++;
					keypoints2.push_back(keypoints_2[j]);
					 //VisIrdes<<keypoints_2[j].pt<<": " ;
					Mat temp = sim1.ComputeDescriptor(thermal, 0, 0, keypoints_2[j]);
					IrSimDes.push_back(temp);	
					tempResult2.push_back(temp);
					 //VisIrdes<<sim1.ComputeDescriptor(thermal,0,0,keypoints_2[j])<<endl;
		          }
	   }
	  end = clock();
	  std::cout << "Number " << number << " " << (double)(end - start) / CLOCKS_PER_SEC << endl;

	  LssParallelStruct pVisible(visible.rows, visible.cols, visible, keypoints1);
	  LssParallelStruct pThermal(thermal.rows, thermal.cols, thermal, keypoints2);
	  EulerDistanceStruct eulerDistance(keypoints1, keypoints2, pVisible, pThermal);
	  start = clock();
	  Mat parallelVis = pVisible.lssParallel(visible.rows, visible.cols, keypoints1);
	  //vector<Mat> parallelVis = lssParallel(visible.rows, visible.cols, keypoints1);
	  end = clock();
	  std::cout << "GPU time " << (double)(end - start) / CLOCKS_PER_SEC << endl;

	  start = clock();
	  Mat parallelThermal = pThermal.lssParallel(thermal.rows, thermal.cols, keypoints2);
	  //vector<Mat> parallelThermal = lssParallel(thermal.rows, thermal.cols, keypoints2);
	  end = clock();
	  std::cout << "GPU time " << (double)(end - start) / CLOCKS_PER_SEC << endl;

	  vector<pair<KeyPoint, KeyPoint> > matchesEuler, matchesEuler1;
	  start = clock();
	  eulerDistance.eulerParallel(matchesEuler, matchesEuler1, keypoints1, keypoints2, pVisible, pThermal);
	  end = clock();
	  std::cout << "GPU euler time " << (double)(end - start) / CLOCKS_PER_SEC << endl;

	  //combine both the descs
	  Mat VisDes,IrDes;
	  //VisDes=mergeCols(1*VisSimDes,0*SurfDesc1);
	  //IrDes=mergeCols(1*IrSimDes,0*SurfDesc2);
	
	  VisDes=VisSimDes;
	  IrDes=IrSimDes;

	  VisDes = parallelVis;
	  IrDes = parallelThermal;
	
	
	  /*加入双向匹配约束来选择合适的匹配点对*/
	  vector<pair<KeyPoint, KeyPoint> > matches,matches1;//分别存储正向匹配和反向匹配的特征点对
	  start = clock();
	  matches=MatchDes(VisDes,IrDes, keypoints1, keypoints2);
	  matches1=MatchDes(IrDes,VisDes, keypoints2, keypoints1);
	  end = clock();
	  std::cout << "CPU matching time " << (double)(end - start) / CLOCKS_PER_SEC << endl;

	  pVisible.dumpResult(VisDes, keypoints1, Visdes);
	  pVisible.dumpResult(IrDes, keypoints2, surfDes);
	  pVisible.dumpResult(parallelVis, keypoints1, parallelVisDes);
	  pVisible.dumpResult(parallelThermal, keypoints2, parallelThermalDes);

	  eulerDistance.dumpResult(matches, "Mathces.txt");
	  eulerDistance.dumpResult(matches1, "Mathces_1.txt");
	  eulerDistance.dumpResult(matchesEuler, "MathcesEuler.txt");
	  eulerDistance.dumpResult(matchesEuler1, "MathcesEuler_1.txt");
	  matches = matchesEuler;
	  matches1 = matchesEuler1;

	  {
		  vector<pair<KeyPoint, KeyPoint> > eulerMatches, eulerMatches1;
		  eulerStart(keypoints1, keypoints2, VisSimDes, IrSimDes);
		  eulerExecute(eulerMatches, eulerMatches1, keypoints1, keypoints2);
		  eulerEnd();
		  //matches = eulerMatches;
		  //matches1 = eulerMatches1;
	  }

	  /*根据双相匹配选择特征点对并显示*/
	  IplImage *vis=cvLoadImage("PA_1.jpg");
	  IplImage *ther=cvLoadImage("PA_2.jpg");
	  vector< Point2f > obj;//存储参考图像（可见光）点
      vector< Point2f > scene;//存储待配准图像（红外）点
	  vector<KeyPoint> goodVisKeypoints,goodTherKeypoints;
	  int matchCount=0;//统计得到的匹配点对数目
	  const int & w = vis->width;
	  for(int i=0;i!=matches.size();++i)
		{
		  for(int j=0;j!=matches1.size();++j)
		   {
		      if((matches[i].first.pt==matches1[j].second.pt)&&(matches[i].second.pt==matches1[j].first.pt))
			  {  
					 obj.push_back( matches[i].first.pt );
					 goodVisKeypoints.push_back( matches[i].first);
					 //circle(visible,matches[i].first.pt,1,Scalar(0,0,0));
                     scene.push_back(matches[i].second.pt);
					 goodTherKeypoints.push_back(matches[i].second);
					 //circle(thermal,matches[i].second.pt,1,Scalar(0,0,0));
					 cvLine(vis,cvPoint(matches[i].first.pt.x,matches[i].first.pt.y),cvPoint(matches[i].second.pt.x+w,matches[i].second.pt.y), Scalar(255,0,0),0);
                     cvLine(ther,cvPoint(matches[i].first.pt.x-w,matches[i].first.pt.y),cvPoint(matches[i].second.pt.x,matches[i].second.pt.y), Scalar(255,0,0),0);
                     //cout<<matches[i].first.pt<<matches[i].second.pt<<endl;
					 matchCount++;
			   }
		   }
		}
	  cout<<"matchcount "<<matchCount<<endl;

    /*RANSAC去除误配得到变换矩阵*/
	Mat TransMat;
	vector<bool>inliner;
	float rmse=0;
	float threshold=1.5;
	string model="affine";
	start = clock();
	TransMat=ransac(obj,scene,model, threshold,inliner,rmse);
	end = clock();
	std::cout << "CPU ransac time " << (double)(end - start) / CLOCKS_PER_SEC << endl;
	//cout<<"仿射矩阵："<<TransMat<<endl;
	cout<<"配准误差RMSE："<<rmse<<endl;

	Mat warp_dst;
	warpPerspective(thermal, warp_dst, TransMat, visible.size());
	//imshow("warp",warp_dst);
	//imwrite("reg.bmp",warp_dst);

	Mat fusion_image;
	image_fusion(visible, thermal,TransMat,fusion_image);
	//imshow("fusion",fusion_image);
	//imwrite("fusion.bmp",fusion_image);
	//Mat regfu;
	//regfu=0.5*visible+0.5*warp_dst;
	//imshow("fus",regfu);
	Mat chessboard_1, chessboard_2, mosaic_image;
	mosaic_map(visible, warp_dst, chessboard_1, chessboard_2, mosaic_image,60);
	rectangle( mosaic_image, cvPoint(340, 260), cvPoint(380, 290), Scalar(255, 0, 0), 2, 4, 0 );
	rectangle( mosaic_image, cvPoint(520, 260), cvPoint(560, 290), Scalar(255, 0, 0), 2, 4, 0 );
	//imshow("mosaic_map",mosaic_image);
	//imwrite("mosaic_map.bmp",mosaic_image);
	
	//cvNamedWindow("1", 1 );
    //cvNamedWindow("2", 1 );
	//cvShowImage("1", vis);
    //cvShowImage("2",ther);

	//waitKey(0);
	system("pause");
	return 0; 
}

static void showDifference(const Mat& image1, const Mat& image2, const char* title)
{
    Mat img1, img2;
    image1.convertTo(img1, CV_32FC3);
    image2.convertTo(img2, CV_32FC3);
    if(img1.channels() != 1)
        cvtColor(img1, img1, CV_RGB2GRAY);
    if(img2.channels() != 1)
        cvtColor(img2, img2, CV_RGB2GRAY);

    Mat imgDiff;
    img1.copyTo(imgDiff);
    imgDiff -= img2;
    imgDiff /= 2.f;
    imgDiff += 128.f;

    Mat imgSh;
    imgDiff.convertTo(imgSh, CV_8UC3);
    imshow(title, imgSh);
}

Mat mergeCols(Mat A, Mat B)  //按列合并矩阵
{
	int totalCols = A.cols + B.cols;
	Mat mergedDescriptors(A.rows, totalCols, A.type());
	Mat submat = mergedDescriptors.colRange(0, A.cols);
	A.copyTo(submat);
	submat = mergedDescriptors.colRange(A.cols, totalCols);
	B.copyTo(submat);
	return mergedDescriptors;
}
Mat getSobel(Mat src)
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y,dst;

	//求X方向梯度
    Sobel( src, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	//求Y方向梯度
	Sobel( src, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	//合并梯度(近似)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );
	return dst;
}

//斜率一致性
vector<pair<KeyPoint, KeyPoint> > slopeHist(vector<pair<KeyPoint, KeyPoint> >matches)
{
		vector<int> a;
		int arcStep=20;
		int *ms = new int[360/arcStep];
		for(int i=0;i!=matches.size();++i)
		{
           float tan=cvFastArctan((matches[i].first.pt.y-matches[i].second.pt.y),(matches[i].first.pt.x-matches[i].second.pt.x));//计算匹配点对斜率
		   int ang=(int)tan/arcStep*arcStep;//斜率相差arcStep°算作一样斜率
		   a.push_back(ang);
		   ms[ang/arcStep]=ms[ang/arcStep]+1;//统计相同斜率的点对数   
		}
		int count=0,idx=0;
		for(int i=0;i!=(360/arcStep);++i)
		{
		  if(ms[i]>count)
		  {
		     count=ms[i];
			 idx=i;
		  }
		}
		idx=idx*arcStep;
		vector<std::pair<KeyPoint, KeyPoint> > goodMatches;
		for(int i=0;i!=matches.size();++i)
		{
		  if(a[i]==idx)
		  {
		     goodMatches.push_back(matches[i]);
		  }
		}
		return goodMatches;
}

//图像反相
void opposImg(Mat &Img)
{   
   for(int i=0;i !=Img.rows; ++ i)  
     { 
        for(int j=0;j != Img.cols; ++ j) 
         { 
			Img.at<uchar>(i,j)=abs(255-Img.at<uchar>(i,j));
         } 
     }  
   //equalizeHist( Img, Img );
}

//计算配准误差RMSE
vector<float> CalRMSE(vector<pair<KeyPoint, KeyPoint> > matches,Mat warp){
	vector<float> RMSE;
	for(int i=0;i!=matches.size();++i){
	   float rmse=0,rmse_x=0,rmse_y=0;
	   rmse_x=abs(matches[i].first.pt.x-(matches[i].second.pt.x*warp.at<float>(0,0)+matches[i].second.pt.y*warp.at<float>(0,1)+warp.at<float>(0,2)));
	   rmse_y=abs(matches[i].first.pt.y-(matches[i].second.pt.x*warp.at<float>(1,0)+matches[i].second.pt.y*warp.at<float>(1,1)+warp.at<float>(1,2)));
	   rmse=sqrt(pow(rmse_x,2)+pow(rmse_y,2));
	   RMSE.push_back(rmse);
	}
	return RMSE;
}



