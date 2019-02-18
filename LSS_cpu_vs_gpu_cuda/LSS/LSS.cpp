#include "LSS.h"
#include <fstream>
SelfSimilarity::SelfSimilarity(): 
largeSize(41),
smallSize(5)
{
	SelfDis= new SelfSimDescriptor(5,41);
		
}

//----------------------------------------------------------------------------------------------------------------------------------//
SelfSimilarity::~SelfSimilarity()
{
}


//----------------------------------------------------------------------------------------------------------------------------------//
Mat SelfSimilarity::ComputeDescriptor(Mat &img,float Saliency,bool flag,KeyPoint locations){
	//Initilization of parameters
    //descriptors.clear();
	
    mapping0.clear();
	
	//Specify the winstride size
    Size winStride=Size(1,1);
    // Specify image depth 
	CV_Assert( img.depth() == CV_8U ); 


    winStride.width = std::max(winStride.width, 1);
    winStride.height = std::max(winStride.height, 1);
    Size gridSize = SelfDis->getGridSize(img.size(), winStride);
	int i=0, nwindows = 1;
	int border = largeSize/2 + smallSize/2;
    fsize = (int)SelfDis->getDescriptorSize();
	//std::cout << "fsize " << fsize << std::endl;
    vector<float> tempFeature(fsize+1);
	if (flag==0)
	{
		Descriptors0.resize(fsize*nwindows + 1);
		mapping0.resize(nwindows);
	}
	
	// initialize SSD
    Mat ssd(largeSize, largeSize, CV_32F), mappingMask;
	// Define LogPolarMapping 
    SelfDis->computeLogPolarMapping(mappingMask);
    int j,k,d, S_Num;
	// Compute descriptor and map to LogPolar
		    KeyPoint pts;
			float* feature0;
			S_Num = 0;

		    feature0 = &Descriptors0[fsize*i];
			float* feature = &tempFeature[0];
		    pts = locations;
           /* if( pts.pt.x < border || pts.pt.x >= img.cols - border ||
                pts.pt.y < border || pts.pt.y >= img.rows - border )
            {
                for( j = 0; j < fsize; j++ )
                    feature0[j] = 0.f;
            }*/
           // pts.pt = Point((i % gridSize.width)*winStride.width + border,
                    //   (i / gridSize.width)*winStride.height + border);

			SelfDis->SSD(img, pts.pt, ssd);//调试在这里中断，由于上述代码点坐标超出边界
			
			// Determine in the local neighborhood the largest difference and use for normalization
			float var_noise = 1000.f;
			for( k = -1; k <= 1 ; k++ )
				for( d = -1 ; d <= 1 ; d++ )
					var_noise = std::max(var_noise, ssd.at<float>(largeSize/2+k, largeSize/2+d));
                    //  var_noise =  ssd.at<float>(largeSize/2+k, largeSize/2+d);

			for( j = 0; j <= fsize; j++ )
				feature[j] = FLT_MAX;
					//feature[j] = 0;

			// Derive feature vector before exp(-x) computation
			// Idea: for all  x,a >= 0, a=const.   we have:
			//       max [ exp( -x / a) ] = exp ( -min(x) / a )
			// Thus, determine min(ssd) and store in feature[...]
			for( k = 0; k < ssd.rows; k++ )
			{
				const schar *mappingMaskPtr = mappingMask.ptr<schar>(k);//m.ptr(row) 返回第 row 行数据的首地址
				const float *ssdPtr = ssd.ptr<float>(k);
				for(d = 0 ; d < ssd.cols; d++ )
				{
					int index = mappingMaskPtr[d];
					feature[index] = std::min(feature[index], ssdPtr[d]);
					//feature[index] = std::max(feature[index], ssdPtr[d]);

				}
			}

			var_noise = -1.f/var_noise;

			for( j = 0; j < fsize; j++ )
			{
				// Counting number of non-informitive bins of a descriptor for saliency
				//if (feature[j] < Saliency || feature[j] == 0 )
				//	S_Num++;
					//feature[j]=0
				feature0[j] = feature[j]*var_noise;			
			}

			//Mat _f(1, fsize, CV_32F, feature0);
			Mat _f(1, fsize, CV_32F);
			for (j = 0; j < fsize; j++) _f.at<float>(0, j) = feature0[j];
            cv::exp(_f, _f);
			for(j=0;j<fsize;j++)
			{
			   if(_f.at<float>(0,j)<0.01)
				   S_Num++;
			}
			if(S_Num>24)
			{
			  for(j=0;j<fsize;j++)
			  {
			     _f.at<float>(0,j)=0;
			  }
			}
			return _f;

		//	des_result<<_f<<endl;
 }

