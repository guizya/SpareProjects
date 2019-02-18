#ifndef _LSS_h
#define _LSS_h

#include "opencv2/opencv.hpp"  

using namespace cv;
using namespace std;


class SelfSimilarity  {

public:
    // Self Similarity constructor
	SelfSimilarity();
	// Compute Descriptor
	Mat ComputeDescriptor(Mat &img, float Saliency,bool flag,KeyPoint locations);

	virtual ~SelfSimilarity();

    vector<float> Descriptors0;
	
private:
	//Self similarity
	SelfSimDescriptor*  SelfDis;
	// Self similarity ROI size
	int largeSize; 
	// Self similarity patch size
	int smallSize;
	
	vector<double> MappingTable;
	
	
    // Descriptor size
	int fsize; 
	vector<int> mapping0;

};

#endif