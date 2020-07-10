#include "dynaslam/MaskNet.h"
#include <iostream>
#include <fstream>
#include <iomanip>

namespace DynaSLAM
{

//#define U_SEGSt(a)\
		gettimeofday(&tvsv,0);\
		a = tvsv.tv_sec + tvsv.tv_usec/1000000.0
//struct timeval tvsv;
//double t1sv, t2sv,t0sv,t3sv;
//void tic_initsv(){U_SEGSt(t0sv);}
//void toc_finalsv(double &time){U_SEGSt(t3sv); time =  (t3sv- t0sv)/1;}
//void ticsv(){U_SEGSt(t1sv);}
//void tocsv(){U_SEGSt(t2sv);}
// std::cout << (t2sv - t1sv)/1 << std::endl;}

SegmentDynObject::SegmentDynObject(){

}

/*SingleViewDepthEstimator::~SingleViewDepthEstimator(){
	delete this->py_module;
	delete this->py_class;
	delete this->net;
	delete this->cvt;
}*/

cv::Mat SegmentDynObjectStereo::GetSegmentation(const cv::Mat &image1, const cv::Mat &image2, cv::Mat &MaskRight){
;
}

void SegmentDynObjectStereo::ImportSettings(){
;
}


}






















