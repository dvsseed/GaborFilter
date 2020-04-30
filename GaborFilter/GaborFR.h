#pragma once
#include "opencv2\opencv.hpp"
#include <vector>

using namespace std;
using namespace cv;

class GaborFR
{
	public:
		GaborFR();
		static Mat getImageGaborKernel(Size ksize, double sigma, double theta, double nu, double gamma = 1, int ktype = CV_32F);
		static Mat getRealGaborKernel(Size ksize, double sigma, double theta, double nu, double gamma = 1, int ktype = CV_32F);
		static Mat getPhase(Mat &real, Mat &imaginary);
		static Mat getMagnitude(Mat &real, Mat &imaginary);
		static void getFilterRealImagePart(Mat &src, Mat &real, Mat &imaginary, Mat &outReal, Mat &outImaginary);
		static Mat getFilterRealPart(Mat &src, Mat &real);
		static Mat getFilterImagePart(Mat &src, Mat &imaginary);
		static void normPush(Mat &mreal, Mat &showreal, Mat &linereal);
		static void toShow(Mat &showreal, String windowname);

		//static Mat gaborFilter(Mat &img, Mat &filter);
		//static Mat getGaborFilter(double lambda, double theta, double sigma2, double gamma, double psi = 0.0f);
		//static Mat normalizeFilterShow(Mat gaber);

		void Init(Size ksize = Size(19, 19), double sigma = 2 * CV_PI, double gamma = 1, int ktype = CV_32FC1);

	private:
		vector<Mat> gaborRealKernels;
		vector<Mat> gaborImageKernels;
		bool isInited;
};