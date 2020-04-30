#include "opencv2\opencv.hpp"
#include "opencv2\imgcodecs.hpp"
//#include "opencv2\core\core.hpp"
//#include "opencv2\imgproc\imgproc.hpp"    
//#include "opencv2\highgui\highgui.hpp"
#include "GaborFR.h"
#include <iostream>
//#include <vector>

//using namespace std;
using namespace cv;

// ref. https://blog.csdn.net/yeyang911/article/details/18353651?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-27&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-27

int main()
{
	// Mat M = getGaborKernel(Size(9,9),2*CV_PI,u*CV_PI/8, 2*CV_PI/pow(2,CV_PI*(v+2)/2),1,0);

	String file_name = "./vpc_black1.png";  // 158(W)x96(H)
	//Mat src = imread("d:\\vpc_black.png", 0);  // cv::IMREAD_GRAYSCALE = 0
	//Mat src = imread("c:\\vpc_black_101.png", cv::IMREAD_GRAYSCALE);  // 101(W)x101(H)
	Mat srcimg = imread(file_name, cv::IMREAD_UNCHANGED);  // 158(W)x96(H), cv::IMREAD_UNCHANGED = -1, cv::IMREAD_COLOR = 1

	// show the original image
	String windowName = "Original Image";
	GaborFR::toShow(srcimg, windowName);

	Mat grayimg;
	cvtColor(srcimg, grayimg, cv::COLOR_BGR2GRAY);

	// normalize value of pixel from 0 to 1
	normalize(grayimg, grayimg, 1, 0, CV_MINMAX, CV_32F);

	Mat showreal, showReal;
	Mat showoutreal, showoutReal;
	Mat outReal, outImaginary;
	Mat Mreal, Moutreal;
	Mat Linereal, Lineoutreal;
	Mat totalReal;

	int iSize = 12;  // 10
	bool switchtotal = true;
	//二維Gabor濾波後的效果圖: 每列i是同一尺度scale(5)，每行j是同一方向orientation(8)
	for (int i = 0; i < 8; i++)
	{
		showreal.release();
		showoutreal.release();
		for (int j = 0; j < 5; j++)
		{
			//Mat GaborFR::getRealGaborKernel(Size ksize, double sigma, double theta, double nu, double gamma, int ktype)
			Mat real = GaborFR::getRealGaborKernel(Size(iSize, iSize), 2 * CV_PI, i * CV_PI / 8 + CV_PI / 2, j, 1);  // CV_32F, CV_64F
			Mat imaginary = GaborFR::getImageGaborKernel(Size(iSize, iSize), 2 * CV_PI, i * CV_PI / 8 + CV_PI / 2, j, 1);

			GaborFR::getFilterRealImagePart(grayimg, real, imaginary, outReal, outImaginary);

			//cout << outImaginary.data << endl;
			//outImaginary.convertTo(outImaginary, CV_8UC3);

			//M = GaborFR::getPhase(real, imaginary);
			//M = GaborFR::getMagnitude(real, imaginary);
			//M = GaborFR::getPhase(outR, outI);
			//M = GaborFR::getMagnitude(outR, outI);
			//MatTemp2 = GaborFR::getPhase(outR, outI);
			//M = outR;
			//imshow("saveM", M);

			Mreal = real;
			// resize(Mreal, Mreal, Size(100, 100));
			// normalize value of pixel from 255 to 0
			GaborFR::normPush(Mreal, showreal, Linereal);

			//Mat filtered_src = GaborFR::gaborFilter(src, real);
			//filtered_src.convertTo(filtered_src, CV_8UC3);
			//Moutreal = filtered_src;

			Moutreal = outReal;
			GaborFR::normPush(Moutreal, showoutreal, Lineoutreal);

			// 將 outReal 圖像相加
			if (switchtotal == true)
			{
				totalReal = outReal.clone();
				switchtotal = false;
			}
			//影像相加：void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype = -1)
			addWeighted(outReal, 1, totalReal, 1, 0, totalReal);

			//Mat gaber_show = GaborFR::normalizeFilterShow(filtered_src);
		}

		showreal = showreal.t();
		Linereal = Mat::ones(4, showreal.cols, showreal.type()) * 255;
		showReal.push_back(showreal);
		showReal.push_back(Linereal);

		showoutreal = showoutreal.t();
		Lineoutreal = Mat::ones(4, showoutreal.cols, showoutreal.type()) * 255;
		showoutReal.push_back(showoutreal);
		showoutReal.push_back(Lineoutreal);
	}

	//outImaginary.convertTo(outImaginary, CV_8UC3);

	showReal = showReal.t();
	// bool flag = imwrite("c:\\out.png", showReal);
	windowName = "Show Gabot Filter Kernel";  // name of the window
	GaborFR::toShow(showReal, windowName);

	showoutReal = showoutReal.t();
	windowName = "Show Result of Gabor Filter/Transform";
	GaborFR::toShow(showoutReal, windowName);

	windowName = "addWeighted";
	GaborFR::toShow(totalReal, windowName);

	printf("Press Any Key To Exit...\n");
	waitKey(0);
	//destroyWindow(windowName);  // destroy the created window
	destroyAllWindows();

	return 0;
}
