//#include "stdafx.h"
#include "GaborFR.h"

GaborFR::GaborFR()
{
	isInited = false;
}

void GaborFR::Init(Size ksize, double sigma, double gamma, int ktype)
{
	gaborRealKernels.clear();
	gaborImageKernels.clear();
	double mu[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // eight orientations
	double nu[5] = { 0, 1, 2, 3, 4 };  // five scale
	int i, j;
	for (i = 0; i < 5; i++)
	{
		for (j = 0; j < 8; j++)
		{
			gaborRealKernels.push_back(getRealGaborKernel(ksize, sigma, mu[j] * CV_PI / 8, nu[i], gamma, ktype));
			gaborImageKernels.push_back(getImageGaborKernel(ksize, sigma, mu[j] * CV_PI / 8, nu[i], gamma, ktype));
		}
	}
	isInited = true;
}

Mat GaborFR::getImageGaborKernel(Size ksize, double sigma, double theta, double nu, double gamma, int ktype)
{
	double sigma_x = sigma;
	double sigma_y = sigma / gamma;
	int nstds = 3;
	double kmax = CV_PI / 2;
	double f = cv::sqrt(2.0);
	int xmin, xmax, ymin, ymax;
	double c = cos(theta), s = sin(theta);
	if (ksize.width > 0)
	{
		xmax = ksize.width / 2;
	}
	else  // same with matlab, default is 19
	{
		xmax = cvRound(std::max(fabs(nstds * sigma_x * c), fabs(nstds * sigma_y * s)));
	}
	if (ksize.height > 0)
	{
		ymax = ksize.height / 2;
	}
	else
	{
		ymax = cvRound(std::max(fabs(nstds * sigma_x * s), fabs(nstds * sigma_y * c)));
	}
	xmin = -xmax;
	ymin = -ymax;
	CV_Assert(ktype == CV_32F || ktype == CV_64F);
	float *pFloat;
	double *pDouble;
	Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
	double k = kmax / pow(f, nu);
	double scaleReal = k * k / sigma_x / sigma_y;
	for (int y = ymin; y <= ymax; y++)
	{
		if (ktype == CV_32F)
		{
			pFloat = kernel.ptr<float>(ymax - y);
		}
		else
		{
			pDouble = kernel.ptr<double>(ymax - y);
		}
		for (int x = xmin; x <= xmax; x++)
		{
			double xr = x * c + y * s;
			double v = scaleReal * exp(-(x * x + y * y) * scaleReal / 2);
			double temp = sin(k * xr);
			v = temp * v;
			if (ktype == CV_32F)
			{
				pFloat[xmax - x] = (float)v;
			}
			else
			{
				pDouble[xmax - x] = v;
			}
		}
	}
	return kernel;
}

// sigma is 2 * pi
Mat GaborFR::getRealGaborKernel(Size ksize, double sigma, double theta, double nu, double gamma, int ktype)
{
	double sigma_x = sigma;
	double sigma_y = sigma / gamma;
	int nstds = 3;
	double kmax = CV_PI / 2;
	double f = cv::sqrt(2.0);
	int xmin, xmax, ymin, ymax;
	double c = cos(theta), s = sin(theta);
	if (ksize.width > 0)
	{
		xmax = ksize.width / 2;
	}
	else  // same with matlab, default is19
	{
		xmax = cvRound(std::max(fabs(nstds * sigma_x * c), fabs(nstds * sigma_y * s)));
	}

	if (ksize.height > 0)
		ymax = ksize.height / 2;
	else
		ymax = cvRound(std::max(fabs(nstds * sigma_x * s), fabs(nstds * sigma_y * c)));
	xmin = -xmax;
	ymin = -ymax;
	CV_Assert(ktype == CV_32F || ktype == CV_64F);
	float *pFloat;
	double *pDouble;
	Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
	double k = kmax / pow(f, nu);
	double exy = sigma_x * sigma_y / 2;
	double scaleReal = k * k / sigma_x / sigma_y;
	int x, y;
	for (y = ymin; y <= ymax; y++)
	{
		if (ktype == CV_32F)
		{
			pFloat = kernel.ptr<float>(ymax - y);
		}
		else
		{
			pDouble = kernel.ptr<double>(ymax - y);
		}
		for (x = xmin; x <= xmax; x++)
		{
			double xr = x * c + y * s;
			double v = scaleReal * exp(-(x * x + y * y) * scaleReal / 2);
			double temp = cos(k * xr) - exp(-exy);
			v = temp * v;
			if (ktype == CV_32F)
			{
				pFloat[xmax - x] = (float)v;
			}
			else
			{
				pDouble[xmax - x] = v;
			}
		}
	}
	return kernel;
}

Mat GaborFR::getMagnitude(Mat &real, Mat &imaginary)
{
	CV_Assert(real.type() == imaginary.type());
	CV_Assert(real.size() == imaginary.size());
	int ktype = real.type();
	int row = real.rows, col = real.cols;
	int i, j;
	float *pFloat, *pFloatR, *pFloatI;
	double *pDouble, *pDoubleR, *pDoubleI;
	Mat kernel(row, col, real.type());
	for (i = 0; i < row; i++)
	{
		if (ktype == CV_32FC1)
		{
			pFloat = kernel.ptr<float>(i);
			pFloatR = real.ptr<float>(i);
			pFloatI = imaginary.ptr<float>(i);
		}
		else
		{
			pDouble = kernel.ptr<double>(i);
			pDoubleR = real.ptr<double>(i);
			pDoubleI = imaginary.ptr<double>(i);
		}
		for (j = 0; j < col; j++)
		{
			if (ktype == CV_32FC1)
			{
				pFloat[j] = sqrt(pFloatI[j] * pFloatI[j] + pFloatR[j] * pFloatR[j]);
			}
			else
			{
				pDouble[j] = sqrt(pDoubleI[j] * pDoubleI[j] + pDoubleR[j] * pDoubleR[j]);
			}
		}
	}
	return kernel;
}

Mat GaborFR::getPhase(Mat &real, Mat &imaginary)
{
	CV_Assert(real.type() == imaginary.type());
	CV_Assert(real.size() == imaginary.size());
	int ktype = real.type();
	int row = real.rows, col = real.cols;
	int i, j;
	float *pFloat, *pFloatR, *pFloatI;
	double *pDouble, *pDoubleR, *pDoubleI;
	Mat kernel(row, col, real.type());
	for (i = 0; i < row; i++)
	{
		if (ktype == CV_32FC1)
		{
			pFloat = kernel.ptr<float>(i);
			pFloatR = real.ptr<float>(i);
			pFloatI = imaginary.ptr<float>(i);
		}
		else
		{
			pDouble = kernel.ptr<double>(i);
			pDoubleR = real.ptr<double>(i);
			pDoubleI = imaginary.ptr<double>(i);
		}
		for (j = 0; j < col; j++)
		{
			if (ktype == CV_32FC1)
			{
				// if(pFloatI[j] / (pFloatR[j] + pFloatI[j]) > 0.99)
				// {
				//     pFloat[j] = CV_PI / 2;
				// }
				// else
				// {
				//     pFloat[j] = atan(pFloatI[j] / pFloatR[j]);
				pFloat[j] = asin(pFloatI[j] / sqrt(pFloatR[j] * pFloatR[j] + pFloatI[j] * pFloatI[j]));
				// }
				// pFloat[j] = atan2(pFloatI[j], pFloatR[j]);
			}   // CV_32F
			else
			{
				if (pDoubleI[j] / (pDoubleR[j] + pDoubleI[j]) > 0.99)
				{
					pDouble[j] = CV_PI / 2;
				}
				else
				{
					pDouble[j] = atan(pDoubleI[j] / pDoubleR[j]);
				}
				// pDouble[j]=atan2(pDoubleI[j],pDoubleR[j]);
			}   // CV_64F
		}
	}
	return kernel;
}

Mat GaborFR::getFilterRealPart(Mat &src, Mat &real)
{
	CV_Assert(real.type() == src.type());
	Mat dst, kernel;
	flip(real, kernel, -1);  // kernel mirror
	filter2D(src, dst, CV_32F, kernel, Point(-1, -1), 0, BORDER_REPLICATE);  // BORDER_CONSTANT, BORDER_DEFAULT
	return dst;
}

Mat GaborFR::getFilterImagePart(Mat &src, Mat &imaginary)
{
	CV_Assert(imaginary.type() == src.type());
	Mat dst, kernel;
	flip(imaginary, kernel, -1);  // kernel mirror
	filter2D(src, dst, CV_32F, kernel, Point(-1, -1), 0, BORDER_REPLICATE);  // BORDER_CONSTANT, BORDER_DEFAULT
	return dst;
}

void GaborFR::getFilterRealImagePart(Mat &src, Mat &real, Mat &imaginary, Mat &outReal, Mat &outImaginary)
{
	outReal = getFilterRealPart(src, real);
	outImaginary = getFilterImagePart(src, imaginary);
}

void GaborFR::normPush(Mat &mreal, Mat &showreal, Mat &linereal)
{
	// Mreal = mreal;
	// resize(Mreal, Mreal, Size(100, 100));
	// normalize value of pixel from 255 to 0
	normalize(mreal, mreal, 0, 255, CV_MINMAX, CV_8U);
	showreal.push_back(mreal);
	linereal = Mat::ones(4, mreal.cols, mreal.type()) * 255;
	showreal.push_back(linereal);
}

void GaborFR::toShow(Mat &showreal, String windowname)
{
	//namedWindow(windowName);  // 恢復正常大小
	namedWindow(windowname, CV_WINDOW_NORMAL);  // create a window, 可以滑鼠隨意拖動視窗改變大小(CV_WINDOW_NORMAL=0)
	imshow(windowname, showreal);  // show the image inside the created window
}


/*
Mat GaborFR::gaborFilter(Mat &img, Mat &filter)
{
	int half_filter_size = (max(filter.rows, filter.cols) - 1) / 2;
	Mat filtered_img(img.rows, img.cols, CV_32F);
	for (int i = 0; i < img.rows; i++) {
		uchar *img_p = img.ptr<uchar>(i);
		float *img_f = filtered_img.ptr<float>(i);
		for (int j = 0; j < img.cols; j++) {
			float filter_value = 0.0f;
			for (int fi = 0; fi < filter.rows; fi++) {
				float *f = filter.ptr<float>(fi);
				int img_i = i + fi - half_filter_size;
				img_i = img_i < 0 ? 0 : img_i;
				img_i = img_i >= img.rows ? (img.rows - 1) : img_i;
				uchar *p = img.ptr<uchar>(img_i);
				for (int fj = 0; fj < filter.cols; fj++) {
					int img_j = j + fj - half_filter_size;
					img_j = img_j < 0 ? 0 : img_j;
					img_j = (img_j >= img.cols) ? (img.cols - 1) : img_j;
					float tmp = (float)p[img_j] * f[fj];
					filter_value += tmp;
				}
			}
			img_f[j] = filter_value;
		}
	}
	return filtered_img;
}

Mat GaborFR::getGaborFilter(double lambda, double theta, double sigma2, double gamma, double psi)
{
	if (abs(lambda - 0.0f) < 1e-6) {
		lambda = 1.0f;
	}
	double sigma_x = sigma2;
	double sigma_y = sigma2 / (gamma * gamma);
	int nstds = 3;
	double sqrt_sigma_x = sqrt(sigma_x);
	double sqrt_sigma_y = sqrt(sigma_y);
	double xmax = max(abs(nstds * sqrt_sigma_x * cos(theta)), abs(nstds * sqrt_sigma_y * sin(theta)));
	double ymax = max(abs(nstds * sqrt_sigma_x * sin(theta)), abs(nstds * sqrt_sigma_y * cos(theta)));
	int half_filter_size = xmax > ymax ? xmax : ymax;
	int filter_size = 2 * half_filter_size + 1;
	Mat gaber = Mat::zeros(filter_size, filter_size, CV_32F);
	for (int i = 0; i < filter_size; i++) {
		double *f = gaber.ptr<double>(i);
		for (int j = 0; j < filter_size; j++) {
			int x = j - half_filter_size;
			double y = i - half_filter_size;
			double x_theta = x * cos(theta) + y * sin(theta);
			double y_theta = -x * sin(theta) + y * cos(theta);
			f[x] = exp(-.5 * (x_theta * x_theta / sigma_x + y_theta * y_theta / sigma_y));
			f[x] = f[x] * cos(2 * CV_PI * x_theta / lambda + psi);
		};
	}
	return gaber;
}

Mat GaborFR::normalizeFilterShow(Mat gaber)
{
	Mat gaber_show = Mat::zeros(gaber.rows, gaber.cols, CV_8UC1);
	float gaber_max = FLT_MIN;
	float gaber_min = FLT_MAX;
	for (int i = 0; i < gaber.rows; i++) {
		float *f = gaber.ptr<float>(i);
		for (int j = 0; j < gaber.cols; j++) {
			if (f[j] > gaber_max) {
				gaber_max = f[j];
			}
			if (f[j] < gaber_min) {
				gaber_min = f[j];
			}
		}
	}
	float gaber_max_min = gaber_max - gaber_min;
	for (int i = 0; i < gaber_show.rows; i++) {
		uchar *p = gaber_show.ptr<uchar>(i);
		float *f = gaber.ptr<float>(i);
		for (int j = 0; j < gaber_show.cols; j++) {
			if (gaber_max_min != 0.0f) {
				float tmp = (f[j] - gaber_min) * 255.0f / gaber_max_min;
				p[j] = (uchar)tmp;
			}
			else {
				p[j] = 255;
			}
		}
	}
	return gaber_show;
}
*/