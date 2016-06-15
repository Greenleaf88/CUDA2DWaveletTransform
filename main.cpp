#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath> 
#include <algorithm>
#include <time.h>
#include "wavelet.h"
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include <stdio.h>
#include <string.h>
using namespace std;
using namespace cv;

int main()  
{  
	IplImage* img = cvLoadImage("2048.bmp",0);
	//IplImage* img = cvLoadImage("4096.bmp",0);
	//IplImage* img = cvLoadImage("large.jpg",0);
	if (!img)
	{
		cout << " Can't read Image. Try Different Format." << endl;
		exit(1);
	}

	int height = img->height;
	int width = img->width;
	int nc = img->nChannels;
	int pix_depth = img->depth;

	cout << "depth" << pix_depth <<  "Channels" << nc << endl;

	Mat matimg(img);//把图片转化成矩阵

	vector<double> vecdata;
	int k=1;
	for (int i=0; i < height; i++) 
	{
		for (int j =0; j < width; j++)
		{
			unsigned char temp;
			temp = ((uchar*) matimg.data + i * matimg.step)[j  * matimg.elemSize() + k ];
			vecdata.push_back((double)temp);
		}
	}

	string nm = "sym10";
	double sigma=10;
	int J =5;
	vector<int> length;
	vector<double> output;

	//-----------------------------------------------小波分解--------------------------------------------------------------------
	clock_t t1;
	t1=clock();
	FILE* fp=fopen("time.txt","a");
	fprintf(fp,"小波类型是sym10\n");
	fprintf(fp,"图片分辨率是2048\n");
	Dwt2d_GPU(vecdata,width,height,J,nm,output,length);
	t1=clock()-t1;
	fprintf(fp,"正变换执行时间:%f ms\n",(float)t1);	



	//-----------------------------------------------高斯滤波--------------------------------------------------------------------
	//GaussianFilt_GPU(output,length,height,J,sigma);


	//-----------------------------------------------小波重构--------------------------------------------------------------------
	vector<double> idwt_output_gpu;
	t1=clock();
	iDwt2d_GPU(output,J,nm,idwt_output_gpu,length);
	t1 = clock()-t1;	
	fprintf(fp,"逆变换执行时间:%f ms\n",(float)t1);
	fclose(fp);

	//-----------------------------------------------图像显示--------------------------------------------------------------------
	IplImage *dvImg;
	CvSize dvSize; // size of output image

	dvSize.width = width;
	dvSize.height = height;
	dvImg = cvCreateImage( dvSize, 8, 1 );

	CvMat* mat1;
	mat1=cvCreateMat(dvSize.height,dvSize.width,CV_64FC1);

	for (int i = 0; i < dvSize.height; i++ )
	{
		for (int j = 0; j < dvSize.width; j++ )
		{
			double temp=idwt_output_gpu[i*dvSize.width+j];
			if(temp<0)
				temp=0;
			else if(temp>255)
				temp=255;
			CV_MAT_ELEM(*mat1, double, i, j)=temp;
		}
	}


	for (int i = 0; i < dvSize.height; i++ )
	{
		for (int j = 0; j < dvSize.width; j++ )
			((uchar*)(dvImg->imageData + dvImg->widthStep*i))[j] =(char) (CV_MAT_ELEM(*mat1, double, i, j));
	}			
	cvReleaseMat(&mat1);

	cvNamedWindow( "Reconstructed Image", 1 ); 
	cvShowImage( "Reconstructed Image", dvImg ); 
	cvWaitKey();
	cvDestroyWindow("Reconstructed Image");
	cvSaveImage("recon.bmp",dvImg);
	return 0; 
}  


