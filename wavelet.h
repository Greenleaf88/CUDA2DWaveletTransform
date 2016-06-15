#ifndef WAVELET_H
#define WAVELET_H
#include <vector>
using namespace std;

//小波变换正变换
void Dwt2d_GPU(vector<double> &origsig,int sig_width,int sig_height,int level, string nm, 
			   vector<double> &dwt_output, vector<int> &length) ;

//小波变换逆变换
void iDwt2d_GPU(vector<double> &dwtop,int level, string nm,
				vector<double> &idwt_output, vector<int> &length);

//高斯滤波
void GaussianFilt_GPU(vector<double> &dwt_data,vector<int> &length,int orign_height,int dwtlevel,double sigma);

//取滤波器系数
int filtcoef(string , vector<double> &, vector<double> &, vector<double> &,
			 vector<double> &);
#endif