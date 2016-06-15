#ifndef WAVELET_H
#define WAVELET_H
#include <vector>
using namespace std;

//С���任���任
void Dwt2d_GPU(vector<double> &origsig,int sig_width,int sig_height,int level, string nm, 
			   vector<double> &dwt_output, vector<int> &length) ;

//С���任��任
void iDwt2d_GPU(vector<double> &dwtop,int level, string nm,
				vector<double> &idwt_output, vector<int> &length);

//��˹�˲�
void GaussianFilt_GPU(vector<double> &dwt_data,vector<int> &length,int orign_height,int dwtlevel,double sigma);

//ȡ�˲���ϵ��
int filtcoef(string , vector<double> &, vector<double> &, vector<double> &,
			 vector<double> &);
#endif