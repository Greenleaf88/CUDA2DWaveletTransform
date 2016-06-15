#ifndef WAVELETCUDA_H
#define WAVELETCUDA_H

//��ά����С���任
extern "C" void Dwt2D_CUDA
	(
	int level,int *length,
	double *originData,int data_width,int data_height,
	double *lp_filter,double *hp_filter,int filter_len,
	double *output,int total_size
	);

//��ά����С���任
extern "C" void iDwt2D_CUDA
	(
	int level,int *length,
	double *data,int data_size,
	double *lpfilter,double *hpfilter,int filter_len,
	double *output
	);


//��˹�˲�
extern "C" void GaussianFilt_CUDA(double *data,int *length,int orign_height,int dwtlevel,double sigma);


#endif