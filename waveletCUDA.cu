
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include "cufft.h"
using namespace std;

#define TILE_X 16
#define TILE_Y 16
#define MARGIN 35

//���ڴ洢�˲�ϵ���ĳ����ڴ�
__constant__ double c_lpFilter[MARGIN];
__constant__ double c_hpFilter[MARGIN];
__constant__ double c_iLpFilter[MARGIN];
__constant__ double c_iHpFilter[MARGIN];

//-------------------------------------------------------------------���任����------------------------------------------------------------------------------------------------

//�Զ�ά���ݵ��н���С���任һά����,����������lp_output,hp_output���Ѿ�ת�ù���Ĵ�С,��һ���˺����н�����չ��������²�����ת��
__global__ void Dwt1D_Row(double *data,int width,int height,size_t pitch_data,int filter_len,
						  double *lp_output,size_t pitch_lpout,double *hp_output,size_t pitch_hpout)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
	int row=threadIdx.y+blockIdx.y*blockDim.y;

	int aftWidth=(width+filter_len-1)/2;//���+��������ĳ���
	if(col>=aftWidth||row>=height)
		return;

	double* row_data=(double*)((char*)data+row*pitch_data);//�������ݵ���ʼ�е�ַ
	double* row_lpout=(double*)((char*)lp_output+col*pitch_lpout);//��ͨ�˲��������ʼ�е�ַ
	double* row_hpout=(double*)((char*)hp_output+col*pitch_hpout);//��ͨ�˲��������ʼ�е�ַ

	int symIndex=filter_len+2*col;//ֻ����ֽ��һ��������ֻ����������������֣�tmpIndex�ǵõ����������Ǹ����ڶԳ���չ���е�λ��
	int oriIndex=symIndex-filter_len+1;//symIndex-(filter_len-1),���ڼ�����Ҫȡ��������ԭʼ�����е�λ��,2*col+1

	double lp_result=0;//��ͨ�˲����ľ�����
	double hp_result=0;//��ͨ�˲����ľ�����

#pragma unroll
	for(int i=0;i<filter_len;i++)
	{
		int index=oriIndex-i;
		double tmpValue;//�洢����������������ݵ�ֵ
		if(index<0)
		{
			tmpValue=row_data[-index-1];//-1ȡ0��-2ȡ1��-3ȡ2��-4ȡ3��-5ȡ4,|index|-1
		}
		else
		{
			if(index>=width)
				tmpValue=row_data[2*width-index-1];//lastIndex-[(index-lastIndex)-1];(width-1)-{[index-(width-1)]-1}
			else
				tmpValue=row_data[index];
		}
		lp_result+=tmpValue*c_lpFilter[i];
		hp_result+=tmpValue*c_hpFilter[i];
	}
	
	//���ֱ�ӽ���ת��
	row_lpout[row]=lp_result;//�������Ŀ���Ǿ�������һ�룬�߶Ȳ���
	row_hpout[row]=hp_result;
}

//�Զ�ά���ݵ��н���С���任һά���㣬�����з����ϵ����������һ����������������������һ���ģ����մ�СҲ��һ���ģ����Կ��԰�������������̺ϲ���һ���˺����У�
//��������һ�������������ת�ú�Ľ�������һ�������ڴ����һ�ּ����CLL��һ�������յĴ��λ�á�
//width���Ѿ�ת�ú����Ŀ�ȣ�height���Ѿ�ת�ú����ĸ߶ȣ��˺����Ĳ��������п�Ⱥ͸߶ȷֱ���width��heightΪ����,
__global__ void Dwt1D_Col(double *lpOutput,size_t pitch_lpout,double *hpOutput,size_t pitch_hpout,int width,int height,
						  int filter_len,double *CLL,size_t pitch_cll,double *output,int offset)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
	int row=threadIdx.y+blockIdx.y*blockDim.y;

	int aftWidth=(width+filter_len-1)/2;//���+��������ĳ���
	if(col>=aftWidth||row>=height)
		return;

	double* row_lpData=(double*)((char*)lpOutput+row*pitch_lpout);//�������ݵ���ʼ�е�ַ
	double* row_hpData=(double*)((char*)hpOutput+row*pitch_hpout);//�������ݵ���ʼ�е�ַ
	double* row_cll=(double*)((char*)CLL+col*pitch_cll);//CLL�������ʼ�е�ַ

	int symIndex=filter_len+2*col;//ֻ����ֽ��һ��������ֻ����������������֣�tmpIndex�ǵõ����������Ǹ����ڶԳ���չ���е�λ��
	int oriIndex=symIndex-filter_len+1;//symIndex-(filter_len-1),���ڼ�����Ҫȡ��������ԭʼ�����е�λ��,2*col+1

	double cll_result=0;//CLL�Ľ��
	double clh_result=0;//CLH�Ľ��
	double chl_result=0;//CLL�Ľ��
	double chh_result=0;//CLH�Ľ��

#pragma unroll
	for(int i=0;i<filter_len;i++)
	{
		int index=oriIndex-i;
		double lpTmpValue;//�洢�������������lpOoutput���ݵ�ֵ
		double hpTmpValue;//�洢�������������hpOoutput���ݵ�ֵ
		if(index<0)
		{
			lpTmpValue=row_lpData[-index-1];//-1ȡ0��-2ȡ1��-3ȡ2��-4ȡ3��-5ȡ4,|index|-1
			hpTmpValue=row_hpData[-index-1];
		}
		else
		{
			if(index>=width)
			{
				lpTmpValue=row_lpData[2*width-index-1];//lastIndex-[(index-lastIndex)-1];(width-1)-{[index-(width-1)]-1}
				hpTmpValue=row_hpData[2*width-index-1];
			}			
			else
			{
				lpTmpValue=row_lpData[index];
				hpTmpValue=row_hpData[index];
			}
				
		}
		cll_result+=lpTmpValue*c_lpFilter[i];
		clh_result+=lpTmpValue*c_hpFilter[i];
		chl_result+=hpTmpValue*c_lpFilter[i];
		chh_result+=hpTmpValue*c_hpFilter[i];
	}

	//���ֱ�ӽ���ת��
	//offset=totalsize-sum_step���ο���һ�����Wavelet�е���������
	row_cll[row]=cll_result;
	output[offset-3*aftWidth*height+col*height+row]=clh_result;
	output[offset-2*aftWidth*height+col*height+row]=chl_result;
	output[offset-aftWidth*height+col*height+row]=chh_result;
}

//��������С���任
void Dwt2D(double *data,int data_width,int data_height,size_t pitch_data,int filter_len,double *CLL,size_t pitch_cll,double *output,int offset)
{
	//----------------------��һ�׶Σ��ȶ�������ά���ݽ����з����ϵı任----------------------------------------------------
	//���ڵ�һ����������������ת�ú��������󣬹��������ڴ�ʱֱ�Ӱ���ת�ú�Ĵ�С����
	int height=(data_width+filter_len-1)/2;//ת�ú�ĸ߶ȣ���ԭ���
	int width=data_height;//ת�ú�Ŀ�ȣ���ԭ�߶�

	//������������������
	double *lpOutput;
	double *hpOutput;
	size_t pitch_lpout;
	size_t pitch_hpout;
	cudaMallocPitch((void **)&lpOutput,&pitch_lpout,sizeof(double)*width,height);
	cudaMallocPitch((void **)&hpOutput,&pitch_hpout,sizeof(double)*width,height);

	dim3 threads(TILE_X,TILE_Y);
	dim3 blocks_row((height+TILE_X-1)/TILE_X,(width+TILE_Y-1)/TILE_Y);//��ԭ��Ⱥ�ԭ�߶���ȷ��grid��С
	Dwt1D_Row<<<blocks_row,threads>>>(data,data_width,data_height,pitch_data,filter_len,lpOutput,pitch_lpout,hpOutput,pitch_hpout);

	//----------------------�ڶ��׶Σ���֮ǰ�õ��������Ѿ�ת�ú�ľ�������з����ϵı任----------------------------------------
	int aftwidth=(width+filter_len-1)/2;
	dim3 blocks_col((aftwidth+TILE_X-1)/TILE_X,(height+TILE_Y-1)/TILE_Y);
	Dwt1D_Col<<<blocks_col,threads>>>(lpOutput,pitch_lpout,hpOutput,pitch_hpout,width,height,filter_len,CLL,pitch_cll,output,offset);

	//�ͷ�
	cudaFree(lpOutput);
	cudaFree(hpOutput);
}


//��ά����С���任
extern "C" void Dwt2D_CUDA(	int level,int *length,
	double *originData,int data_width,int data_height,
	double *lpfilter,double *hpfilter,int filter_len,
	double *output,int total_size)
{
	int height=data_height;
	int width=data_width;

	int sum_step=0;
	double *d_output;
	cudaMalloc((void **)&d_output,sizeof(double)*total_size);

	//----------------------�Ѷ�ά���ݸ��Ƶ�GPU��------------------------------------------------------
	double *d_data;
	size_t pitch_data;
	cudaMallocPitch((void **)&d_data,&pitch_data,sizeof(double)*width,height);

	cudaMemcpy2D(d_data,pitch_data,originData,sizeof(double)*width,sizeof(double)*width,height,cudaMemcpyHostToDevice);	//����֮ǰ��sizeof(double)*width��д��*height��������ݴ������
	cudaMemcpyToSymbol(c_lpFilter, lpfilter, filter_len*sizeof(double));
	cudaMemcpyToSymbol(c_hpFilter, hpfilter, filter_len*sizeof(double));
	//----------------------���ݸ��ƽ׶����-----------------------------------------------------------

	//----------------------����J��С���ֽ�-----------------------------------------------------------
	for(int iter=0;iter<level;iter++)
	{
		int next_width=(width+filter_len-1)/2;
		int next_height=(height+filter_len-1)/2;

		//�洢ÿһ��任��ĳ���
		length[iter*2]=next_height;
		length[iter*2+1]=next_width;
		int offset=total_size-sum_step;//CLH,CHL,CHH��output�е����λ��

		double *cLL;//�洢������һ��С���任�ĳ�ʼ����
		size_t pitch_cll;
		cudaMallocPitch((void **)&cLL,&pitch_cll,sizeof(double)*next_width,next_height);

		//ִ�ж�ά�任
		Dwt2D(d_data,width,height,pitch_data,filter_len,cLL,pitch_cll,d_output,offset);

		width=next_width;
		height=next_height;

		//��һ��ѭ���ĳ�ʼ����
		cudaFree(d_data);
		d_data=cLL;
		pitch_data=pitch_cll;

		//�洢���
		if(iter==level-1)
		{
			cudaMemcpy2D((d_output+offset-4*next_width*next_height),sizeof(double)*next_width,cLL,pitch_cll,sizeof(double)*next_width,next_height,cudaMemcpyDeviceToDevice);
			cudaFree(cLL);
		}
		sum_step+=next_width*next_height*3;
	}
	//----------------------J��С���ֽ����-----------------------------------------------------------
	cudaMemcpy(output,d_output,sizeof(double)*total_size,cudaMemcpyDeviceToHost);
	cudaFree(d_data);
	cudaFree(d_output);
}




//-------------------------------------------------------------------��任����------------------------------------------------------------------------------------------------



//С����任���з����ϵ�����
//��֮ǰ�ķֽ����е�����ȡ��CLL���ٴ�ȫ���ֽ�����ͨ��offset����ȡCLH��CHL��CHH�������ֵ����ݣ�offset��һ��ֽ������ܴ�С��
//��ȡ����ʱֱ��һ��һ�е�ȡ���൱��ת�ú�������㣬���������൱��ת�ú��������У�����������з����ϵļ��㣬��������з���
//�ϵļ��㲻��Ҫ�ٽ���ת�ã������ú˺����Ĳ���ʱ��blocks�Ŀ������nextHeight���߶���ԭwidthΪ���ݣ�����ת�ú�Ľ�������С����
//���е�width,height��ת�ú�ģ��൱��ԭheight,ԭwidth
//CLL����ʹ��Pitch����ʽ����ΪҪ����ȡ�����ʺ�Pitch����ʽ��Ҳ��Ϊ���ԭ��iDwt1D_row��outputҲ������Pitch����ʽ
__global__ void iDwt1D_Col(double *CLL,double *data,int offset,int width,int height,int nextWidth,int filter_len,
						   double *app,size_t pitch_app,double *detail,size_t pitch_detail)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
 	int row=threadIdx.y+blockIdx.y*blockDim.y;

	if(col>=nextWidth||row>=height)
		return;

	//ʣ�������������ݵ���ʼλ��
	int clh_begin=offset;
	int chl_begin=offset+width*height;
	int chh_begin=offset+2*width*height;

	//�������
	double* row_appData=(double*)((char*)app+col*pitch_app);//��ʼ�е�ַ��col�൱��ת�ú���к�
	double* row_detailData=(double*)((char*)detail+col*pitch_detail);//��ʼ�е�ַ��col�൱��ת�ú���к�

	int oriIndex=col+filter_len-2;//�������������ʼ�������ϲ������λ��
	
	double cll_result=0;
	double clh_result=0;
	double chl_result=0;
	double chh_result=0;

#pragma unroll
	for(int i=0;i<filter_len;i++)
	{
		int index=oriIndex-i;
		double cllTmpValue=0;
		double clhTmpValue=0;
		double chlTmpValue=0;
		double chhTmpValue=0;

		//��С���ֽ�������ȡ���ݲ���Ҫ�򵥣���λ�ò����ϲ�����������У���ȡ0�����ڣ����������Ƿ���ż������ż������ԭʼ������ȡ������/2��λ�õ�����
		//������������ֱ��ȡ0
		if(index>=0&&index<(2*width-1)&&(index%2)==0)
		{
			cllTmpValue=CLL[index/2*height+row];//һ��һ�е�ȡ����
			clhTmpValue=data[clh_begin+index/2*height+row];
			chlTmpValue=data[chl_begin+index/2*height+row];
			chhTmpValue=data[chh_begin+index/2*height+row];
		}
		else
		{
			cllTmpValue=0;
			clhTmpValue=0;
			chlTmpValue=0;
			chhTmpValue=0;
		}

		cll_result+=cllTmpValue*c_iLpFilter[i];
		clh_result+=clhTmpValue*c_iHpFilter[i];
		chl_result+=chlTmpValue*c_iLpFilter[i];
		chh_result+=chhTmpValue*c_iHpFilter[i];
	}

	//�൱����ת�ú���������
	row_appData[row]=cll_result+clh_result;
	row_detailData[row]=chl_result+chh_result;
}


//С����任���з����ϵ�����
//���з����ϵ�����õ������������ֱ���ٽ����з����ϵľ�����㣬���ɻ�ԭ������ݣ��˺����Ĳ�����blocks�Ŀ�Ⱥ͸߶ȷֱ���nextWidth,
//nextHeightΪ���ݣ����������������һ�������˲���ת��
__global__ void iDwt1D_Row(double *app,size_t pitch_app,double *detail,size_t pitch_detail,int width,int height,
						   int nextWidth,int filter_len,double *output)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
 	int row=threadIdx.y+blockIdx.y*blockDim.y;

	if(col>=nextWidth||row>=height)
		return;

	//�������ݺ���������е���ʼλ��
	double* row_appData=(double*)((char*)app+row*pitch_app);
	double* row_detailData=(double*)((char*)detail+row*pitch_detail);

	int oriIndex=col+filter_len-2;
	
	double app_result=0;
	double detail_result=0;

#pragma unroll
	for(int i=0;i<filter_len;i++)
	{
		int index=oriIndex-i;
		double appTmpValue=0;
		double detailTmpValue=0;

		if(index>=0&&index<(2*width-1)&&(index%2)==0)
		{
			appTmpValue=row_appData[index/2];
			detailTmpValue=row_detailData[index/2];
		}
		else
		{
			appTmpValue=0;
			detailTmpValue=0;
		}

		app_result+=appTmpValue*c_iLpFilter[i];
		detail_result+=detailTmpValue*c_iHpFilter[i];
	}
	output[row*nextWidth+col]=app_result+detail_result;
}


//��ά����С���任
extern "C" void iDwt2D_CUDA
(
	int level,int *length,
	double *data,int data_size,
	double *lpfilter,double *hpfilter,int filter_len,
	double *output
)
{
	int height=length[0];
	int width=length[1];
	int offset=height*width;

	double *CLL;
	cudaMalloc((void **)&CLL,sizeof(double)*width*height);

	//----------------------------------���ݸ��ƽ׶�---------------------------------------------
	double *d_data;
	cudaMalloc((void **)&d_data,sizeof(double)*data_size);

	cudaMemcpy(d_data,data,sizeof(double)*data_size,cudaMemcpyHostToDevice);
	cudaMemcpy(CLL,d_data,sizeof(double)*width*height,cudaMemcpyDeviceToDevice);//��ʼCLL�������
	cudaMemcpyToSymbol(c_iLpFilter, lpfilter, filter_len*sizeof(double));
	cudaMemcpyToSymbol(c_iHpFilter, hpfilter, filter_len*sizeof(double));
	//----------------------------------���ݸ��ƽ׶����------------------------------------------

	
	//----------------------------------J��С���ع�------------------------------------------------
	dim3 threads(TILE_X,TILE_Y);
	for(int iter=0;iter<level;iter++)
	{
		//��ǰ��ĸ߿�
		int tmp_height=length[2*iter];
		int tmp_width=length[2*iter + 1];
		//��һ��ĸ߿�
		int next_height=length[2*iter+2];
		int next_width=length[2*iter+3];

		//----------------------��һ�׶Σ��Ƚ����з����ϵ���任---------------------------------
		double *app;
		double *detail;
		size_t pitch_app;
		size_t pitch_detail;
		cudaMallocPitch((void **)&app,&pitch_app,sizeof(double)*tmp_width,next_height);
		cudaMallocPitch((void **)&detail,&pitch_detail,sizeof(double)*tmp_width,next_height);
		
		dim3 blocks_col((next_height+TILE_X-1)/TILE_X,(tmp_width+TILE_Y-1)/TILE_Y);
		iDwt1D_Col<<<blocks_col,threads>>>(CLL,d_data,offset,tmp_height,tmp_width,next_height,filter_len,app,pitch_app,detail,pitch_detail);

		//----------------------�ڶ��׶Σ������з����ϵ���任-------------------------------------
		cudaFree(CLL);
		cudaMalloc((void **)&CLL,sizeof(double)*next_height*next_width);//���֮ǰ��CLL���ݣ����������CLL����Ϊ��һ��ѭ���ĳ�ʼ����
		dim3 blocks_row((next_width+TILE_X-1)/TILE_X,(next_height+TILE_Y-1)/TILE_Y);
		iDwt1D_Row<<<blocks_row,threads>>>(app,pitch_app,detail,pitch_detail,tmp_width,next_height,next_width,filter_len,CLL);
		offset+=3*tmp_height*tmp_width;	

		cudaFree(app);
		cudaFree(detail);
	}
	//-----------------------------J��С���ع�����--------------------------------------------------

	//���ƻ�����
	height=length[2*level];
	width=length[2*level+1];
	cudaMemcpy(output,CLL,sizeof(double)*height*width,cudaMemcpyDeviceToHost);
	cudaFree(CLL);
	cudaFree(d_data);
}



/*
//���ڻ�ȡת�ú������
__global__ void getTransposeData_shared(double *CLL,size_t pitch_CLL,double *data,int offset,int width,int height,
									    double *tr_cll,size_t pitch_cll,double *tr_clh,size_t pitch_clh,
									    double *tr_chl,size_t pitch_chl,double *tr_chh,size_t pitch_chh)
{
	int col=blockIdx.x*blockDim.x+threadIdx.x;//width
	int row=blockIdx.y*blockDim.y+threadIdx.y;//height
	if(col>=width||row>=height)
		return;

	int clh_begin=offset;
 	int chl_begin=offset+width*height;
 	int chh_begin=offset+2*width*height;


	double* row_CLL=(double*)((char*)CLL+row*pitch_CLL);
	//�����ʼ�е�ַ
	double* row_cll=(double*)((char*)tr_cll+col*pitch_cll);
	double* row_clh=(double*)((char*)tr_clh+col*pitch_clh);
	double* row_chl=(double*)((char*)tr_chl+col*pitch_chl);
	double* row_chh=(double*)((char*)tr_chh+col*pitch_chh);

	__shared__ double s_Data[TILE_Y][TILE_X*4];

	s_Data[threadIdx.y][threadIdx.x]=row_CLL[col];
	s_Data[threadIdx.y][threadIdx.x+TILE_X]=data[clh_begin+row*width+col];
	s_Data[threadIdx.y][threadIdx.x+TILE_X*2]=data[chl_begin+row*width+col];
	s_Data[threadIdx.y][threadIdx.x+TILE_X*3]=data[chh_begin+row*width+col];

	__syncthreads();

	row_cll[row]=s_Data[threadIdx.y][threadIdx.x];
	row_clh[row]=s_Data[threadIdx.y][threadIdx.x+TILE_X];
	row_chl[row]=s_Data[threadIdx.y][threadIdx.x+TILE_X*2];
	row_chh[row]=s_Data[threadIdx.y][threadIdx.x+TILE_X*3];
}

//С����任���з����ϵ�����
__global__ void iDwt1D_Col(double *CLL,size_t pitch_cll,double *CLH,size_t pitch_clh,double *CHL,size_t pitch_chl,double *CHH,size_t pitch_chh,
						   int width,int height,int nextWidth,int filter_len,
						   double *app,size_t pitch_app,double *detail,size_t pitch_detail)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
 	int row=threadIdx.y+blockIdx.y*blockDim.y;

	if(col>=nextWidth||row>=height)
		return;

	//�������ݵ���ʼ�е�ַ
	double* row_CLLData=(double*)((char*)CLL+row*pitch_cll);
	double* row_CLHData=(double*)((char*)CLH+row*pitch_clh);
	double* row_CHLData=(double*)((char*)CHL+row*pitch_chl);
	double* row_CHHData=(double*)((char*)CHH+row*pitch_chh);

	//������ݵ���ʼ�е�ַ��col�൱��ת�ú���к�
	double* row_appData=(double*)((char*)app+col*pitch_app);
	double* row_detailData=(double*)((char*)detail+col*pitch_detail);

	int oriIndex=col+filter_len-2;//�������������ʼ�������ϲ������λ��
	
	double cll_result=0;
	double clh_result=0;
	double chl_result=0;
	double chh_result=0;

#pragma unroll
	for(int i=0;i<filter_len;i++)
	{
		int index=oriIndex-i;
		double cllTmpValue=0;
		double clhTmpValue=0;
		double chlTmpValue=0;
		double chhTmpValue=0;

		//��С���ֽ�������ȡ���ݲ���Ҫ�򵥣���λ�ò����ϲ�����������У���ȡ0�����ڣ����������Ƿ���ż������ż������ԭʼ������ȡ������/2��λ�õ�����
		//������������ֱ��ȡ0
		if(index>=0&&index<(2*width-1)&&(index%2)==0)
		{
			cllTmpValue=row_CLLData[index/2];
			clhTmpValue=row_CLHData[index/2];
			chlTmpValue=row_CHLData[index/2];
			chhTmpValue=row_CHHData[index/2];
		}
		else
		{
			cllTmpValue=0;
			clhTmpValue=0;
			chlTmpValue=0;
			chhTmpValue=0;
		}
		cll_result+=cllTmpValue*c_iLpFilter[i];
		clh_result+=clhTmpValue*c_iHpFilter[i];
		chl_result+=chlTmpValue*c_iLpFilter[i];
		chh_result+=chhTmpValue*c_iHpFilter[i];
	}

	//�൱����ת�ú���������
	row_appData[row]=cll_result+clh_result;
	row_detailData[row]=chl_result+chh_result;
}

//С����任���з����ϵ�����
__global__ void iDwt1D_Row(double *app,size_t pitch_app,double *detail,size_t pitch_detail,int width,int height,
						   int nextWidth,int filter_len,double *output,size_t pitch_out)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
 	int row=threadIdx.y+blockIdx.y*blockDim.y;

	if(col>=nextWidth||row>=height)
		return;

	//�������ݺ���������е���ʼλ��
	double* row_appData=(double*)((char*)app+row*pitch_app);
	double* row_detailData=(double*)((char*)detail+row*pitch_detail);
	double* row_output=(double*)((char*)output+row*pitch_out);

	int oriIndex=col+filter_len-2;
	
	double app_result=0;
	double detail_result=0;

#pragma unroll
	for(int i=0;i<filter_len;i++)
	{
		int index=oriIndex-i;
		double appTmpValue=0;
		double detailTmpValue=0;

		if(index>=0&&index<(2*width-1)&&(index%2)==0)
		{
			appTmpValue=row_appData[index/2];
			detailTmpValue=row_detailData[index/2];
		}
		else
		{
			appTmpValue=0;
			detailTmpValue=0;
		}

		app_result+=appTmpValue*c_iLpFilter[i];
		detail_result+=detailTmpValue*c_iHpFilter[i];
	}
	row_output[col]=app_result+detail_result;
}

//��ά����С���任
extern "C" void iDwt2D_CUDA
(
	int level,int *length,
	double *data,int data_size,
	double *lpfilter,double *hpfilter,int filter_len,
	double *output
)
{
	int height=length[0];
	int width=length[1];
	int offset=height*width;

	double *CLL;
	size_t pitch_CLL;
	cudaMallocPitch((void **)&CLL,&pitch_CLL,sizeof(double)*width,height);

	//----------------------------------���ݸ��ƽ׶�---------------------------------------------
	double *d_data;
	cudaMalloc((void **)&d_data,sizeof(double)*data_size);

	cudaMemcpy(d_data,data,sizeof(double)*data_size,cudaMemcpyHostToDevice);
	cudaMemcpy2D(CLL,pitch_CLL,d_data,sizeof(double)*width,sizeof(double)*width,height,cudaMemcpyDeviceToDevice);//��ʼCLL�������
	cudaMemcpyToSymbol(c_iLpFilter, lpfilter, filter_len*sizeof(double));
	cudaMemcpyToSymbol(c_iHpFilter, hpfilter, filter_len*sizeof(double));
	//----------------------------------���ݸ��ƽ׶����------------------------------------------

	
	//----------------------------------J��С���ع�------------------------------------------------
	dim3 threads(TILE_X,TILE_Y);
	for(int iter=0;iter<level;iter++)
	{
		//��ǰ��ĸ߿�
		int tmp_height=length[2*iter];
		int tmp_width=length[2*iter + 1];
		//��һ��ĸ߿�
		int next_height=length[2*iter+2];
		int next_width=length[2*iter+3];

		//----------------------��ȡת������-----------------------------------------------------
		double *cll;size_t pitch_cll;
		double *clh;size_t pitch_clh;
		double *chl;size_t pitch_chl;
		double *chh;size_t pitch_chh;
		cudaMallocPitch((void **)&cll,&pitch_cll,sizeof(double)*tmp_height,tmp_width);
		cudaMallocPitch((void **)&clh,&pitch_clh,sizeof(double)*tmp_height,tmp_width);
		cudaMallocPitch((void **)&chl,&pitch_chl,sizeof(double)*tmp_height,tmp_width);
		cudaMallocPitch((void **)&chh,&pitch_chh,sizeof(double)*tmp_height,tmp_width);
		dim3 blocks_trans((tmp_width+TILE_X-1)/TILE_X,(tmp_height+TILE_Y-1)/TILE_Y);
		getTransposeData_shared<<<blocks_trans,threads>>>(CLL,pitch_CLL,d_data,offset,tmp_width,tmp_height,
														  cll,pitch_cll,clh,pitch_clh,chl,pitch_chl,chh,pitch_chh);
		//----------------------��һ�׶Σ��Ƚ����з����ϵ���任---------------------------------
		double *app;
		double *detail;
		size_t pitch_app;
		size_t pitch_detail;
		cudaMallocPitch((void **)&app,&pitch_app,sizeof(double)*tmp_width,next_height);
		cudaMallocPitch((void **)&detail,&pitch_detail,sizeof(double)*tmp_width,next_height);
		
		dim3 blocks_col((next_height+TILE_X-1)/TILE_X,(tmp_width+TILE_Y-1)/TILE_Y);
		iDwt1D_Col<<<blocks_col,threads>>>(cll,pitch_cll,clh,pitch_clh,chl,pitch_chl,chh,pitch_chh,
										   tmp_height,tmp_width,next_height,filter_len,app,pitch_app,detail,pitch_detail);
		//������ͷ�
		cudaFree(cll);
		cudaFree(clh);
		cudaFree(chl);
		cudaFree(chh);

		//----------------------�ڶ��׶Σ������з����ϵ���任-------------------------------------
		double *out;size_t pitch_out;
		cudaMallocPitch((void **)&out,&pitch_out,sizeof(double)*next_width,next_height);
		dim3 blocks_row((next_width+TILE_X-1)/TILE_X,(next_height+TILE_Y-1)/TILE_Y);
		iDwt1D_Row<<<blocks_row,threads>>>(app,pitch_app,detail,pitch_detail,tmp_width,next_height,next_width,filter_len,out,pitch_out);
		offset+=3*tmp_height*tmp_width;	

		//������ͷ�
		cudaFree(app);
		cudaFree(detail);

		//-----------------------������һ���CLL��ʼ����----------------------------------------------
		cudaFree(CLL);
		CLL=out;
		pitch_CLL=pitch_out;

		if(iter==level-1)
		{
			cudaMemcpy2D(output,sizeof(double)*next_width,CLL,pitch_CLL,sizeof(double)*next_width,next_height,cudaMemcpyDeviceToHost);
			cudaFree(out);
		}
	}
	//-----------------------------J��С���ع�����--------------------------------------------------

	cudaFree(CLL);
	cudaFree(d_data);
}

*/



//-------------------------------------------------------------------��˹�˲�����------------------------------------------------------------------------------------------------


__global__ void damp(int height,int width,int filter_len,cufftDoubleComplex *d_inp_fft,double *d_filter)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(col>=width||row>=height||col>=filter_len)
		return;

	d_inp_fft[row*width+col].x*= d_filter[col];
	d_inp_fft[row*width+col].y*= d_filter[col];
}

//��˹�˲�
extern "C" void GaussianFilt_CUDA(double *dwt_data,int *length,int orig_height,int dwtlevel,double sigma)
{
	int rows,cols,begin,margin,end;
	int level=dwtlevel;
	begin=length[0]*length[1];

	for(int i=0;i<level;i++)
	{		
		rows=length[2*i];
		cols=length[2*i+1];
		margin=rows*cols;
		begin+=margin;
		end=begin+margin;
		//�˲���
		int filter_len=cols/2+1;
		double *filter=(double*)malloc(sizeof(double)*filter_len);

		for(int k=0;k<filter_len;k++)
				filter[k]=1-exp(-(double)(k)*(k)/(2*sigma*sigma));	

		double *d_filter;
		cudaMalloc((void**)&d_filter, sizeof(double)*filter_len);
		cudaMemcpy(d_filter, filter, sizeof(double)*filter_len, cudaMemcpyHostToDevice);
		free(filter);

		//�������ݲ���
		int size = sizeof(cufftDoubleReal)*rows*cols;
		cufftDoubleReal *inp,*d_inp;
		inp=(cufftDoubleReal *)malloc(size);
		cudaMalloc((void**)&d_inp, size);
	
		int tempindex=begin;
		for(int k=0;k<rows;k++)
		{
			for(int j=0;j<cols;j++)
			{
				inp[ j*rows+ k]=dwt_data[tempindex];	
				tempindex++;
			}
		}		
		cudaMemcpy(d_inp, inp, size, cudaMemcpyHostToDevice);

		cufftDoubleComplex *inp_fft;
		cufftHandle plan_forward,plan_backward;

		int half=rows/2+1;
		cudaMalloc((void**)&inp_fft, sizeof(cufftDoubleComplex)*(half)*cols);

		cufftPlan1d(&plan_forward, rows, CUFFT_D2Z, cols);
		cufftExecD2Z(plan_forward, d_inp,inp_fft);
		cufftDestroy(plan_forward);
		
		//damp<<<(half*cols+PITCH-1)/PITCH,PITCH>>>(half*cols,half,filter_len,inp_fft,d_filter);
		dim3 threads(TILE_X,TILE_Y);
		dim3 blocks((half+TILE_X-1)/TILE_X,(cols+TILE_Y-1)/TILE_Y);
		damp<<<blocks,threads>>>(cols,half,filter_len,inp_fft,d_filter);

		cufftPlan1d(&plan_backward, rows, CUFFT_Z2D, cols);
		cufftExecZ2D(plan_backward, inp_fft, d_inp);
		cufftDestroy(plan_backward);

		cudaMemcpy(inp, d_inp, size, cudaMemcpyDeviceToHost);

		tempindex=begin;
		for(int k=0;k<rows;k++)
		{
			for(int j=0;j<cols;j++)
				dwt_data[tempindex++]=inp[ j*rows+ k]/rows;			
		}
		begin=end+margin;

		free(inp);
		cudaFree(d_inp);
		cudaFree(inp_fft);
		cudaFree(d_filter);	
	}	
	
}

