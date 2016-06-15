
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

//用于存储滤波系数的常量内存
__constant__ double c_lpFilter[MARGIN];
__constant__ double c_hpFilter[MARGIN];
__constant__ double c_iLpFilter[MARGIN];
__constant__ double c_iHpFilter[MARGIN];

//-------------------------------------------------------------------正变换部分------------------------------------------------------------------------------------------------

//对二维数据的行进行小波变换一维运算,两个输出结果lp_output,hp_output是已经转置过后的大小,在一个核函数中进行扩展、卷积、下采样和转置
__global__ void Dwt1D_Row(double *data,int width,int height,size_t pitch_data,int filter_len,
						  double *lp_output,size_t pitch_lpout,double *hp_output,size_t pitch_hpout)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
	int row=threadIdx.y+blockIdx.y*blockDim.y;

	int aftWidth=(width+filter_len-1)/2;//卷积+降采样后的长度
	if(col>=aftWidth||row>=height)
		return;

	double* row_data=(double*)((char*)data+row*pitch_data);//输入数据的起始行地址
	double* row_lpout=(double*)((char*)lp_output+col*pitch_lpout);//低通滤波结果的起始行地址
	double* row_hpout=(double*)((char*)hp_output+col*pitch_hpout);//低通滤波结果的起始行地址

	int symIndex=filter_len+2*col;//只计算分解的一半结果，即只需卷积结果的奇数部分，tmpIndex是得到卷积结果的那个数在对称扩展后中的位置
	int oriIndex=symIndex-filter_len+1;//symIndex-(filter_len-1),用于计算需要取的数据在原始数据中的位置,2*col+1

	double lp_result=0;//低通滤波器的卷积结果
	double hp_result=0;//高通滤波器的卷积结果

#pragma unroll
	for(int i=0;i<filter_len;i++)
	{
		int index=oriIndex-i;
		double tmpValue;//存储卷积计算中所需数据的值
		if(index<0)
		{
			tmpValue=row_data[-index-1];//-1取0，-2取1，-3取2，-4取3，-5取4,|index|-1
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
	
	//结果直接进行转置
	row_lpout[row]=lp_result;//输出结果的宽度是卷积结果的一半，高度不变
	row_hpout[row]=hp_result;
}

//对二维数据的列进行小波变换一维运算，由于列方向上的运算对于上一步的两个结果的运算过程是一样的，最终大小也是一样的，所以可以把这两个运算过程合并到一个核函数中；
//输入是上一步行运算的两个转置后的结果，输出一个是用于存放下一轮计算的CLL，一个是最终的存放位置。
//width是已经转置后矩阵的宽度，height是已经转置后矩阵的高度，核函数的参数配置中宽度和高度分别以width和height为依据,
__global__ void Dwt1D_Col(double *lpOutput,size_t pitch_lpout,double *hpOutput,size_t pitch_hpout,int width,int height,
						  int filter_len,double *CLL,size_t pitch_cll,double *output,int offset)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
	int row=threadIdx.y+blockIdx.y*blockDim.y;

	int aftWidth=(width+filter_len-1)/2;//卷积+降采样后的长度
	if(col>=aftWidth||row>=height)
		return;

	double* row_lpData=(double*)((char*)lpOutput+row*pitch_lpout);//输入数据的起始行地址
	double* row_hpData=(double*)((char*)hpOutput+row*pitch_hpout);//输入数据的起始行地址
	double* row_cll=(double*)((char*)CLL+col*pitch_cll);//CLL结果的起始行地址

	int symIndex=filter_len+2*col;//只计算分解的一半结果，即只需卷积结果的奇数部分，tmpIndex是得到卷积结果的那个数在对称扩展后中的位置
	int oriIndex=symIndex-filter_len+1;//symIndex-(filter_len-1),用于计算需要取的数据在原始数据中的位置,2*col+1

	double cll_result=0;//CLL的结果
	double clh_result=0;//CLH的结果
	double chl_result=0;//CLL的结果
	double chh_result=0;//CLH的结果

#pragma unroll
	for(int i=0;i<filter_len;i++)
	{
		int index=oriIndex-i;
		double lpTmpValue;//存储卷积计算中所需lpOoutput数据的值
		double hpTmpValue;//存储卷积计算中所需hpOoutput数据的值
		if(index<0)
		{
			lpTmpValue=row_lpData[-index-1];//-1取0，-2取1，-3取2，-4取3，-5取4,|index|-1
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

	//结果直接进行转置
	//offset=totalsize-sum_step，参考上一版程序Wavelet中的两个参数
	row_cll[row]=cll_result;
	output[offset-3*aftWidth*height+col*height+row]=clh_result;
	output[offset-2*aftWidth*height+col*height+row]=chl_result;
	output[offset-aftWidth*height+col*height+row]=chh_result;
}

//单次正向小波变换
void Dwt2D(double *data,int data_width,int data_height,size_t pitch_data,int filter_len,double *CLL,size_t pitch_cll,double *output,int offset)
{
	//----------------------第一阶段：先对整个二维数据进行行方向上的变换----------------------------------------------------
	//由于第一步的两个输出结果是转置后两个矩阵，故在申请内存时直接按照转置后的大小申请
	int height=(data_width+filter_len-1)/2;//转置后的高度，即原宽度
	int width=data_height;//转置后的宽度，即原高度

	//存放行运算后的两个输出
	double *lpOutput;
	double *hpOutput;
	size_t pitch_lpout;
	size_t pitch_hpout;
	cudaMallocPitch((void **)&lpOutput,&pitch_lpout,sizeof(double)*width,height);
	cudaMallocPitch((void **)&hpOutput,&pitch_hpout,sizeof(double)*width,height);

	dim3 threads(TILE_X,TILE_Y);
	dim3 blocks_row((height+TILE_X-1)/TILE_X,(width+TILE_Y-1)/TILE_Y);//以原宽度和原高度来确定grid大小
	Dwt1D_Row<<<blocks_row,threads>>>(data,data_width,data_height,pitch_data,filter_len,lpOutput,pitch_lpout,hpOutput,pitch_hpout);

	//----------------------第二阶段：对之前得到的两个已经转置后的矩阵进行列方向上的变换----------------------------------------
	int aftwidth=(width+filter_len-1)/2;
	dim3 blocks_col((aftwidth+TILE_X-1)/TILE_X,(height+TILE_Y-1)/TILE_Y);
	Dwt1D_Col<<<blocks_col,threads>>>(lpOutput,pitch_lpout,hpOutput,pitch_hpout,width,height,filter_len,CLL,pitch_cll,output,offset);

	//释放
	cudaFree(lpOutput);
	cudaFree(hpOutput);
}


//二维正向小波变换
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

	//----------------------把二维数据复制到GPU中------------------------------------------------------
	double *d_data;
	size_t pitch_data;
	cudaMallocPitch((void **)&d_data,&pitch_data,sizeof(double)*width,height);

	cudaMemcpy2D(d_data,pitch_data,originData,sizeof(double)*width,sizeof(double)*width,height,cudaMemcpyHostToDevice);	//这里之前把sizeof(double)*width错写成*height，造成数据传输错误
	cudaMemcpyToSymbol(c_lpFilter, lpfilter, filter_len*sizeof(double));
	cudaMemcpyToSymbol(c_hpFilter, hpfilter, filter_len*sizeof(double));
	//----------------------数据复制阶段完成-----------------------------------------------------------

	//----------------------进行J层小波分解-----------------------------------------------------------
	for(int iter=0;iter<level;iter++)
	{
		int next_width=(width+filter_len-1)/2;
		int next_height=(height+filter_len-1)/2;

		//存储每一层变换后的长度
		length[iter*2]=next_height;
		length[iter*2+1]=next_width;
		int offset=total_size-sum_step;//CLH,CHL,CHH在output中的最后位置

		double *cLL;//存储用于下一轮小波变换的初始数据
		size_t pitch_cll;
		cudaMallocPitch((void **)&cLL,&pitch_cll,sizeof(double)*next_width,next_height);

		//执行二维变换
		Dwt2D(d_data,width,height,pitch_data,filter_len,cLL,pitch_cll,d_output,offset);

		width=next_width;
		height=next_height;

		//下一轮循环的初始数据
		cudaFree(d_data);
		d_data=cLL;
		pitch_data=pitch_cll;

		//存储结果
		if(iter==level-1)
		{
			cudaMemcpy2D((d_output+offset-4*next_width*next_height),sizeof(double)*next_width,cLL,pitch_cll,sizeof(double)*next_width,next_height,cudaMemcpyDeviceToDevice);
			cudaFree(cLL);
		}
		sum_step+=next_width*next_height*3;
	}
	//----------------------J层小波分解完成-----------------------------------------------------------
	cudaMemcpy(output,d_output,sizeof(double)*total_size,cudaMemcpyDeviceToHost);
	cudaFree(d_data);
	cudaFree(d_output);
}




//-------------------------------------------------------------------逆变换部分------------------------------------------------------------------------------------------------



//小波逆变换中列方向上的运算
//从之前的分解结果中单独提取出CLL，再从全部分解结果中通过offset来提取CLH、CHL、CHH三个部分的数据，offset是一层分解结果的总大小，
//在取数据时直接一列一列的取，相当于转置后的行运算，运算结果亦相当于转置后放入输出中，这样完成在列方向上的计算，方便后续行方向
//上的计算不需要再进行转置，在配置核函数的参数时，blocks的宽度是以nextHeight，高度以原width为依据，即以转置后的结果矩阵大小，形
//参中的width,height是转置后的，相当于原height,原width
//CLL不能使用Pitch的形式，因为要按列取，不适合Pitch的形式，也因为这个原因，iDwt1D_row的output也不能用Pitch的形式
__global__ void iDwt1D_Col(double *CLL,double *data,int offset,int width,int height,int nextWidth,int filter_len,
						   double *app,size_t pitch_app,double *detail,size_t pitch_detail)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
 	int row=threadIdx.y+blockIdx.y*blockDim.y;

	if(col>=nextWidth||row>=height)
		return;

	//剩余三个部分数据的起始位置
	int clh_begin=offset;
	int chl_begin=offset+width*height;
	int chh_begin=offset+2*width*height;

	//输出数据
	double* row_appData=(double*)((char*)app+col*pitch_app);//起始行地址，col相当于转置后的行号
	double* row_detailData=(double*)((char*)detail+col*pitch_detail);//起始行地址，col相当于转置后的行号

	int oriIndex=col+filter_len-2;//参与卷积计算的起始数据在上采样后的位置
	
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

		//比小波分解运算中取数据部分要简单，若位置不在上采样后的数据中，则取0，若在，则看其索引是否是偶数，是偶数则在原始数据中取（索引/2）位置的数据
		//若是奇数，则直接取0
		if(index>=0&&index<(2*width-1)&&(index%2)==0)
		{
			cllTmpValue=CLL[index/2*height+row];//一列一列的取数据
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

	//相当于再转置后放入输出中
	row_appData[row]=cll_result+clh_result;
	row_detailData[row]=chl_result+chh_result;
}


//小波逆变换中行方向上的运算
//由列方向上的运算得到的两个结果，直接再进行行方向上的卷积运算，即可还原最后数据，核函数的参数，blocks的宽度和高度分别以nextWidth,
//nextHeight为依据，方法基本和上面的一样，除了不用转置
__global__ void iDwt1D_Row(double *app,size_t pitch_app,double *detail,size_t pitch_detail,int width,int height,
						   int nextWidth,int filter_len,double *output)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
 	int row=threadIdx.y+blockIdx.y*blockDim.y;

	if(col>=nextWidth||row>=height)
		return;

	//输入数据和输出数据行的起始位置
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


//二维逆向小波变换
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

	//----------------------------------数据复制阶段---------------------------------------------
	double *d_data;
	cudaMalloc((void **)&d_data,sizeof(double)*data_size);

	cudaMemcpy(d_data,data,sizeof(double)*data_size,cudaMemcpyHostToDevice);
	cudaMemcpy(CLL,d_data,sizeof(double)*width*height,cudaMemcpyDeviceToDevice);//初始CLL数据填充
	cudaMemcpyToSymbol(c_iLpFilter, lpfilter, filter_len*sizeof(double));
	cudaMemcpyToSymbol(c_iHpFilter, hpfilter, filter_len*sizeof(double));
	//----------------------------------数据复制阶段完成------------------------------------------

	
	//----------------------------------J层小波重构------------------------------------------------
	dim3 threads(TILE_X,TILE_Y);
	for(int iter=0;iter<level;iter++)
	{
		//当前层的高宽
		int tmp_height=length[2*iter];
		int tmp_width=length[2*iter + 1];
		//下一层的高宽
		int next_height=length[2*iter+2];
		int next_width=length[2*iter+3];

		//----------------------第一阶段：先进行列方向上的逆变换---------------------------------
		double *app;
		double *detail;
		size_t pitch_app;
		size_t pitch_detail;
		cudaMallocPitch((void **)&app,&pitch_app,sizeof(double)*tmp_width,next_height);
		cudaMallocPitch((void **)&detail,&pitch_detail,sizeof(double)*tmp_width,next_height);
		
		dim3 blocks_col((next_height+TILE_X-1)/TILE_X,(tmp_width+TILE_Y-1)/TILE_Y);
		iDwt1D_Col<<<blocks_col,threads>>>(CLL,d_data,offset,tmp_height,tmp_width,next_height,filter_len,app,pitch_app,detail,pitch_detail);

		//----------------------第二阶段：进行行方向上的逆变换-------------------------------------
		cudaFree(CLL);
		cudaMalloc((void **)&CLL,sizeof(double)*next_height*next_width);//清空之前的CLL数据，把输出放入CLL中作为下一次循环的初始数据
		dim3 blocks_row((next_width+TILE_X-1)/TILE_X,(next_height+TILE_Y-1)/TILE_Y);
		iDwt1D_Row<<<blocks_row,threads>>>(app,pitch_app,detail,pitch_detail,tmp_width,next_height,next_width,filter_len,CLL);
		offset+=3*tmp_height*tmp_width;	

		cudaFree(app);
		cudaFree(detail);
	}
	//-----------------------------J层小波重构结束--------------------------------------------------

	//复制回数据
	height=length[2*level];
	width=length[2*level+1];
	cudaMemcpy(output,CLL,sizeof(double)*height*width,cudaMemcpyDeviceToHost);
	cudaFree(CLL);
	cudaFree(d_data);
}



/*
//用于获取转置后的数据
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
	//输出起始行地址
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

//小波逆变换中列方向上的运算
__global__ void iDwt1D_Col(double *CLL,size_t pitch_cll,double *CLH,size_t pitch_clh,double *CHL,size_t pitch_chl,double *CHH,size_t pitch_chh,
						   int width,int height,int nextWidth,int filter_len,
						   double *app,size_t pitch_app,double *detail,size_t pitch_detail)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
 	int row=threadIdx.y+blockIdx.y*blockDim.y;

	if(col>=nextWidth||row>=height)
		return;

	//输入数据的起始行地址
	double* row_CLLData=(double*)((char*)CLL+row*pitch_cll);
	double* row_CLHData=(double*)((char*)CLH+row*pitch_clh);
	double* row_CHLData=(double*)((char*)CHL+row*pitch_chl);
	double* row_CHHData=(double*)((char*)CHH+row*pitch_chh);

	//输出数据的起始行地址，col相当于转置后的行号
	double* row_appData=(double*)((char*)app+col*pitch_app);
	double* row_detailData=(double*)((char*)detail+col*pitch_detail);

	int oriIndex=col+filter_len-2;//参与卷积计算的起始数据在上采样后的位置
	
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

		//比小波分解运算中取数据部分要简单，若位置不在上采样后的数据中，则取0，若在，则看其索引是否是偶数，是偶数则在原始数据中取（索引/2）位置的数据
		//若是奇数，则直接取0
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

	//相当于再转置后放入输出中
	row_appData[row]=cll_result+clh_result;
	row_detailData[row]=chl_result+chh_result;
}

//小波逆变换中行方向上的运算
__global__ void iDwt1D_Row(double *app,size_t pitch_app,double *detail,size_t pitch_detail,int width,int height,
						   int nextWidth,int filter_len,double *output,size_t pitch_out)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
 	int row=threadIdx.y+blockIdx.y*blockDim.y;

	if(col>=nextWidth||row>=height)
		return;

	//输入数据和输出数据行的起始位置
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

//二维逆向小波变换
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

	//----------------------------------数据复制阶段---------------------------------------------
	double *d_data;
	cudaMalloc((void **)&d_data,sizeof(double)*data_size);

	cudaMemcpy(d_data,data,sizeof(double)*data_size,cudaMemcpyHostToDevice);
	cudaMemcpy2D(CLL,pitch_CLL,d_data,sizeof(double)*width,sizeof(double)*width,height,cudaMemcpyDeviceToDevice);//初始CLL数据填充
	cudaMemcpyToSymbol(c_iLpFilter, lpfilter, filter_len*sizeof(double));
	cudaMemcpyToSymbol(c_iHpFilter, hpfilter, filter_len*sizeof(double));
	//----------------------------------数据复制阶段完成------------------------------------------

	
	//----------------------------------J层小波重构------------------------------------------------
	dim3 threads(TILE_X,TILE_Y);
	for(int iter=0;iter<level;iter++)
	{
		//当前层的高宽
		int tmp_height=length[2*iter];
		int tmp_width=length[2*iter + 1];
		//下一层的高宽
		int next_height=length[2*iter+2];
		int next_width=length[2*iter+3];

		//----------------------获取转置数据-----------------------------------------------------
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
		//----------------------第一阶段：先进行列方向上的逆变换---------------------------------
		double *app;
		double *detail;
		size_t pitch_app;
		size_t pitch_detail;
		cudaMallocPitch((void **)&app,&pitch_app,sizeof(double)*tmp_width,next_height);
		cudaMallocPitch((void **)&detail,&pitch_detail,sizeof(double)*tmp_width,next_height);
		
		dim3 blocks_col((next_height+TILE_X-1)/TILE_X,(tmp_width+TILE_Y-1)/TILE_Y);
		iDwt1D_Col<<<blocks_col,threads>>>(cll,pitch_cll,clh,pitch_clh,chl,pitch_chl,chh,pitch_chh,
										   tmp_height,tmp_width,next_height,filter_len,app,pitch_app,detail,pitch_detail);
		//用完就释放
		cudaFree(cll);
		cudaFree(clh);
		cudaFree(chl);
		cudaFree(chh);

		//----------------------第二阶段：进行行方向上的逆变换-------------------------------------
		double *out;size_t pitch_out;
		cudaMallocPitch((void **)&out,&pitch_out,sizeof(double)*next_width,next_height);
		dim3 blocks_row((next_width+TILE_X-1)/TILE_X,(next_height+TILE_Y-1)/TILE_Y);
		iDwt1D_Row<<<blocks_row,threads>>>(app,pitch_app,detail,pitch_detail,tmp_width,next_height,next_width,filter_len,out,pitch_out);
		offset+=3*tmp_height*tmp_width;	

		//用完就释放
		cudaFree(app);
		cudaFree(detail);

		//-----------------------设置下一层的CLL初始数据----------------------------------------------
		cudaFree(CLL);
		CLL=out;
		pitch_CLL=pitch_out;

		if(iter==level-1)
		{
			cudaMemcpy2D(output,sizeof(double)*next_width,CLL,pitch_CLL,sizeof(double)*next_width,next_height,cudaMemcpyDeviceToHost);
			cudaFree(out);
		}
	}
	//-----------------------------J层小波重构结束--------------------------------------------------

	cudaFree(CLL);
	cudaFree(d_data);
}

*/



//-------------------------------------------------------------------高斯滤波部分------------------------------------------------------------------------------------------------


__global__ void damp(int height,int width,int filter_len,cufftDoubleComplex *d_inp_fft,double *d_filter)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(col>=width||row>=height||col>=filter_len)
		return;

	d_inp_fft[row*width+col].x*= d_filter[col];
	d_inp_fft[row*width+col].y*= d_filter[col];
}

//高斯滤波
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
		//滤波器
		int filter_len=cols/2+1;
		double *filter=(double*)malloc(sizeof(double)*filter_len);

		for(int k=0;k<filter_len;k++)
				filter[k]=1-exp(-(double)(k)*(k)/(2*sigma*sigma));	

		double *d_filter;
		cudaMalloc((void**)&d_filter, sizeof(double)*filter_len);
		cudaMemcpy(d_filter, filter, sizeof(double)*filter_len, cudaMemcpyHostToDevice);
		free(filter);

		//输入数据部分
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

