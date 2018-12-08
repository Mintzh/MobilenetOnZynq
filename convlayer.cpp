#include "stdafx.h"
#include "convlayer.h"
#include <stdio.h>
#include<assert.h>
//implement the convolution

clock_t startclk, finishclk;
double  duration;
double durationprod;
double  prodtime;
const char* Conv2weightdir = "./wbin/Conv2d_0_weights.bin";
const char* Conv2biasdir = "./wbin/Conv2d_0_biases.bin";

const char* dep1weightdir = "./wbin/Conv2d_1_depthwise_depthwise_weights.bin";
const char* dep1biasdir = "./wbin/Conv2d_1_depthwise_biases.bin";
const char* poi1weightdir = "./wbin/Conv2d_1_pointwise_weights.bin";
const char* poi1biasdir = "./wbin/Conv2d_1_pointwise_biases.bin";

const char* dep2weightdir = "./wbin/Conv2d_2_depthwise_depthwise_weights.bin";
const char* dep2biasdir = "./wbin/Conv2d_2_depthwise_biases.bin";
const char* poi2weightdir = "./wbin/Conv2d_2_pointwise_weights.bin";
const char* poi2biasdir = "./wbin/Conv2d_2_pointwise_biases.bin";

const char* dep3weightdir = "./wbin/Conv2d_3_depthwise_depthwise_weights.bin";
const char* dep3biasdir = "./wbin/Conv2d_3_depthwise_biases.bin";
const char* poi3weightdir = "./wbin/Conv2d_3_pointwise_weights.bin";
const char* poi3biasdir = "./wbin/Conv2d_3_pointwise_biases.bin";

const char* dep4weightdir = "./wbin/Conv2d_4_depthwise_depthwise_weights.bin";
const char* dep4biasdir = "./wbin/Conv2d_4_depthwise_biases.bin";
const char* poi4weightdir = "./wbin/Conv2d_4_pointwise_weights.bin";
const char* poi4biasdir = "./wbin/Conv2d_4_pointwise_biases.bin";

const char* dep5weightdir = "./wbin/Conv2d_5_depthwise_depthwise_weights.bin";
const char* dep5biasdir = "./wbin/Conv2d_5_depthwise_biases.bin";
const char* poi5weightdir = "./wbin/Conv2d_5_pointwise_weights.bin";
const char* poi5biasdir = "./wbin/Conv2d_5_pointwise_biases.bin";

const char* dep6weightdir = "./wbin/Conv2d_6_depthwise_depthwise_weights.bin";
const char* dep6biasdir = "./wbin/Conv2d_6_depthwise_biases.bin";
const char* poi6weightdir = "./wbin/Conv2d_6_pointwise_weights.bin";
const char* poi6biasdir = "./wbin/Conv2d_6_pointwise_biases.bin";

const char* dep7weightdir = "./wbin/Conv2d_7_depthwise_depthwise_weights.bin";
const char* dep7biasdir = "./wbin/Conv2d_7_depthwise_biases.bin";
const char* poi7weightdir = "./wbin/Conv2d_7_pointwise_weights.bin";
const char* poi7biasdir = "./wbin/Conv2d_7_pointwise_biases.bin";

const char* dep8weightdir = "./wbin/Conv2d_8_depthwise_depthwise_weights.bin";
const char* dep8biasdir = "./wbin/Conv2d_8_depthwise_biases.bin";
const char* poi8weightdir = "./wbin/Conv2d_8_pointwise_weights.bin";
const char* poi8biasdir = "./wbin/Conv2d_8_pointwise_biases.bin";

const char* dep9weightdir = "./wbin/Conv2d_9_depthwise_depthwise_weights.bin";
const char* dep9biasdir = "./wbin/Conv2d_9_depthwise_biases.bin";
const char* poi9weightdir = "./wbin/Conv2d_9_pointwise_weights.bin";
const char* poi9biasdir = "./wbin/Conv2d_9_pointwise_biases.bin";

const char* dep10weightdir = "./wbin/Conv2d_10_depthwise_depthwise_weights.bin";
const char* dep10biasdir = "./wbin/Conv2d_10_depthwise_biases.bin";
const char* poi10weightdir = "./wbin/Conv2d_10_pointwise_weights.bin";
const char* poi10biasdir = "./wbin/Conv2d_10_pointwise_biases.bin";

const char* dep11weightdir = "./wbin/Conv2d_11_depthwise_depthwise_weights.bin";
const char* dep11biasdir = "./wbin/Conv2d_11_depthwise_biases.bin";
const char* poi11weightdir = "./wbin/Conv2d_11_pointwise_weights.bin";
const char* poi11biasdir = "./wbin/Conv2d_11_pointwise_biases.bin";

const char* dep12weightdir = "./wbin/Conv2d_12_depthwise_depthwise_weights.bin";
const char* dep12biasdir = "./wbin/Conv2d_12_depthwise_biases.bin";
const char* poi12weightdir = "./wbin/Conv2d_12_pointwise_weights.bin";
const char* poi12biasdir = "./wbin/Conv2d_12_pointwise_biases.bin";

const char* dep13weightdir = "./wbin/Conv2d_13_depthwise_depthwise_weights.bin";
const char* dep13biasdir = "./wbin/Conv2d_13_depthwise_biases.bin";
const char* poi13weightdir = "./wbin/Conv2d_13_pointwise_weights.bin";
const char* poi13biasdir = "./wbin/Conv2d_13_pointwise_biases.bin";

void debugpr(paratype* datasee, int num)
{
	for (int inindex = 0; inindex < num; inindex++)
	{
		printf("%f,", datasee[inindex]);
	}
	printf("\n");
}
void debugpr_offset(paratype* datasee, int num, int offset)
{
	for (int inindex = 0; inindex < num; inindex++)
	{
		printf("%f,", datasee[inindex*offset]);
	}
	printf("\n");
}

inline void relu6(paratype* dataforrelu)
{
	if (*dataforrelu<0)
	{
		*dataforrelu = 0;
	}
	else if (*dataforrelu>6)
	{
		*dataforrelu = 6;
	}
}
inline paratype relu6_num(paratype dataforrelu)
{
	if (dataforrelu<0)
	{
		return 0;
	}
	else if (dataforrelu>6)
	{
		return 6;
	}
	return dataforrelu;
}
inline void relu6mul(paratype* dataforrelu, int num)
{
	for (int i = 0; i<num; i++)
	{
		if (dataforrelu[i]<0)
		{
			dataforrelu[i] = 0;
		}
		else if (dataforrelu[i]>6)
		{
			dataforrelu[i] = 6;
		}
	}
}
//this function calculate prod
inline void tensorprod_norelu(paratype* graph, paratype* convweight, paratype* result, int prodnum, int woffset, paratype bias)
{
	paratype product = 0;
	for (int inindex = 0; inindex<prodnum; inindex++)
	{
		product += graph[inindex] * convweight[inindex*woffset];
	}
	*result = product + bias;
	//printf("from%f\n", product);
}

inline void tensorprod(paratype* graph, paratype* convweight, paratype* result, int prodnum, int woffset,paratype bias)
{
	startclk = clock();
		
	
	paratype product = 0;
	for (int inindex = 0; inindex<prodnum; inindex++)
	{
		product = product + graph[inindex] * convweight[inindex*woffset];
	}
	*result = product + bias;
	relu6(result);
	finishclk = clock();
	durationprod+= (double)(finishclk - startclk) / CLOCKS_PER_SEC;
	//printf("from%f\n", product);
}

inline void tensorprod_offsetw(paratype* graph, paratype* convweight, paratype* result, int prodnum,int offset, paratype bias)
{
	paratype product = 0;
	for (int inindex = 0; inindex<prodnum; inindex++)
	{
		product += graph[inindex] * convweight[inindex*offset];
	}
	*result = product + bias;
	relu6(result);
	//printf("from%f\n", product);
}


inline void tensorprod_depwise(paratype* graph, paratype* convweight, paratype* result, int channels, int kernelsize, paratype* bias)
{
	for (int cind = 0; cind < channels; cind++)
	{
		result[cind] = 0;
	}
	for (int inindex = 0; inindex < kernelsize*kernelsize; inindex++)
	{
		for (int cind = 0; cind < channels; cind++)
		{
			result[cind] += graph[inindex*channels + cind] * convweight[inindex*channels + cind];
		}
	}
	for (int cind = 0; cind < channels; cind++)
	{
		result[cind] += bias[cind];
		relu6(result+cind);
	}
	//printf("from%f\n", product);
}

//this function add bias to a block
inline void bias(paratype* data, paratype* bias, int datanum)
{
	for (int inindex = 0; inindex<datanum; inindex++)
	{
		data[inindex] += *bias;
	}
}

//copy value to array
inline void copyval(paratype* indata, paratype* mem, int datanum)
{
	for (int inindex = 0; inindex<datanum; inindex++)
	{
		indata[inindex] = mem[inindex];
	}
}
inline void copyval_offset(paratype* indata, paratype* mem, int datanum, int offset)
{
	for (int inindex = 0; inindex<datanum; inindex++)
	{
		indata[inindex] = mem[inindex*offset];
	}
}
//initialize 
inline void setzero(paratype* datablock, int num)
{
	for (int inindex = 0; inindex < num; inindex++) { datablock[inindex] = 0; }
}

static int firstread = 0;
void convmodule(IN paratype* graphin, OUT paratype* grahout)
{
	prodtime = 0;
	/*read parameters*/
	static paratype Conv2W[Conv2Kernel*Conv2Kernel*graphinC*Conv2Channel];
	static paratype Conv2B[Conv2Channel];

	static paratype Conv1_depW[Conv1_depKernel*Conv1_depKernel*Conv1_depChannel];
	static paratype Conv1_depB[Conv1_depChannel];
	static paratype Conv1_poiW[Conv1_poiKernel*Conv1_poiKernel*Conv1_depChannel*Conv1_poiChannel];
	static paratype Conv1_poiB[Conv1_poiChannel];

	static paratype Conv2_depW[Conv2_depKernel*Conv2_depKernel*Conv2_depChannel];
	static paratype Conv2_depB[Conv2_depChannel];
	static paratype Conv2_poiW[Conv2_poiKernel*Conv2_poiKernel*Conv2_depChannel*Conv2_poiChannel];
	static paratype Conv2_poiB[Conv2_poiChannel];

	static paratype Conv3_depW[Conv3_depKernel*Conv3_depKernel*Conv3_depChannel];
	static paratype Conv3_depB[Conv3_depChannel];
	static paratype Conv3_poiW[Conv3_poiKernel*Conv3_poiKernel*Conv3_depChannel*Conv3_poiChannel];
	static paratype Conv3_poiB[Conv3_poiChannel];

	static paratype Conv4_depW[Conv4_depKernel*Conv4_depKernel*Conv4_depChannel];
	static paratype Conv4_depB[Conv4_depChannel];
	static paratype Conv4_poiW[Conv4_poiKernel*Conv4_poiKernel*Conv4_depChannel*Conv4_poiChannel];
	static paratype Conv4_poiB[Conv4_poiChannel];

	static paratype Conv5_depW[Conv5_depKernel*Conv5_depKernel*Conv5_depChannel];
	static paratype Conv5_depB[Conv5_depChannel];
	static paratype Conv5_poiW[Conv5_poiKernel*Conv5_poiKernel*Conv5_depChannel*Conv5_poiChannel];
	static paratype Conv5_poiB[Conv5_poiChannel];

	static paratype Conv6_depW[Conv6_depKernel*Conv6_depKernel*Conv6_depChannel];
	static paratype Conv6_depB[Conv6_depChannel];
	static paratype Conv6_poiW[Conv6_poiKernel*Conv6_poiKernel*Conv6_depChannel*Conv6_poiChannel];
	static paratype Conv6_poiB[Conv6_poiChannel];

	static paratype Conv7_depW[Conv7_depKernel*Conv7_depKernel*Conv7_depChannel];
	static paratype Conv7_depB[Conv7_depChannel];
	static paratype Conv7_poiW[Conv7_poiKernel*Conv7_poiKernel*Conv7_depChannel*Conv7_poiChannel];
	static paratype Conv7_poiB[Conv7_poiChannel];

	static paratype Conv8_depW[Conv8_depKernel*Conv8_depKernel*Conv8_depChannel];
	static paratype Conv8_depB[Conv8_depChannel];
	static paratype Conv8_poiW[Conv8_poiKernel*Conv8_poiKernel*Conv8_depChannel*Conv8_poiChannel];
	static paratype Conv8_poiB[Conv8_poiChannel];

	static paratype Conv9_depW[Conv9_depKernel*Conv9_depKernel*Conv9_depChannel];
	static paratype Conv9_depB[Conv9_depChannel];
	static paratype Conv9_poiW[Conv9_poiKernel*Conv9_poiKernel*Conv9_depChannel*Conv9_poiChannel];
	static paratype Conv9_poiB[Conv9_poiChannel];

	static paratype Conv10_depW[Conv10_depKernel*Conv10_depKernel*Conv10_depChannel];
	static paratype Conv10_depB[Conv10_depChannel];
	static paratype Conv10_poiW[Conv10_poiKernel*Conv10_poiKernel*Conv10_depChannel*Conv10_poiChannel];
	static paratype Conv10_poiB[Conv10_poiChannel];

	static paratype Conv11_depW[Conv11_depKernel*Conv11_depKernel*Conv11_depChannel];
	static paratype Conv11_depB[Conv11_depChannel];
	static paratype Conv11_poiW[Conv11_poiKernel*Conv11_poiKernel*Conv11_depChannel*Conv11_poiChannel];
	static paratype Conv11_poiB[Conv11_poiChannel];

	static paratype Conv12_depW[Conv12_depKernel*Conv12_depKernel*Conv12_depChannel];
	static paratype Conv12_depB[Conv12_depChannel];
	static paratype Conv12_poiW[Conv12_poiKernel*Conv12_poiKernel*Conv12_depChannel*Conv12_poiChannel];
	static paratype Conv12_poiB[Conv12_poiChannel];

	static paratype Conv13_depW[Conv13_depKernel*Conv13_depKernel*Conv13_depChannel];
	static paratype Conv13_depB[Conv13_depChannel];
	static paratype Conv13_poiW[Conv13_poiKernel*Conv13_poiKernel*Conv13_depChannel*Conv13_poiChannel];
	static paratype Conv13_poiB[Conv13_poiChannel];

	paratype* conv1_result = new paratype[257 * 257 * 24];
	paratype* dep1_result = new paratype[257 * 257 * 24];
	paratype* poi1_result = new paratype[257 * 257 * 48];

	paratype* dep2_result = new paratype[Conv2_depoutH * Conv2_depoutH * Conv2_depChannel];
	paratype* poi2_result = new paratype[Conv2_poioutH * Conv2_poioutH * Conv2_poiChannel];

	paratype* dep3_result = new paratype[Conv3_depoutH * Conv3_depoutH * Conv3_depChannel];
	paratype* poi3_result = new paratype[Conv3_poioutH * Conv3_poioutH * Conv3_poiChannel];

	paratype* dep4_result = new paratype[Conv4_depoutH * Conv4_depoutH * Conv4_depChannel];
	paratype* poi4_result = new paratype[Conv4_poioutH * Conv4_poioutH * Conv4_poiChannel];

	paratype* dep5_result = new paratype[Conv5_depoutH * Conv5_depoutH * Conv5_depChannel];
	paratype* poi5_result = new paratype[Conv5_poioutH * Conv5_poioutH * Conv5_poiChannel];

	paratype* dep6_result = new paratype[Conv6_depoutH * Conv6_depoutH * Conv6_depChannel];

	paratype* dep7_result = new paratype[Conv7_depoutH * Conv7_depoutH * Conv7_depChannel];
	paratype* poi7_result = new paratype[Conv7_poioutH * Conv7_poioutH * Conv7_poiChannel];
	//read in the params read once
	if (!firstread)
	{
		firstread = 1;
		FILE *fr = fopen(Conv2weightdir, "rb");
		fread(Conv2W, sizeof(paratype), Conv2Kernel*Conv2Kernel*graphinC*Conv2Channel, fr);
		fclose(fr);
		fr = fopen(Conv2biasdir, "rb");
		fread(Conv2B, sizeof(paratype), Conv2Channel, fr);
		fclose(fr);

		fr = fopen(dep1weightdir, "rb");
		fread(Conv1_depW, sizeof(paratype), Conv1_depKernel*Conv1_depKernel*Conv1_depChannel, fr);
		fclose(fr);

		fr = fopen(dep1biasdir, "rb");
		fread(Conv1_depB, sizeof(paratype), Conv1_depChannel, fr);
		fclose(fr);

		fr = fopen(poi1weightdir, "rb");
		fread(Conv1_poiW, sizeof(paratype), Conv1_poiKernel*Conv1_poiKernel*Conv1_depChannel*Conv1_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi1biasdir, "rb");
		fread(Conv1_poiB, sizeof(paratype), Conv1_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep2weightdir, "rb");
		fread(Conv2_depW, sizeof(paratype), Conv2_depKernel*Conv2_depKernel*Conv2_depChannel, fr);
		fclose(fr);

		fr = fopen(dep2biasdir, "rb");
		fread(Conv2_depB, sizeof(paratype), Conv2_depChannel, fr);
		fclose(fr);

		fr = fopen(poi2weightdir, "rb");
		fread(Conv2_poiW, sizeof(paratype), Conv2_poiKernel*Conv2_poiKernel*Conv2_depChannel*Conv2_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi2biasdir, "rb");
		fread(Conv2_poiB, sizeof(paratype), Conv2_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep3weightdir, "rb");
		fread(Conv3_depW, sizeof(paratype), Conv3_depKernel*Conv3_depKernel*Conv3_depChannel, fr);
		fclose(fr);

		fr = fopen(dep3biasdir, "rb");
		fread(Conv3_depB, sizeof(paratype), Conv3_depChannel, fr);
		fclose(fr);

		fr = fopen(poi3weightdir, "rb");
		fread(Conv3_poiW, sizeof(paratype), Conv3_poiKernel*Conv3_poiKernel*Conv3_depChannel*Conv3_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi3biasdir, "rb");
		fread(Conv3_poiB, sizeof(paratype), Conv3_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep4weightdir, "rb");
		fread(Conv4_depW, sizeof(paratype), Conv4_depKernel*Conv4_depKernel*Conv4_depChannel, fr);
		fclose(fr);

		fr = fopen(dep4biasdir, "rb");
		fread(Conv4_depB, sizeof(paratype), Conv4_depChannel, fr);
		fclose(fr);

		fr = fopen(poi4weightdir, "rb");
		fread(Conv4_poiW, sizeof(paratype), Conv4_poiKernel*Conv4_poiKernel*Conv4_depChannel*Conv4_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi4biasdir, "rb");
		fread(Conv4_poiB, sizeof(paratype), Conv4_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep5weightdir, "rb");
		fread(Conv5_depW, sizeof(paratype), Conv5_depKernel*Conv5_depKernel*Conv5_depChannel, fr);
		fclose(fr);

		fr = fopen(dep5biasdir, "rb");
		fread(Conv5_depB, sizeof(paratype), Conv5_depChannel, fr);
		fclose(fr);

		fr = fopen(poi5weightdir, "rb");
		fread(Conv5_poiW, sizeof(paratype), Conv5_poiKernel*Conv5_poiKernel*Conv5_depChannel*Conv5_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi5biasdir, "rb");
		fread(Conv5_poiB, sizeof(paratype), Conv5_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep6weightdir, "rb");
		fread(Conv6_depW, sizeof(paratype), Conv6_depKernel*Conv6_depKernel*Conv6_depChannel, fr);
		fclose(fr);

		fr = fopen(dep6biasdir, "rb");
		fread(Conv6_depB, sizeof(paratype), Conv6_depChannel, fr);
		fclose(fr);

		fr = fopen(poi6weightdir, "rb");
		fread(Conv6_poiW, sizeof(paratype), Conv6_poiKernel*Conv6_poiKernel*Conv6_depChannel*Conv6_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi6biasdir, "rb");
		fread(Conv6_poiB, sizeof(paratype), Conv6_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep7weightdir, "rb");
		fread(Conv7_depW, sizeof(paratype), Conv7_depKernel*Conv7_depKernel*Conv7_depChannel, fr);
		fclose(fr);

		fr = fopen(dep7biasdir, "rb");
		fread(Conv7_depB, sizeof(paratype), Conv7_depChannel, fr);
		fclose(fr);

		fr = fopen(poi7weightdir, "rb");
		fread(Conv7_poiW, sizeof(paratype), Conv7_poiKernel*Conv7_poiKernel*Conv7_depChannel*Conv7_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi7biasdir, "rb");
		fread(Conv7_poiB, sizeof(paratype), Conv7_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep8weightdir, "rb");
		fread(Conv8_depW, sizeof(paratype), Conv8_depKernel*Conv8_depKernel*Conv8_depChannel, fr);
		fclose(fr);

		fr = fopen(dep8biasdir, "rb");
		fread(Conv8_depB, sizeof(paratype), Conv8_depChannel, fr);
		fclose(fr);

		fr = fopen(poi8weightdir, "rb");
		fread(Conv8_poiW, sizeof(paratype), Conv8_poiKernel*Conv8_poiKernel*Conv8_depChannel*Conv8_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi8biasdir, "rb");
		fread(Conv8_poiB, sizeof(paratype), Conv8_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep9weightdir, "rb");
		fread(Conv9_depW, sizeof(paratype), Conv9_depKernel*Conv9_depKernel*Conv9_depChannel, fr);
		fclose(fr);

		fr = fopen(dep9biasdir, "rb");
		fread(Conv9_depB, sizeof(paratype), Conv9_depChannel, fr);
		fclose(fr);

		fr = fopen(poi9weightdir, "rb");
		fread(Conv9_poiW, sizeof(paratype), Conv9_poiKernel*Conv9_poiKernel*Conv9_depChannel*Conv9_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi9biasdir, "rb");
		fread(Conv9_poiB, sizeof(paratype), Conv9_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep10weightdir, "rb");
		fread(Conv10_depW, sizeof(paratype), Conv10_depKernel*Conv10_depKernel*Conv10_depChannel, fr);
		fclose(fr);

		fr = fopen(dep10biasdir, "rb");
		fread(Conv10_depB, sizeof(paratype), Conv10_depChannel, fr);
		fclose(fr);

		fr = fopen(poi10weightdir, "rb");
		fread(Conv10_poiW, sizeof(paratype), Conv10_poiKernel*Conv10_poiKernel*Conv10_depChannel*Conv10_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi10biasdir, "rb");
		fread(Conv10_poiB, sizeof(paratype), Conv10_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep11weightdir, "rb");
		fread(Conv11_depW, sizeof(paratype), Conv11_depKernel*Conv11_depKernel*Conv11_depChannel, fr);
		fclose(fr);

		fr = fopen(dep11biasdir, "rb");
		fread(Conv11_depB, sizeof(paratype), Conv11_depChannel, fr);
		fclose(fr);

		fr = fopen(poi11weightdir, "rb");
		fread(Conv11_poiW, sizeof(paratype), Conv11_poiKernel*Conv11_poiKernel*Conv11_depChannel*Conv11_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi11biasdir, "rb");
		fread(Conv11_poiB, sizeof(paratype), Conv11_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep12weightdir, "rb");
		fread(Conv12_depW, sizeof(paratype), Conv12_depKernel*Conv12_depKernel*Conv12_depChannel, fr);
		fclose(fr);

		fr = fopen(dep12biasdir, "rb");
		fread(Conv12_depB, sizeof(paratype), Conv12_depChannel, fr);
		fclose(fr);

		fr = fopen(poi12weightdir, "rb");
		fread(Conv12_poiW, sizeof(paratype), Conv12_poiKernel*Conv12_poiKernel*Conv12_depChannel*Conv12_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi12biasdir, "rb");
		fread(Conv12_poiB, sizeof(paratype), Conv12_poiChannel, fr);
		fclose(fr);

		fr = fopen(dep13weightdir, "rb");
		fread(Conv13_depW, sizeof(paratype), Conv13_depKernel*Conv13_depKernel*Conv13_depChannel, fr);
		fclose(fr);

		fr = fopen(dep13biasdir, "rb");
		fread(Conv13_depB, sizeof(paratype), Conv13_depChannel, fr);
		fclose(fr);

		fr = fopen(poi13weightdir, "rb");
		fread(Conv13_poiW, sizeof(paratype), Conv13_poiKernel*Conv13_poiKernel*Conv13_depChannel*Conv13_poiChannel, fr);
		fclose(fr);

		fr = fopen(poi13biasdir, "rb");
		fread(Conv13_poiB, sizeof(paratype), Conv13_poiChannel, fr);
		fclose(fr);

	}
	//start caculate
	//printf("weight\n");
	//debugpr_offset(Conv1_poiW, 24,48);
	//printf("weight\n");
	//debugpr(Conv1_poiB, 1);
	//paratype* Conv2out=new paratype[Conv2outH*Conv2outH*Conv2Channel];
	//setzero(Conv2out, Conv2outH*Conv2outH*Conv2Channel);
	startclk = clock();
	conv2layer(IN graphin, Conv2W, Conv2B, OUT conv1_result);
	finishclk = clock();
	duration = (double)(finishclk - startclk) / CLOCKS_PER_SEC;
	printf("conv2: %lf\n", duration);
	//separable 1
	startclk = clock();
	depthwiselayer(IN conv1_result, Conv1_depW, Conv1_depB, OUT dep1_result,
		Conv2Channel, Conv2outH, Conv1_depKernel, Conv1_depChannel, Conv1_depoutH, Conv1_depStride);
	finishclk = clock();
	duration = (double)(finishclk - startclk) / CLOCKS_PER_SEC;
	printf("dep1: %lf\n", duration);
	//printf("dep1\n");
	//debugpr(dep1_result, 24);
	//pointwise
	startclk = clock();
	pointwiselayer_nopad(IN dep1_result, Conv1_poiW, Conv1_poiB, poi1_result, Conv1_depChannel, Conv1_depoutH, Conv1_poiKernel, Conv1_poiChannel, Conv1_poioutH);
	finishclk = clock();
	duration = (double)(finishclk - startclk) / CLOCKS_PER_SEC;
	printf("poi1: %lf\n", duration);
	//delete[] dep1_result;
	/*
	printf("weight\n");
	debugpr_offset(Conv2_poiW,48,96);
	printf("weight\n");
	debugpr(Conv2_poiB, 1);
	*/
	//separable 2
	depthwiselayer(IN poi1_result, Conv2_depW, Conv2_depB, OUT dep2_result,
		Conv1_poiChannel, Conv1_poioutH, Conv2_depKernel, Conv2_depChannel, Conv2_depoutH, Conv2_depStride);
	//delete[] poi1_result;
	//pointwise
	pointwiselayer_nopad(IN dep2_result, Conv2_poiW, Conv2_poiB, poi2_result, Conv2_depChannel, Conv2_depoutH, Conv2_poiKernel, Conv2_poiChannel, Conv2_poioutH);
	//delete[] dep2_result;
	//separable 3
	depthwiselayer(IN poi2_result, Conv3_depW, Conv3_depB, OUT dep3_result,
		Conv2_poiChannel, Conv2_poioutH, Conv3_depKernel, Conv3_depChannel, Conv3_depoutH, Conv3_depStride);
	//delete[] poi2_result;
	//pointwise
	pointwiselayer_nopad(IN dep3_result, Conv3_poiW, Conv3_poiB, poi3_result, Conv3_depChannel, Conv3_depoutH, Conv3_poiKernel, Conv3_poiChannel, Conv3_poioutH);
	//delete[] dep3_result;
	//separable 4
	depthwiselayer(IN poi3_result, Conv4_depW, Conv4_depB, OUT dep4_result,
		Conv3_poiChannel, Conv3_poioutH, Conv4_depKernel, Conv4_depChannel, Conv4_depoutH, Conv4_depStride);
	//delete[] poi3_result;
	//pointwise
	pointwiselayer_nopad(IN dep4_result, Conv4_poiW, Conv4_poiB, poi4_result, Conv4_depChannel, Conv4_depoutH, Conv4_poiKernel, Conv4_poiChannel, Conv4_poioutH);
	//delete[] dep4_result;
	//separable 5
	depthwiselayer(IN poi4_result, Conv5_depW, Conv5_depB, OUT dep5_result,
		Conv4_poiChannel, Conv4_poioutH, Conv5_depKernel, Conv5_depChannel, Conv5_depoutH, Conv5_depStride);
	//delete[] poi4_result;
	//pointwise
	pointwiselayer_nopad(IN dep5_result, Conv5_poiW, Conv5_poiB, poi5_result, Conv5_depChannel, Conv5_depoutH, Conv5_poiKernel, Conv5_poiChannel, Conv5_poioutH);
	//delete[] dep5_result;
	//separable 6
	depthwiselayer(IN poi5_result, Conv6_depW, Conv6_depB, OUT dep6_result,
		Conv5_poiChannel, Conv5_poioutH, Conv6_depKernel, Conv6_depChannel, Conv6_depoutH, Conv6_depStride);
	//delete[] poi5_result;
	//pointwise
	pointwiselayer_nopad(IN dep6_result, Conv6_poiW, Conv6_poiB, poi7_result, Conv6_depChannel, Conv6_depoutH, Conv6_poiKernel, Conv6_poiChannel, Conv6_poioutH);
	//delete[] dep6_result;

	//separable 7
	depthwiselayer(IN poi7_result, Conv7_depW, Conv7_depB, OUT dep7_result,
		Conv6_poiChannel, Conv6_poioutH, Conv7_depKernel, Conv7_depChannel, Conv7_depoutH, Conv7_depStride);

	//pointwise
	pointwiselayer_nopad(IN dep7_result, Conv7_poiW, Conv7_poiB, poi7_result, Conv7_depChannel, Conv7_depoutH, Conv7_poiKernel, Conv7_poiChannel, Conv7_poioutH);


	//separable 8
	depthwiselayer(IN poi7_result, Conv8_depW, Conv8_depB, OUT dep7_result,
		Conv7_poiChannel, Conv7_poioutH, Conv8_depKernel, Conv8_depChannel, Conv8_depoutH, Conv8_depStride);

	//pointwise
	pointwiselayer_nopad(IN dep7_result, Conv8_poiW, Conv8_poiB, poi7_result, Conv8_depChannel, Conv8_depoutH, Conv8_poiKernel, Conv8_poiChannel, Conv8_poioutH);

	//separable 9
	depthwiselayer(IN poi7_result, Conv9_depW, Conv9_depB, OUT dep7_result,
		Conv8_poiChannel, Conv8_poioutH, Conv9_depKernel, Conv9_depChannel, Conv9_depoutH, Conv9_depStride);

	//pointwise
	pointwiselayer_nopad(IN dep7_result, Conv9_poiW, Conv9_poiB, poi7_result, Conv9_depChannel, Conv9_depoutH, Conv9_poiKernel, Conv9_poiChannel, Conv9_poioutH);

	startclk = clock();
	//separable 10
	depthwiselayer(IN poi7_result, Conv10_depW, Conv10_depB, OUT dep7_result,
		Conv9_poiChannel, Conv9_poioutH, Conv10_depKernel, Conv10_depChannel, Conv10_depoutH, Conv10_depStride);
	finishclk = clock();
	duration = (double)(finishclk - startclk) / CLOCKS_PER_SEC;
	printf("dep10: %lf\n", duration);
	//pointwise
	startclk = clock();
	pointwiselayer_nopad(IN dep7_result, Conv10_poiW, Conv10_poiB, poi7_result, Conv10_depChannel, Conv10_depoutH, Conv10_poiKernel, Conv10_poiChannel, Conv10_poioutH);
	finishclk = clock();
	duration = (double)(finishclk - startclk) / CLOCKS_PER_SEC;
	printf("poi10: %lf\n", duration);
	//separable 11
	startclk = clock();
	depthwiselayer(IN poi7_result, Conv11_depW, Conv11_depB, OUT dep7_result,
		Conv10_poiChannel, Conv10_poioutH, Conv11_depKernel, Conv11_depChannel, Conv11_depoutH, Conv11_depStride);
	finishclk = clock();
	duration = (double)(finishclk - startclk) / CLOCKS_PER_SEC;
	printf("dep11: %lf\n", duration);
	//pointwise
	startclk = clock();
	pointwiselayer_nopad(IN dep7_result, Conv11_poiW, Conv11_poiB, poi7_result, Conv11_depChannel, Conv11_depoutH, Conv11_poiKernel, Conv11_poiChannel, Conv11_poioutH);
	finishclk = clock();
	duration = (double)(finishclk - startclk) / CLOCKS_PER_SEC;
	printf("poi11: %lf\n", duration);
	//separable 12
	depthwiselayer(IN poi7_result, Conv12_depW, Conv12_depB, OUT dep7_result,
		Conv11_poiChannel, Conv11_poioutH, Conv12_depKernel, Conv12_depChannel, Conv12_depoutH, Conv12_depStride);

	//pointwise
	pointwiselayer_nopad(IN dep7_result, Conv12_poiW, Conv12_poiB, poi7_result, Conv12_depChannel, Conv12_depoutH, Conv12_poiKernel, Conv12_poiChannel, Conv12_poioutH);

	//separable 13
	depthwiselayer(IN poi7_result, Conv13_depW, Conv13_depB, OUT dep7_result,
		Conv12_poiChannel, Conv12_poioutH, Conv13_depKernel, Conv13_depChannel, Conv13_depoutH, Conv13_depStride);

	//pointwise
	pointwiselayer_nopad(IN dep7_result, Conv13_poiW, Conv13_poiB, grahout, Conv13_depChannel, Conv13_depoutH, Conv13_poiKernel, Conv13_poiChannel, Conv13_poioutH);
	printf("poiproddurationsum,:%lf\n", prodtime);
}

//graghin HWC
//Conv2W in   careful for padding
//
void conv2layer(IN paratype* graphin, paratype* Conv2W, paratype* Conv2B, OUT paratype* Conv2out)
{
	//we have to care about cache use!
	//paratype *convkernel = new paratype[Conv2Kernel*Conv2Kernel*graphinC];
	paratype *onepatch = new paratype[Conv2Kernel*Conv2Kernel*graphinC];
	int prodnum = Conv2Kernel * Conv2Kernel*graphinC;
	for (int hi = 0; hi < Conv2outH; hi++)  //height
	{
		for (int j = 0; j < Conv2outH; j++)  //width
		{
			//input data reuse
			//handle paddle
			if (hi == 0 || j == 0 || hi == Conv2outH - 1 || j == Conv2outH - 1)
			{ //for corner and edge
				if (hi == 0 && j == 0)
				{

					int offset = 0;
					setzero(onepatch,
						(1 + Conv2Kernel)*graphinC);
					offset += (1 + Conv2Kernel)*graphinC;
					//copy 
					copyval(onepatch + offset,
						graphin, graphinC * 2);
					offset += graphinC * 2;
					setzero(onepatch + offset,
						graphinC);
					offset += graphinC;
					copyval(onepatch + offset,
						graphin + graphinW * graphinC, graphinC * 2);
					//printf("patchzero\n");
					//debugpr(onepatch, 27);

				}
				else if (hi == Conv2outH - 1 && j == 0)
				{
					int offset = 0;
					setzero(onepatch, graphinC);
					offset += graphinC;
					//copy 
					copyval(onepatch + offset,
						graphin + (graphinW - 2)*graphinW*graphinC, graphinC * 2);
					offset += graphinC * 2;
					//set zero
					setzero(onepatch + offset, graphinC);

					offset += graphinC;

					copyval(onepatch + offset,
						graphin + (graphinW - 1)*graphinW*graphinC, graphinC * 2);
					offset += graphinC * 2;
					setzero(onepatch + offset, graphinC*Conv2Kernel);
				}
				else if (hi == Conv2outH - 1 && j == Conv2outH - 1)
				{
					int offset = 0;

					copyval(onepatch + offset,
						graphin + ((graphinW - 1)*graphinW - 2)*graphinC, graphinC * 2);
					offset += graphinC * 2;
					setzero(onepatch + offset, graphinC);
					offset += graphinC;
					//copy 
					copyval(onepatch + offset,
						graphin + ((graphinW)*graphinW - 2)*graphinC, graphinC * 2);
					offset += graphinC * 2;
					//set zero
					setzero(onepatch + offset, graphinC * 4);
					//printf("patch");
					//debugpr(onepatch, 27);
				}
				else if (hi == 0 && j == Conv2outH - 1)
				{
					int offset = 0;
					setzero(onepatch, graphinC * 3);
					offset += graphinC * 3;

					copyval(onepatch + offset,
						graphin + (graphinW - 2)*graphinC, graphinC * 2);
					offset += graphinC * 2;

					setzero(onepatch + offset, graphinC);

					offset += graphinC;

					copyval(onepatch + offset,
						graphin + (graphinW * 2 - 2)*graphinC, graphinC * 2);
					offset += graphinC * 2;

					setzero(onepatch + offset, graphinC);
				}
				else if (j == 0)
				{  //add the right two cols
					int offset = 0;
					assert((Conv2Stride*hi + Conv2Kernel - 1)*graphinW*graphinC < 513 * 513 * 3);
					for (int patchind = 0; patchind < Conv2Kernel; patchind++)
					{
						setzero(onepatch + offset, graphinC);
						offset += graphinC;
						//copy 
						copyval(onepatch + offset,
							graphin + (Conv2Stride*hi + patchind - 1)*graphinW*graphinC, graphinC * 2);
						offset += graphinC * 2;
					}
				}
				else if (hi == 0)
				{  //add the upper two rows
					int offset = 0;
					assert((Conv2Stride*j + (Conv2Kernel - 1) * graphinW - 1)*graphinC, graphinC * Conv2Kernel< 513 * 513 * 3);
					setzero(onepatch, graphinC*Conv2Kernel);
					offset += graphinC * Conv2Kernel;
					for (int patchind = 0; patchind < Conv2Kernel - 1; patchind++)
					{
						copyval(onepatch + offset,
							graphin + (Conv2Stride*j + patchind * graphinW - 1)*graphinC, graphinC * Conv2Kernel);
						offset += graphinC * Conv2Kernel;
					}
				}
				else if (j == Conv2outH - 1)
				{  //add the right two cols
					int offset = 0;
					assert(((Conv2Stride * hi + (Conv2Kernel - 1)) * graphinW - 2)*graphinC< 513 * 513 * 3);
					for (int patchind = 0; patchind < Conv2Kernel; patchind++)
					{
						copyval(onepatch + offset,
							graphin + ((Conv2Stride * hi + patchind) * graphinW - 2)*graphinC, graphinC * 2);
						offset += graphinC * 2;
						setzero(onepatch + offset, graphinC);
						offset += graphinC;
					}
				}
				else if (hi == Conv2outH - 1)
				{  //add the lower two rows
					int offset = 0;
					assert(((graphinW - 2 + Conv2Kernel - 2) * graphinW + Conv2Stride * j - 1)*graphinC< 513 * 513 * 3);
					for (int patchind = 0; patchind < Conv2Kernel - 1; patchind++)
					{
						copyval(onepatch + offset,
							graphin + ((graphinW - 2 + patchind) * graphinW + Conv2Stride * j - 1)*graphinC, graphinC * Conv2Kernel);
						offset += graphinC * Conv2Kernel;
					}
					setzero(onepatch + offset, graphinC * Conv2Kernel);
				}

			}
			else
			{

				assert((Conv2Stride * (j - 1) + Conv2Kernel - 1 + (1 + Conv2Stride * (hi - 1) + Conv2Kernel - 1)*graphinW)*graphinC< 513 * 513 * 3);
				for (int ph = 0; ph < Conv2Kernel; ph++)  //without paddle
				{
					for (int pw = 0; pw < Conv2Kernel; pw++)
					{   //copy image in 

						copyval(onepatch + (pw + ph * Conv2Kernel)*graphinC, graphin + (Conv2Stride*j - 1 + pw + (Conv2Stride * hi - 1 + ph)*graphinW)*graphinC, graphinC);
					}
				}
			}


			for (int c = 0; c < Conv2Channel; c++)  //channel
			{

				//copyval_offset(convkernel, Conv2W + c, Conv2Kernel*Conv2Kernel*graphinC, Conv2Channel);
				//conv and add bias
				assert(hi * (Conv2outH*Conv2Channel) + j * Conv2Channel + c < 257 * 257 * 24);
				tensorprod_offsetw(onepatch, Conv2W + c,
					Conv2out + hi * (Conv2outH*Conv2Channel) + j * Conv2Channel + c, prodnum, Conv2Channel, Conv2B[c]);
				//relu
				//relu6(Conv2out + hi * (Conv2outH*Conv2Channel) + j * Conv2Channel + c);
				/*
				printf("weight\n");
				debugpr(convkernel, 27);

				printf("patchzero\n");
				debugpr(Conv2out + hi * (Conv2outH*Conv2Channel) + j * Conv2Channel + c, 1);
				*/
			}
		}

	}

	delete[]onepatch;
	//delete[]convkernel;
}

//careful for padding 
//this is a vec product based implemention
void depthwiselayer(IN paratype* graphin, paratype* depthwiseW, paratype* depthwiseB, OUT paratype* Convout, int inputc, int inputH, int depKernel, int depChannel, int depoutH, int depStride)
{
	//we have to care about cache use!

	//paratype *convkernel = new paratype[depKernel*depKernel*inputc];
	paratype *onepatch = new paratype[depKernel*depKernel*inputc];

	//copyval(convkernel, depthwiseW, depKernel*depKernel*inputc);

	for (int hi = 0; hi < depoutH; hi++)  //height
	{
		for (int j = 0; j < depoutH; j++)  //width
		{
			//input data reuse
			//handle paddle
			if (hi == 0 || j == 0 || hi == depoutH - 1 || j == depoutH - 1)
			{ //for corner and edge
				if (hi == 0 && j == 0)
				{
					int offset = 0;
					setzero(onepatch,
						(1 + depKernel)*inputc);
					offset += (1 + depKernel)*inputc;
					//copy 
					copyval(onepatch + offset,
						graphin, inputc * 2);
					offset += inputc * 2;
					setzero(onepatch + offset,
						inputc);
					offset += inputc;
					copyval(onepatch + offset,
						graphin + inputH * inputc, inputc * 2);
					//printf("patchzero\n");
					//debugpr(onepatch, 27);

				}
				else if (hi == depoutH - 1 && j == 0)
				{
					int offset = 0;
					setzero(onepatch, inputc);
					offset += inputc;
					//copy 
					copyval(onepatch + offset,
						graphin + (inputH - 2)*inputH*inputc, inputc * 2);
					offset += inputc * 2;
					//set zero
					setzero(onepatch + offset, inputc);

					offset += inputc;

					copyval(onepatch + offset,
						graphin + (inputH - 1)*inputH*inputc, inputc * 2);
					offset += inputc * 2;
					setzero(onepatch + offset, inputc*depKernel);
				}
				else if (hi == depoutH - 1 && j == depoutH - 1)
				{
					int offset = 0;

					copyval(onepatch + offset,
						graphin + ((inputH - 1)*inputH - 2)*inputc, inputc * 2);
					offset += inputc * 2;
					setzero(onepatch + offset, inputc);
					offset += inputc;
					//copy 
					copyval(onepatch + offset,
						graphin + ((inputH)*inputH - 2)*inputc, inputc * 2);
					offset += inputc * 2;
					//set zero
					setzero(onepatch + offset, inputc * 4);
				}
				else if (hi == 0 && j == depoutH - 1)
				{
					int offset = 0;
					setzero(onepatch, inputc * 3);
					offset += inputc * 3;

					copyval(onepatch + offset,
						graphin + (inputH - 2)*inputc, inputc * 2);
					offset += inputc * 2;

					setzero(onepatch + offset, inputc);

					offset += inputc;

					copyval(onepatch + offset,
						graphin + (inputH * 2 - 2)*inputc, inputc * 2);
					offset += inputc * 2;

					setzero(onepatch + offset, inputc);
				}
				else if (j == 0)
				{  //add the right two cols
					int offset = 0;
					assert((depStride*hi + depKernel - 2)*inputH*inputc < inputH*inputH*inputc);
					for (int patchind = 0; patchind < depKernel; patchind++)
					{
						setzero(onepatch + offset, inputc);
						offset += inputc;
						//copy 
						copyval(onepatch + offset,
							graphin + (depStride*hi + patchind - 1)*inputH*inputc, inputc * 2);
						offset += inputc * 2;
					}
				}
				else if (hi == 0)
				{  //add the upper two rows
					int offset = 0;
					assert((depStride*j + (depKernel - 1) * inputH - 1)*inputc, inputc * depKernel< inputH*inputH*inputc);
					setzero(onepatch, inputc*depKernel);
					offset += inputc * depKernel;
					for (int patchind = 0; patchind < depKernel - 1; patchind++)
					{
						copyval(onepatch + offset,
							graphin + (depStride*j + patchind * inputH - 1)*inputc, inputc * depKernel);
						offset += inputc * depKernel;
					}
				}
				else if (j == depoutH - 1)
				{  //add the right two cols
					int offset = 0;
					assert(((depStride * hi + (depKernel - 1)) * inputH - 2)*inputc< inputH*inputH*inputc);
					for (int patchind = 0; patchind < depKernel; patchind++)
					{
						copyval(onepatch + offset,
							graphin + ((depStride * hi + patchind) * inputH - 2)*inputc, inputc * 2);
						offset += inputc * 2;
						setzero(onepatch + offset, inputc);
						offset += inputc;
					}
				}
				else if (hi == depoutH - 1)
				{  //add the lower two rows
					int offset = 0;
					assert(((inputH - 2 + depKernel - 2) * inputH + depStride * j - 1)*inputc< inputH*inputH*inputc);
					for (int patchind = 0; patchind < depKernel - 1; patchind++)
					{
						copyval(onepatch + offset,
							graphin + ((inputH - 2 + patchind) * inputH + depStride * j - 1)*inputc, inputc * depKernel);
						offset += inputc * depKernel;
					}
					setzero(onepatch + offset, inputc * depKernel);
				}

			}
			else
			{

				//assert((depStride*j - 1 + depKernel - 1 + (depStride * hi - 1 + depKernel - 1)*inputH)*inputc< inputH*inputH*inputc);
				for (int ph = 0; ph < depKernel; ph++)  //without paddle
				{
					for (int pw = 0; pw < depKernel; pw++)
					{   //copy image in 

						copyval(onepatch + (pw + ph * depKernel)*inputc, graphin + (depStride*j - 1 + pw + (depStride * hi - 1 + ph)*inputH)*inputc, inputc);
					}
				}
			}

			//do not need multi channel

			//shoot
			//conv and add bias

			tensorprod_depwise(onepatch, depthwiseW,
				Convout + hi * (depoutH*depChannel) + j * depChannel, depChannel, depKernel, depthwiseB);
			//relu

		}

	}

	delete[]onepatch;
	//delete[]convkernel;
}

void pointwiselayer_nopad(IN paratype* graphin, paratype* depthwiseW, paratype* depthwiseB, OUT paratype* Convout, int inputc, int inputH, int depKernel, int depChannel, int depoutH)
{
	//we have to care about cache use!
	durationprod = 0;
	//paratype *convkernel = new paratype[depKernel*depKernel*inputc];
	int outoffset = depoutH * depChannel;
	int prodnum = depKernel * depKernel*inputc;
	//paratype *onepatch = new paratype[depKernel*depKernel*inputc];
	for (int hi = 0; hi < depoutH; hi++)  //height
	{
		for (int j = 0; j < depoutH; j++)  //width
		{

			for (int c = 0; c < depChannel; c++)  //channel
			{
			//input data reuse

			//copyval(onepatch, graphin + (j + hi * inputH)*inputc, inputc);



				//copyval_offset(convkernel, depthwiseW + c, depKernel*depKernel*inputc, depChannel);
				//shoot
				//conv and add bias
			
				tensorprod(graphin + (j + hi * inputH)*inputc, depthwiseW + c,
					Convout + hi * outoffset + j * depChannel + c, prodnum, depChannel,depthwiseB[c]);


				//relu6(result);
				//printf("from%f\n", product);
				//relu
				//relu6(Convout + hi * (depoutH*depChannel) + j * depChannel + c);

			}
		}
	}
	printf("poiprodduration,:%lf\n", durationprod);
	prodtime += durationprod;
}

void outlayer_norelu(IN paratype* graphin, paratype* depthwiseW, paratype* depthwiseB, OUT paratype* Convout, int inputc, int inputH, int depKernel, int depChannel, int depoutH, int depStride)
{
	//we have to care about cache use!
	int outoffset = depoutH * depChannel;
	int prodnum = depKernel*depKernel*inputc;
	//paratype *convkernel = new paratype[depKernel*depKernel*inputc];
	//paratype *onepatch = new paratype[depKernel*depKernel*inputc];
	for (int hi = 0; hi < depoutH; hi++)  //height
	{
		for (int j = 0; j < depoutH; j++)  //width
		{
			//input data reuse

			//assert((depStride*j + depKernel - 1 + (depStride * hi + depKernel - 1)*inputH)*inputc< inputH*inputH*inputc);
			//copyval(onepatch, graphin + (j + hi * inputH)*inputc, inputc);

			for (int c = 0; c < depChannel; c++)  //channel
			{
				
				//shoot
				//conv and add bias
				tensorprod_norelu(graphin + (j + hi * inputH)*inputc, depthwiseW + c,
					Convout + hi * (outoffset) + j * depChannel + c, prodnum, depChannel, depthwiseB[c]);

			}
		}

	}
	//delete[]onepatch;
	//delete[]convkernel;
}

