#include "stdafx.h"

#include<stdio.h>
#include "convlayer.h"
#include <cmath>
#define Heatmap_Kernel 1
#define Heatmap_Channel 17
#define Heatmap_outH 33
#define Heatmap_Stride 1

#define offset_2_Kernel 1
#define offset_2_Channel 34
#define offset_2_outH 33
#define offset_2_Stride 1

#define disp_Kernel 1
#define disp_Channel 34
#define disp_outH 33
#define disp_Stride 1

const char* Heatmap2_weightdir = "./wbin/heatmap_2_weights.bin";
const char* Heatmap2_biasdir = "./wbin/heatmap_2_biases.bin";

const char* offset2_weightdir = "./wbin/offset_2_weights.bin";
const char* offset2_biasdir = "./wbin/offset_2_biases.bin";

const char* disp2_fwd_weightdir = "./wbin/displacement_fwd_2_weights.bin";
const char* disp2_fwd_biasdir = "./wbin/displacement_fwd_2_biases.bin";

const char* disp2_bwd_weightdir = "./wbin/displacement_bwd_2_weights.bin";
const char* disp2_bwd_biasdir = "./wbin/displacement_bwd_2_biases.bin";

inline void sigmoidmul(paratype* dataforsigm, int num)
{
	for (int i = 0; i < num; i++)
	{
		dataforsigm[i] = 1 / (1 + exp(-dataforsigm[i]));
	}
}

int main()
{
	printf("hh");
	paratype* graphin, * graghout,*graphtest,
		*heatmap,*offset2, *displacement_fwd, *displacement_bwd,
		*heatmap_2_weight, *heatmap_2_bias,
		*offset_2_weight, *offset_2_bias, *displacement_fwd_weight, *displacement_fwd_bias, *displacement_bwd_weight, *displacement_bwd_bias;
	
	graphin = new paratype[513 * 513 * 3];
	heatmap = new paratype[Heatmap_outH*Heatmap_outH*Heatmap_Channel];
	offset2 = new paratype[offset_2_outH*offset_2_outH*offset_2_Channel];
	displacement_fwd = new paratype[disp_outH*disp_outH*disp_Channel];
	displacement_bwd = new paratype[disp_outH*disp_outH*disp_Channel];
	heatmap_2_weight = new paratype[Heatmap_outH * Heatmap_outH * Heatmap_Channel];
	heatmap_2_bias = new paratype[Heatmap_Channel];
	offset_2_weight = new paratype[offset_2_outH * offset_2_outH * offset_2_Channel];
	offset_2_bias = new paratype[offset_2_Channel];
	displacement_fwd_weight = new paratype[disp_outH * disp_outH * disp_Channel];
	displacement_fwd_bias = new paratype[disp_Channel];
	displacement_bwd_weight = new paratype[disp_outH * disp_outH * disp_Channel];
	displacement_bwd_bias = new paratype[disp_Channel];
	
	FILE *fr = fopen("./testimage.bin", "rb");
	fread(graphin, sizeof(paratype), 513 * 513 * 3, fr);
	fclose(fr);

	
	int convoutsize = Conv7_poioutH * Conv7_poioutH* Conv13_poiChannel;
	int checksize = offset_2_outH * offset_2_outH*offset_2_Channel;
	
	graphtest = new paratype[checksize];
	fr = fopen("./check/testoffset.bin", "rb");
	fread(graphtest, sizeof(paratype), checksize, fr);
	fclose(fr);
	fr = fopen(Heatmap2_weightdir, "rb");
	fread(heatmap_2_weight, sizeof(paratype), Heatmap_outH * Heatmap_outH * Heatmap_Channel, fr);
	fclose(fr);
	fr = fopen(Heatmap2_biasdir, "rb");
	fread(heatmap_2_bias, sizeof(paratype), Heatmap_Channel, fr);
	fclose(fr);
	fr = fopen(offset2_weightdir, "rb");
	fread(offset_2_weight, sizeof(paratype), offset_2_outH * offset_2_outH * offset_2_Channel, fr);
	fclose(fr);
	fr = fopen(offset2_biasdir, "rb");
	fread(offset_2_bias, sizeof(paratype), offset_2_Channel, fr);
	fclose(fr);
	fr = fopen(disp2_fwd_weightdir, "rb");
	fread(displacement_fwd_weight, sizeof(paratype), disp_outH * disp_outH * disp_Channel, fr);
	fclose(fr);
	fr = fopen(disp2_fwd_weightdir, "rb");
	fread(displacement_fwd_bias, sizeof(paratype), disp_Channel, fr);
	fclose(fr);
	fr = fopen(disp2_bwd_weightdir, "rb");
	fread(displacement_bwd_weight, sizeof(paratype), disp_outH * disp_outH * disp_Channel, fr);
	fclose(fr);
	fr = fopen(disp2_bwd_weightdir, "rb");
	fread(displacement_bwd_weight, sizeof(paratype), disp_Channel, fr);
	fclose(fr);

	graghout = new paratype[convoutsize];

	convmodule(graphin, graghout);
	printf("convcomplet/n");
	
	outlayer_norelu(IN graghout, heatmap_2_weight, heatmap_2_bias, heatmap, Conv13_poiChannel, Conv13_poioutH, Heatmap_Kernel, Heatmap_Channel, Heatmap_outH, Heatmap_Stride);
	sigmoidmul(heatmap, Heatmap_outH*Heatmap_outH*Heatmap_Channel);
	outlayer_norelu(IN graghout, offset_2_weight, offset_2_bias, offset2, Conv13_poiChannel, Conv13_poioutH, offset_2_Kernel, offset_2_Channel, offset_2_outH, offset_2_Stride);

	outlayer_norelu(IN graghout, displacement_fwd_weight, displacement_fwd_bias, displacement_fwd, Conv13_poiChannel, Conv13_poioutH, disp_Kernel, disp_Channel, disp_outH, disp_Stride);

	outlayer_norelu(IN graghout, displacement_bwd_weight, displacement_bwd_bias, displacement_bwd, Conv13_poiChannel, Conv13_poioutH, disp_Kernel, disp_Channel, disp_outH, disp_Stride);

	int checkoffset = (12*33)* 17;
	int numcheck = 17;
	printf("networkoutput\n");
	for (int i = 0; i <numcheck; i++)
	{
		printf("%f,", offset2[checkoffset +i]);

	}
	printf("\nout\n");
	for (int i = 0; i <numcheck; i++)
	{
		printf("%f,", graphtest[checkoffset + i]);

	}
	return 0;
}
