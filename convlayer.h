/*
header for conv module
Conv 2 layer
Separable conv layer 

*/
#include "stdafx.h"
#ifndef CONV_H
#define CONV_H

#define paratype float 
#define IN 
#define OUT 
#define graphinW 513
#define graphinH 513
#define graphinC 3
#define Conv2Kernel 3
#define Conv2Channel 24
#define Conv2outH 257
#define Conv2Stride 2

#define Conv1_depKernel 3
#define Conv1_depChannel 24
#define Conv1_depoutH 257
#define Conv1_depStride 1

#define Conv1_poiKernel 1
#define Conv1_poiChannel 48
#define Conv1_poioutH 257
#define Conv1_poiStride 1

#define Conv2_depKernel 3
#define Conv2_depChannel 48
#define Conv2_depoutH 129
#define Conv2_depStride 2
#define Conv2_poiKernel 1
#define Conv2_poiChannel 96
#define Conv2_poioutH 129
#define Conv2_poiStride 1

#define Conv3_depKernel 3
#define Conv3_depChannel 96
#define Conv3_depoutH 129
#define Conv3_depStride 1
#define Conv3_poiKernel 1
#define Conv3_poiChannel 96
#define Conv3_poioutH 129
#define Conv3_poiStride 1

#define Conv4_depKernel 3
#define Conv4_depChannel 96
#define Conv4_depoutH 65
#define Conv4_depStride 2
#define Conv4_poiKernel 1
#define Conv4_poiChannel 192
#define Conv4_poioutH 65
#define Conv4_poiStride 1

#define Conv5_depKernel 3
#define Conv5_depChannel 192
#define Conv5_depoutH 65
#define Conv5_depStride 1
#define Conv5_poiKernel 1
#define Conv5_poiChannel 192
#define Conv5_poioutH 65
#define Conv5_poiStride 1

#define Conv6_depKernel 3
#define Conv6_depChannel 192
#define Conv6_depoutH 33
#define Conv6_depStride 2
#define Conv6_poiKernel 1
#define Conv6_poiChannel 384
#define Conv6_poioutH 33
#define Conv6_poiStride 1

#define Conv7_depKernel 3
#define Conv7_depChannel 384
#define Conv7_depoutH 33
#define Conv7_depStride 1
#define Conv7_poiKernel 1
#define Conv7_poiChannel 384
#define Conv7_poioutH 33
#define Conv7_poiStride 1

#define Conv8_depKernel 3
#define Conv8_depChannel 384
#define Conv8_depoutH 33
#define Conv8_depStride 1
#define Conv8_poiKernel 1
#define Conv8_poiChannel 384
#define Conv8_poioutH 33
#define Conv8_poiStride 1

#define Conv9_depKernel 3
#define Conv9_depChannel 384
#define Conv9_depoutH 33
#define Conv9_depStride 1
#define Conv9_poiKernel 1
#define Conv9_poiChannel 384
#define Conv9_poioutH 33
#define Conv9_poiStride 1

#define Conv10_depKernel 3
#define Conv10_depChannel 384
#define Conv10_depoutH 33
#define Conv10_depStride 1
#define Conv10_poiKernel 1
#define Conv10_poiChannel 384
#define Conv10_poioutH 33
#define Conv10_poiStride 1

#define Conv11_depKernel 3
#define Conv11_depChannel 384
#define Conv11_depoutH 33
#define Conv11_depStride 1
#define Conv11_poiKernel 1
#define Conv11_poiChannel 384
#define Conv11_poioutH 33
#define Conv11_poiStride 1

#define Conv12_depKernel 3
#define Conv12_depChannel 384
#define Conv12_depoutH 33
#define Conv12_depStride 1
#define Conv12_poiKernel 1
#define Conv12_poiChannel 384
#define Conv12_poioutH 33
#define Conv12_poiStride 1

#define Conv13_depKernel 3
#define Conv13_depChannel 384
#define Conv13_depoutH 33
#define Conv13_depStride 1
#define Conv13_poiKernel 1
#define Conv13_poiChannel 384
#define Conv13_poioutH 33
#define Conv13_poiStride 1

void convmodule(paratype* graphin, paratype* grahout);
void conv2layer(IN paratype* graphin, paratype* Conv2W, paratype* Conv2B, OUT paratype* Conv2out);


void depthwiselayer(IN paratype* graphin, paratype* depthwiseW, paratype* depthwiseB, OUT paratype* Convout,int inputc,int inputH, int depKernel,int depChannel,int depoutH,int depStride);

void pointwiselayer_nopad(IN paratype* graphin, paratype* depthwiseW, paratype* depthwiseB, OUT paratype* Convout, int inputc, int inputH, int depKernel, int depChannel, int depoutH, int depStride);

void outlayer_norelu(IN paratype* graphin, paratype* depthwiseW, paratype* depthwiseB, OUT paratype* Convout, int inputc, int inputH, int depKernel, int depChannel, int depoutH, int depStride);

void pointwiselayer_nopad_multi2(IN paratype* graphin, paratype* depthwiseW, paratype* depthwiseB, OUT paratype* Convout, int inputc, int inputH, int depKernel, int depChannel, int depoutH, int depStride);


#endif