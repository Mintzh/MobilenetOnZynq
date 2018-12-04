#include "convlayer.h"
//implement the convolution
inline void relu6(paratype &dataforrelu)
{
  if(dataforrelu<0)
  {
	  dataforrelu=0;
  }
  else if(dataforrelu>6)
  {
	  dataforrelu=6;
  }
  else
  {
	  dataforrelu=dataforrelu;  
  }	
}

inline void conv3(paratype* graph,paratype* convweight,paratype* result)
{
   for(i=0;i<9;i++)
   {
	   *result += graph[i]*convweight[i];
   }
}

inline void bias(paratype* data,paratype bias,int datanum)
{
	for(i=0;i<datanum;i++)
	{
		data[i] += bias;
	}
	
}

void convmodule()

void conv2layer()

void convDPlayer(paratype* datain,paratype* dataout,paratype* params)

void convoutlayer()

