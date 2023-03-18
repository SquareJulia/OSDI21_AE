
#include <cnpy.h>

#include <vector>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cublas_v2.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
// we are doing AC = AB * BC, reduce across the B dimension
// binding B to the x dimension, A to the y dimension and C to the z dimension

#define Tsy 1
#define Tsz (16 / 2)
#define Gsy Tsz
#define Gy 1
#define Block_size (Gy * Gsy)
#define In_Format 'INPUT_FORMAT'
#define Out_Format 'OUTPUT_FORMAT'

namespace cg = cooperative_groups;

__global__ void mm(const float ** __restrict__ pBC, float ** pAC)
{
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
	{
		printf("In cubin mm!\n");
		printf("pBC:%p, *pBC:%p, pAC:%p, *pAC:%p\n",pBC,*pBC,pAC,*pAC);
	}
    register float ACC[3] = {0.0};
	register float RC = 0.0;
#if Gy > 1	
        __shared__ float result[3][Tsz];
	for(int i = threadIdx.x; i < 3 * Tsz; i += Block_size)
	{
		((float*)result)[i] = 0.0;
	}
	__syncthreads();
#endif
#if In_Format == 'NHWC'
	__shared__ float smem_cache[Tsz][TSB+1];
#endif
#if Out_Format == 'NHWC'
	__shared__ float smem_result[Tsz][3+1];
#endif

	int A_offset = blockIdx.x * (6 / 2);
	int C_offset = blockIdx.y * (16 / 2);
	int groupId = threadIdx.x / (Gsy);
	int lane = threadIdx.x % (Gsy);


if(blockIdx.x == 0)
{



	if(groupId == 0)
	{


		asm("//B0G0;START");

		RC = ((const float *)*pBC)[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 3; i++)
	{
    
        (*pAC)[(0 + i) * 16 + C_offset + lane] = ACC[i];
    }
    
#else
    for(int i = 0; i < 3; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 3 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//(*pAC)[0 + row][C_offset + col] = result[row][col];
		(*pAC)[(0 + row) * 16 + C_offset + col] = result[row][col];
	}
#endif       
       


}

if(blockIdx.x == 1)
{



	if(groupId == 0)
	{


		asm("//B1G0;START");

		RC = ((const float *)*pBC)[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 3; i++)
	{
    
        (*pAC)[(3 + i) * 16 + C_offset + lane] = ACC[i];
    }
    
#else
    for(int i = 0; i < 3; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 3 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//(*pAC)[3 + row][C_offset + col] = result[row][col];
		(*pAC)[(3 + row) * 16 + C_offset + col] = result[row][col];
	}
#endif       
       


}

}
int main()
{

	std::cout << "Group size " << Gsy << std::endl;

	cnpy::NpyArray arr = cnpy::npy_load("../SparseRT/materials/npys/my-test-graphs/g1.npy");
	float * AB = arr.data<float>();
	assert(arr.word_size = sizeof(float));
	assert(arr.shape.size()==2 && arr.shape[0] == 6 && arr.shape[1] == 6); //transposed

	cnpy::NpyArray arr1 = cnpy::npy_load("BC.npy");
	float * BC = arr1.data<float>();
	assert(arr1.word_size = sizeof(float));
#if In_Format == 'NHWC'
	assert(arr1.shape.size()==2 && arr1.shape[0] == 16 && arr1.shape[1] == 6);
#else
	assert(arr1.shape.size()==2 && arr1.shape[0] == 6 && arr1.shape[1] == 16);
#endif
    cnpy::NpyArray arr2 = cnpy::npy_load("ref.npy");
	float * AC = arr2.data<float>();
    std::cout << ((float *)AC)[0] << std::endl;

	float *d_BC, *d_AC, *d_residual;
	cudaMalloc((void**)&d_BC, 6 * 16 *sizeof(float));
	cudaMalloc((void**)&d_AC, 6 * 16 *sizeof(float));


	cudaMemcpy( d_BC,BC, 6 * 16 *sizeof(float), cudaMemcpyHostToDevice);

	float *result;
	result = (float *)malloc(6 * 16 *sizeof(result));

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 GS(2,2);

     std::cout << "warning: sometimes you might want to fix the launch dimensions to 32" << std::endl;

    for(int i = 0;i < 1000;i ++){
#if RESIDUAL
	    mm<<<GS,Gsy>>>(d_BC,d_residual,d_AC);
#else
        mm<<<GS,Gsy>>>(&d_BC,&d_AC);
#endif
    }

	cudaProfilerStart();
	cudaEventRecord(start);

	for(int i = 0;i < 1000;i ++){
#if RESIDUAL
	    mm<<<GS,Gsy>>>(d_BC,d_residual,d_AC);
#else
        mm<<<GS,Gsy>>>(&d_BC,&d_AC);
#endif
    }
	cudaEventRecord(stop);
	cudaProfilerStop();
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << "kernel used " << time / 1000.0 << std::endl;

	cudaMemcpy(result, d_AC, 6 * 16 *sizeof(float), cudaMemcpyDeviceToHost);

	float error = 0;
	for(int i = 0 ; i < 6 * 16; i ++)
	{
        error += abs(result[i] - ((float *)AC)[i]);
	}
	
	#if Out_Format == 'NCHW'
        cnpy::npy_save("result.npy",&result[0],{6,16},"w");
    #else
        cnpy::npy_save("result.npy",&result[0],{16,6},"w");
    #endif

	std::cout << result[0] << result[1] << result[2] << std::endl;
	std::cout << error << std::endl;
	cudaFree(d_BC);
	cudaFree(d_AC);
}
