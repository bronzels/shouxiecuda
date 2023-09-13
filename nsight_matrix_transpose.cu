#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// m the number of rows of input matrix
// n the number of cols of input matrix
__global__ void transposeNative(float *input, float *output, int m, int n)
{
    int colID_input = threadIdx.x + blockDim.x*blockIdx.x;
    int rowID_input = threadIdx.y + blockDim.y*blockIdx.y;

    if (rowID_input < m && colID_input < n)
    {
        int index_input = colID_input + rowID_input*n;
        int index_output = rowID_input + colID_input *m;

        output[index_output] = input[index_input];
    }
}

__global__ void transposeNativeOptimized(float *input, float *output, int m, int n)
{
    int colID_input = threadIdx.x + blockDim.x*blockIdx.x;
    int rowID_input = threadIdx.y + blockDim.y*blockIdx.y;

    __shared__ float sdata[32][33];
    //__shared__ float sdata[32][32];
    if (rowID_input < m && colID_input < n)
    {
        int index_input = colID_input + rowID_input*n;
        sdata[threadIdx.y][threadIdx.x] = input[index_input];

        __syncthreads();

        int dst_col = threadIdx.x + blockIdx.y * blockDim.y;
        int dst_row = threadIdx.y + blockIdx.x * blockDim.x;
        output[dst_col + dst_row*m] = sdata[threadIdx.x][threadIdx.y];
    }
}

inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert : %s %s %d \n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

int main()
{
    int blockx = 32;
    int bloxy = 32;
    int grimx = 4*64;
    int grimy = 4*64;
    dim3 block(blockx, bloxy);
    dim3 grid(grimx, grimy);
    int threads = blockx * bloxy * grimx * grimy;
    int matrix_x = blockx * grimx;
    int matrix_y = bloxy * grimy;

    float *dIn, *dOut;
    cudaMalloc((void **)&dIn, threads * 4);
    gpuErrchk(cudaMalloc((void **)&dOut, threads * 4));
    //transposeNative <<< grid, block >>> (dIn, dOut, matrix_x, matrix_y);
    //8.52 msec, l1/text->L2=2.15GB, l1/text<-L2=268.44MB
    transposeNativeOptimized <<< grid, block >>> (dIn, dOut, matrix_x, matrix_y);
    //col33, 2.52 msec, l1/text->L2=268.44MB, l1/text<-L2=268.44MB
    //col32, 4.11 msec, l1/text->L2=268.44MB, l1/text<-L2=268.44MB
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
