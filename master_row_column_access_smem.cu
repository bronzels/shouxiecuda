#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.hpp"

#define BDIMX 32
#define BDIMY 32

#define IPAD 1

__global__ void setRowReadCol(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadRow(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRow(int * out)
{
    __shared__ int tile[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadColDyn(int * out)
{
    extern __shared__ int tile[];

    int row_index = threadIdx.y * blockDim.x + threadIdx.x;
    int col_index = threadIdx.x * blockDim.y + threadIdx.y;

    tile[row_index] = row_index;

    __syncthreads();

    out[row_index] = tile[col_index];
}

__global__ void setRowReadColPad(int * out)
{
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadRowPad(int * out)
{
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadColDynPad(int * out)
{
    extern __shared__ int tile[];

    int row_index = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    int col_index = threadIdx.x * (blockDim.y + IPAD) + threadIdx.y;

    tile[row_index] = row_index;

    __syncthreads();

    out[row_index] = tile[col_index];
}
/*
0,0---2.91 usec
0,1---2.30 usec, shared store, wavefronts 0.01(-96.88%), bank conflicts 0(-100.00%)
0,2---2.91 usec, shared load, wavefronts 0.01(-3100.00%), % Peak 0.98(+3079.14%), bank conflicts 992(-inf%)
                 shared store, wavefronts 32(-96.88%), % Peak 0.01(-96.90%), bank conflicts 0(-100.00%)
0,3---2.91 usec, shared load, wavefronts 1024(+3100.00%), % Peak 0.98(+3070.24%), bank conflicts 992(+inf%)
                 shared store, wavefronts 32(-96.88%), % Peak 0.01(-96.90%), bank conflicts 0(-100.00%)
0,4---2.18 usec, shared load, % Peak 0.04(+31.74%)
                 shared store, wavefronts 32(-96.88%), % Peak 0.01(-95.88%), bank conflicts 0(-100.00%)
0,5---2.21 usec, shared load, % Peak 0.04(+31.48%)
                 shared store, wavefronts 32(-96.88%), % Peak 0.01(-95.89%), bank conflicts 0(-100.00%)
0,6---2.91 usec, shared load, % Peak 0.04(+32.67%)
                 shared store, wavefronts 32(-96.88%), % Peak 0.01(-95.85%), bank conflicts 0(-100.00%)
设置8bytes查询还是4bytes
 */

int main(int argc, char **argv)
{
	int memconfig = 0;
    int test = 0;
	if (argc > 1)
	{
		memconfig = atoi(argv[1]);
	}
    if (argc > 2)
    {
        test = atoi(argv[2]);
    }


	if (memconfig == 1)
	{   printf("set 8 bytes\n");
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	}
	else
	{   printf("set 4 bytes\n");
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	}


    cudaSharedMemConfig pConfig;
    cudaDeviceGetSharedMemConfig(&pConfig);
	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");


	// set up array size 2048
	int nx = BDIMX;
	int ny = BDIMY;

	bool iprintf = 0;

	if (argc > 3) iprintf = atoi(argv[3]);

	size_t nBytes = nx * ny * sizeof(int);

	// execution configuration
	dim3 block(BDIMX, BDIMY);
	dim3 grid(1, 1);
	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
		block.y);

	// allocate device memory
	int *d_C;
	cudaMalloc((int**)&d_C, nBytes);
	int *gpuRef = (int *)malloc(nBytes);

	cudaMemset(d_C, 0, nBytes);
    void(*kernel)(int*);
    char * kernel_name;
    if(test == 1)
    {
        kernel_name = "set row read row   ";
        kernel = &setRowReadRow;
    }
    else if(test == 2)
    {
        kernel_name = "set row read col   ";
        kernel = &setRowReadCol;
    }
    else if(test == 3) {
        kernel_name = "set row read col dynamic  ";
        kernel = &setRowReadColDyn;
    }
    else if(test == 4)
    {
        kernel_name = "set col read row pad   ";
        kernel = &setColReadRowPad;
    }
    else if(test == 5) {
        kernel_name = "set row read col pad  ";
        kernel = &setRowReadColPad;
    }
    else if(test == 6) {
        kernel_name = "set row read col dynamic pad  ";
        kernel = &setRowReadColDynPad;
    }
    else
    {
        kernel_name = "set col read row   ";
        kernel = &setColReadRow;
    }
    if (test != 3 && test != 6)
        kernel <<<grid, block >>>(d_C);
    else
    {
        int new_nx = nx;
        if ( test == 6)
            new_nx += IPAD;
        kernel <<<grid, block, sizeof(int) * ny*new_nx >>>(d_C);
    }
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    printf("%s\n", kernel_name);
	if (iprintf)  printData(kernel_name, gpuRef, nx * ny);

	// free host and device memory
	cudaFree(d_C);
	free(gpuRef);

	// reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
