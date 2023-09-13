#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void mem_trs_test(int size, int * input)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size)
        printf("tid : %d, gid : %d, value : %d \n", threadIdx.x,gid,input[gid]);
}

int main()
{
    int size = 150;
    int byte_size = sizeof(int) * size;

    int * h_input;
    h_input = (int *)malloc(byte_size);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        h_input[i] = (int)(rand() & 0xff);
    }


    int * d_input;
    cudaMalloc((void **)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(5);
    mem_trs_test <<< grid, block >>> (size, d_input);
    cudaDeviceSynchronize();

    free(h_input);
    cudaFree(d_input);
    cudaDeviceReset();
    return 0;
}