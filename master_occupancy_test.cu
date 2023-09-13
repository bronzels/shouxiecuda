#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void occupancy_test(int * results)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int x1 = 1;
    int x2 = 2;
    int x3 = 3;
    int x4 = 4;
    int x5 = 5;
    int x6 = 6;
    int x7 = 7;
    int x8 = 8;
    results[gid] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8;
}
/*
nvcc --ptxas-options=-v -o master_occupancy_test.out master_occupancy_test.cu
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z14occupancy_testPi' for 'sm_52'
ptxas info    : Function properties for _Z14occupancy_testPi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 4 registers, 328 bytes cmem[0]
 */

int main()
{
    int size = 64;
    int byte_size = sizeof(int) * size;

    int * d_input;
    cudaMalloc((void **)&d_input, byte_size);

    dim3 block(4, 32);
    dim3 grid(1);
    occupancy_test <<< grid, block >>> (d_input);
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}