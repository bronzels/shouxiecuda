#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void code_without_divergence()
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    float a, b;
    a = b = 0;

    int warp_id = gid / 32;

    if (warp_id % 2 == 0)
    {
        a = 100.0;
        b = 50;
    }
    else
    {
        a = 200.0;
        b = 75.0;
    }
}

__global__ void code_with_divergence()
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    float a, b;
    a = b = 0;

    int warp_id = gid / 32;

    if (gid % 2 == 0)
    {
        a = 100.0;
        b = 50.0;
    }
    else
    {
        a = 200.0;
        b = 75.0;
    }
}

int main()
{
    dim3 block(32);
    dim3 grid(2, 2);
    code_without_divergence <<< grid, block >>> ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}

/*
host
nsys profile --stats=true master_warp_divergency.out

mac CP
*/
