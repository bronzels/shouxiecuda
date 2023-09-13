#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda()
{
    printf("Hello CUDa world \n");
}


__global__ void print_threadIdx()
{
    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d)\n",
           threadIdx.x,threadIdx.y,threadIdx.z);

}

__global__ void print_block_details()
{
    printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, blockDim.x: %d, blockDim.y: %d, blockDim.z: %d)\n",
           blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z);

}

__global__ void print_all_details()
{
    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, blockDim.x: %d, blockDim.y: %d, blockDim.z: %d, gridDim.x: %d, gridDim.y: %d, gridDim.z: %d)\n",
           blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);

}
int main()
{
    int nx, ny, nz;
    nx = 4;
    ny = 4;
    nz = 4;

    dim3 block(2, 2); //1024,1024,64, x*y*z < 1024
    dim3 grid(nx / block.x, ny / block.y, nz / block.z); //1<<32-1,65536,65536,
    hello_cuda <<< grid, block >>> ();
    //print_threadIdx <<< grid, block >>> ();
    //print_block_details <<< grid, block >>> ();
    print_all_details <<< grid, block >>> ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
