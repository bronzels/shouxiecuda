#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void unique_idx_calc_threadIdx(int * input)
{
    int tid = threadIdx.x;
    printf("threadIdx : %d, value : %d \n", tid, input[tid]);
}

__global__ void unique_gid_calculation(int * input)
{
    int tid = threadIdx.x;
    int offset = blockDim.x * blockIdx.x;
    int gid = tid + offset;
    printf("blockIdx.x: %d, threadIdx.x : %d, gid: %d, value : %d \n", blockIdx.x, threadIdx.x, gid, input[gid]);
}

__global__ void unique_gid_calculation_g2d(int * input)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int block_offset = blockDim.x * blockIdx.x;
    int row_offset = gridDim.x * blockDim.x * blockIdx.y;
    int gid = row_offset + block_offset + tid;
    printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x : %d, gid: %d, value : %d \n", blockIdx.x, blockIdx.y, threadIdx.x, gid, input[gid]);
}

__global__ void unique_gid_calculation_g2d_b2d(int * input)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int block_offset = (blockDim.x * blockDim.y) * blockIdx.x;
    int row_offset = gridDim.x * (blockDim.x * blockDim.y) * blockIdx.y;
    int gid = row_offset + block_offset + tid;
    printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x : %d, threadIdx.y : %d, gid: %d, value : %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, gid, input[gid]);
}

__global__ void unique_gid_calculation_g3d_b3d(int * input)
{
    int tid = (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    int block_offset =  (blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * blockIdx.y + blockIdx.x);
    int layer_offset = (blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y) * blockIdx.z;
    int gid = layer_offset + block_offset + tid;
    printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d, gid: %d, value : %d \n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, gid, input[gid]);
}

int main()
{
    //int array_size = 16;
    int array_size = 64;
    int array_byte_size = sizeof(int) * array_size;
    //int h_data[] = {23,9,4,53,65,12,1,33,87,45,23,12,342,56,44,99};
    int h_data[] = {23,9,4,53,65,12,1,33,87,45,23,12,342,56,44,99,230,90,40,530,650,120,10,330,870,450,230,120,3420,560,440,990,-23,-9,-4,-53,-65,-12,-1,-33,-87,-45,-23,-12,-342,-56,-44,-99,-230,-90,-40,-530,-650,-120,-10,-330,-870,-450,-230,-120,-3420,-560,-440,-990};

    for (int i = 0; i < array_size; i++)
    {
        printf("%d ", h_data[i]);
    }
    printf("\n \n");

    int * d_data;
    cudaMalloc((void**)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    //dim3 block(4);
    //dim3 grid(4);
    //dim3 block(4);
    //dim3 grid(2, 2);
    dim3 block(2, 2, 2);
    dim3 grid(2, 2, 2);
    //unique_idx_calc_threadIdx <<< grid, block >>> (d_data);
    //unique_gid_calculation <<< grid, block >>> (d_data);
    //unique_gid_calculation_g2d <<< grid, block >>> (d_data);
    //unique_gid_calculation_g2d_b2d <<< grid, block >>> (d_data);
    unique_gid_calculation_g3d_b3d <<< grid, block >>> (d_data);
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}