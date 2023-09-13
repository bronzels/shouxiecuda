#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#include <stdio.h>

#define ARRAY_SIZE 64
#define FULL_MASK 0xff

__global__ void test_shfl_broadcast_32(int * in, int * out)
{
    int x = in[threadIdx.x];
    int y = __shfl_sync(FULL_MASK, x, 3, 32);
    out[threadIdx.x] = y;
}

__global__ void test_shfl_broadcast_16(int * in, int * out)
{
    int x = in[threadIdx.x];
    int y = __shfl_sync(FULL_MASK, x, 3, 16);
    out[threadIdx.x] = y;
}

__global__ void test_shfl_up(int * in, int * out)
{
    int x = in[threadIdx.x];
    int y = __shfl_up_sync(FULL_MASK, x, 2, 16);
    out[threadIdx.x] = y;
}

__global__ void test_shfl_down(int * in, int * out)
{
    int x = in[threadIdx.x];
    int y = __shfl_down_sync(FULL_MASK, x, 2, 16);
    out[threadIdx.x] = y;
}

__global__ void test_shfl_xor(int * in, int * out)
{
    int x = in[threadIdx.x];
    int y = __shfl_xor_sync(FULL_MASK, x, 1, 16);
    out[threadIdx.x] = y;
}

int main()
{
    int size = ARRAY_SIZE;
    int byte_size = size * sizeof(int);

    int * h_in = (int *)malloc(byte_size);
    int * h_ref = (int *)malloc(byte_size);

    for (int i = 0; i < size; i++)
    {
        h_in[i] = i;
    }


    int * d_in, *d_out;
    cudaMalloc((void **)&d_in, byte_size);
    cudaMalloc((void **)&d_out, byte_size);

    dim3 block(size);
    dim3 grid(1);

    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
    test_shfl_broadcast_32 <<< grid, block >>> (d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

    print_array(h_in, size);
    print_array(h_ref, size);


    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
    test_shfl_broadcast_16 <<< grid, block >>> (d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

    print_array(h_in, size);
    print_array(h_ref, size);


    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
    test_shfl_up <<< grid, block >>> (d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

    print_array(h_in, size);
    print_array(h_ref, size);


    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
    test_shfl_down <<< grid, block >>> (d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

    print_array(h_in, size);
    print_array(h_ref, size);


    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
    test_shfl_xor <<< grid, block >>> (d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

    print_array(h_in, size);
    print_array(h_ref, size);


    free(h_in);
    free(h_ref);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaDeviceReset();
    return 0;
}
