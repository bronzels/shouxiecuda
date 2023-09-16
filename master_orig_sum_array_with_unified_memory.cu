#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.hpp"
#include "cuda_common.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void test_unified_memory(int* a, int* b, int *c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
		c[gid] = a[gid] + b[gid];
}

void sum_array_cpu2(int *a, int *b, int *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void compare_arrays2(int * a, int * b, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            printf("Arrays are different on i:%d, a[i]:%d, b[i]:%d \n", i, a[i], b[i]);
            return;
        }
    }
    printf("Arrays are same \n");
}

int main(int argc, char** argv)
{
	printf("Runing 1D grid \n");
	int size = 1 << 22;
	int block_size = 128;

	if (argc > 1)
		block_size = 1 << atoi(argv[1]);

	printf("Entered block size : %d \n", block_size);

	unsigned int byte_size = size * sizeof(float);

	printf("Input size : %d \n", size);

    int * A, *B, *ref, *C;

	cudaMallocManaged((void **)&A, byte_size);
	cudaMallocManaged((void **)&B, byte_size);
	cudaMallocManaged((void **)&ref, byte_size);

	C = (int*)malloc(byte_size);

	if (!A)
		printf("host memory allocation error \n");

	for (size_t i = 0; i < size; i++)
	{
		A[i] = i % 10;
		B[i] = i % 7;
	}

	sum_array_cpu2(A, B, C, size);

	dim3 block(block_size);
	dim3 grid((size + block.x - 1) / block.x);

	printf("Kernel is lauch with grid(%d,%d,%d) and block(%d,%d,%d) \n",
		grid.x, grid.y, grid.z, block.x, block.y, block.z);

	test_unified_memory <<<grid, block >>> (A, B, ref, size);
	gpuErrchk(cudaDeviceSynchronize());

	compare_arrays2(ref, C, size);
	free(C);

	return 0;
}
