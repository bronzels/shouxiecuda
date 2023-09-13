#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void register_usage_test(int * results, int size)
//__global__ void register_usage_test()
{
    /*
    int x1 = 3465;
    int x2 = 1768;
    int x3 = 453;
    int x4 = x1 + x2 + x3;
    */
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int x1 = 3465;
    int x2 = 1768;
    int x3 = 453;
    int x4 = x1 + x2 + x3;

    if (gid < size)
    {
        results[gid] = x4;
    }
}
/*register
empty      Used 2 registers, 320 bytes cmem[0]
x1-x4      Used 2 registers, 320 bytes cmem[0]
assign     Used 4 registers, 332 bytes cmem[0]
 */

/* nsys profile --stats=true ./master_register_usage.out
 * memcpy               copy            alloc
cudaAlloc               4695331         51445957
cudaHostAlloc           4378928         50799221
 */
int main()
{
    int size = 1 << 22;
    int byte_size = sizeof(int) * size;

    int * h_input;
    h_input = (int *)malloc(byte_size);

    int * d_input;
    //cudaMalloc((void **)&d_input, byte_size);
    cudaMallocHost((void **)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((size+block.x-1)/block.x);
    //register_usage_test <<< grid, block >>> ();
    register_usage_test <<< grid, block >>> (d_input, size);
    cudaMemcpy(h_input, d_input, byte_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    free(h_input);
    //cudaFree(d_input);
    cudaFreeHost(d_input);

    cudaDeviceReset();
    return 0;
}