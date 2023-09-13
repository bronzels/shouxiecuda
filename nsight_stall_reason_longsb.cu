/* ptxas -c master_hello_cuda.ptx
 * ptxas -c master_hello_cuda.ptx -o master_hello_cuda.o
 * nvcc -keep master_hello_cuda.cu
 * .ptx,.cpp1.ii,.cpp4.ii,.cudafe1.c,.cudafe1.cpp,cudafe1.gpu,.cudafe1.stub.c,.fatbin,.fatbin.c,.module_id,.sm_52.cubin
 * cuobjdump master_hello_cuda.sm_52.cubin -sass > master_hello_cuda.sass
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void stall_reason_lsb(int * dramptr)
{
    int tid = threadIdx.x;
    int laneid = tid % 32;
    dramptr[laneid] = laneid;
    __syncthreads();

    int idx = laneid;
#pragma unroll
    for(int i = 0; i < 1000; i ++)
        idx = dramptr[idx];
    dramptr[idx] = idx;
}

__global__ void stall_reason_lg_worst(int8_t * dramptr, int8_t * dramptr2)
{
    int tid = threadIdx.x;
    int offset = tid * 2000;
#pragma unroll
    for(int i = 0; i < 2000; i ++)
        dramptr[offset + i] = dramptr2[offset + i];
}

__global__ void stall_reason_lg_worse(int8_t * dramptr, int8_t * dramptr2)
{
    int tid = threadIdx.x;
    int total_thread = 1024 * 2;
#pragma unroll
    for(int i = 0; i < 2000; i ++)
        dramptr[i * total_thread + tid] = dramptr2[i * total_thread + tid];
}

__global__ void stall_reason_lg_normal(int8_t * dramptr, int8_t * dramptr2)
{
    int tid = threadIdx.x;
    int total_thread = 1024 * 2;
    int * ptr = (int*)dramptr;
    int * ptr2 = (int*)dramptr2;
#pragma unroll
    for(int i = 0; i < 500; i ++)
        ptr[i * total_thread + tid] = ptr2[i * total_thread + tid];
}

__global__ void stall_reason_lg_best(int8_t * dramptr, int8_t * dramptr2)
{
    int tid = threadIdx.x;
    int total_thread = 1024 * 2;
    int4 * ptr = (int4*)dramptr;
    int4 * ptr2 = (int4*)dramptr2;
#pragma unroll
    for(int i = 0; i < 125; i ++)
        ptr[i * total_thread + tid] = ptr2[i * total_thread + tid];
}

int main()
{
    dim3 block(1024);
    dim3 grid(2);

    int8_t *dptr, *dptr2;
    cudaMalloc((void **)&dptr, 2 * 1024 * 2000);
    cudaMalloc((void **)&dptr2, 2 * 1024 * 2000);
    //stall_reason_lsb <<< grid, block >>> (dptr, dptr2);//
    //stall_reason_lg_worst <<< grid, block >>> (dptr, dptr2); //4.65msec
    //stall_reason_lg_worse <<< grid, block >>> (dptr, dptr2); //715.49usec
    //stall_reason_lg_normal <<< grid, block >>> (dptr, dptr2); //188.16usec
    stall_reason_lg_best <<< grid, block >>> (dptr, dptr2); //64.93usec
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
