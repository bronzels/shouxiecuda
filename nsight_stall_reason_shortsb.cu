#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void stall_reason_lsb(int * dramptr)
{
    __shared__ int smm[32];
    int tid = threadIdx.x;
    int laneid = tid % 32;
    smm[laneid] = laneid;
    dramptr[laneid] = laneid;
    __syncthreads();

    int idx = laneid;
#pragma unroll
    for(int i = 0; i < 1000; i ++)
        idx = smm[idx];
    dramptr[idx] = idx;
}

__global__ void stall_reason_mio_bad(int * dramptr)
{
    __shared__ int smm[32][32];
    __shared__ int smm2[32][32];
    int tid = threadIdx.x;
    int laneid = tid % 32;
#pragma unroll
    for(int i = 0; i < 32; i ++)
        smm2[laneid][i] = smm[laneid][i];

    __syncthreads();
}

__global__ void stall_reason_mio_good(int * dramptr)
{
    __shared__ int smm[32][32];
    __shared__ int smm2[32][32];
    int tid = threadIdx.x;
    int laneid = tid % 32;
#pragma unroll
    for(int i = 0; i < 32; i ++)
        smm2[i][laneid] = smm[i][laneid];

    __syncthreads();
}

int main()
{
    dim3 block(1024);
    dim3 grid(40);

    int *dptr;
    cudaMalloc((void **)&dptr, 32 * 4);
    //stall_reason_lsb <<< grid, block >>> (dptr); //99.87usec, stall short scoreboard-50, stall MIO throttle-7.5, theoretical/active warps per scheduler 8.0
    //stall_reason_mio_bad <<< grid, block >>> (dptr); //1.86usec, stall short scoreboard-0, stall MIO throttle-0.5, theoretical(8.0)/active(7.0) warps per scheduler, stall IMC Miss-37, stall Barrier-15, Stall Branch Resolving-5
    stall_reason_mio_good <<< grid, block >>> (dptr); //1.86usec, stall short scoreboard-0, stall MIO throttle-0.5, theoretical(8.0)/active(7.0) warps per scheduler, stall IMC Miss-37, stall Barrier-15, Stall Branch Resolving-5
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
