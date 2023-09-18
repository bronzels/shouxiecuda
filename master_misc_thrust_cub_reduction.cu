#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/cub.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "common.hpp"
#include "cuda_common.cuh"

#include "helper_cuda.h"
#include "helper_functions.h"

#include <ctime>
#include <random>
#include <time.h>
using namespace std;

#define BDIM 512
#define TDIM 32
#define FULL_MASK 0xffffffff

/*
10.19 msec

Active Warps Per Scheduler [warp]	11.25
Eligible Warps Per Scheduler [warp]	2.31
Issued Warp Per Scheduler	0.77

Achieved Occupancy [%]	94.21
Achieved Active Warps Per SM [warp]	45.22
*/
__global__ void redunction_powed_pairs(int * input,
                                       int * temp, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid > size)
        return;

    for (int offset = 1; offset <= blockDim.x/2; offset *= 2)
    {
        if ( tid % ( 2 * offset) == 0)
        {
            input[gid] += input[gid + offset];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = input[gid];
    }
}

/*
6.05 msec

Active Warps Per Scheduler [warp]	10.66
Eligible Warps Per Scheduler [warp]	0.82
Issued Warp Per Scheduler	0.53

Achieved Occupancy [%]	89.12
Achieved Active Warps Per SM [warp]	42.78
*/
__global__ void redunction_neighbored_pairs(int * input,
                                            int * temp, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int * i_data = input + blockDim.x * blockIdx.x;

    if(gid > size)
        return;

    for (int offset = 1; offset <= blockDim.x/2; offset *= 2)
    {
        int index = 2 * offset * tid;
        if ( index < blockDim.x)
        {
            i_data[index] += i_data[index + offset];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = input[gid];
    }
}

/*
6.05 msec

Active Warps Per Scheduler [warp]	10.65
Eligible Warps Per Scheduler [warp]	0.82
Issued Warp Per Scheduler	0.53

Achieved Occupancy [%]	89.47
Achieved Active Warps Per SM [warp]	42.95
*/
__global__ void redunction_interleaved_pairs(int * input,
                                            int * temp, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid > size)
        return;

    for (int offset = blockDim.x/2; offset > 0; offset = offset / 2)
    {
        if ( tid < offset)
        {
            input[gid] += input[gid + offset];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = input[gid];
    }
}

/*
hardcode
2, 2.76 msec
4, 2.00 msec
loop
2, 2.83 msec
4, 2.19 msec
8, 1.94 msec
16, 1.82 msec
*/
__global__ void redunction_loop_unrolling_pairs(int * input,
                                                int * temp, int size, int rollingfactor)
{
    int tid = threadIdx.x;
    int BLOCK_OFFSET = blockDim.x * blockIdx.x * rollingfactor;

    int index = BLOCK_OFFSET + tid;

    int * i_data = input + BLOCK_OFFSET;

    if ( (index + blockDim.x * (rollingfactor - 1)) < size)
    {
        if(rollingfactor == 2)
        {
            input[index] += input[index + blockDim.x];
        } else if(rollingfactor == 4)
        {
            //int a1 = input[index];
            int a2 = input[index + blockDim.x];
            int a3 = input[index + 2 * blockDim.x];
            int a4 = input[index + 3 * blockDim.x];
            input[index] += a2 + a3 + a4;
        }
        /*
        for(int j=0;j<rollingfactor-1;j++)
            input[index] += input[index + blockDim.x * (j+1)];
        */
    }

    __syncthreads();

    for (int offset = blockDim.x/2; offset > 0; offset = offset / 2) //offset /= 2 error
    {
        if ( tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

/*
7.04 msec
*/
__global__ void redunction_warp_unrolling_pairs(int * input,
                                                int * temp, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid > size)
        return;

    int * i_data = input + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x/2; offset >= 64; offset = offset / 2)
    {
        if ( tid < offset)
        {
            input[gid] += input[gid + offset];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        volatile int * vsmem = i_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

/*
7.23 msec
template
7.19
*/
template<unsigned int iblock_size>
__global__ void redunction_complete_unrolling_pairs_template(int * input,
                                                             int * temp, int size)
//__global__ void redunction_complete_unrolling_pairs(int * input,
//                                                int * temp, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid > size)
        return;

    int * i_data = input + blockDim.x * blockIdx.x;

    //if (blockDim.x == 1024 && tid < 512)
    if (iblock_size >= 1024 && tid < 512)
        i_data[tid] += i_data[tid + 512];
    __syncthreads();

    //if (blockDim.x == 512 && tid < 256)
    if (iblock_size >= 512 && tid < 256)
        i_data[tid] += i_data[tid + 256];
    __syncthreads();

    //if (blockDim.x == 256 && tid < 128)
    if (iblock_size >= 256 && tid < 128)
        i_data[tid] += i_data[tid + 128];
    __syncthreads();

    //if (blockDim.x == 128 && tid < 64)
    if (iblock_size >= 128 && tid < 64)
        i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (tid < 32)
    {
        volatile int * vsmem = i_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

template<unsigned int iblock_size>
__global__ void redunction_complete_unrolling_pairs_smem_template(int * input,
                                                             int * temp, int size)
{
    extern __shared__ int smem[];

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid > size)
        return;

    int * i_data = input + blockDim.x * blockIdx.x;
    smem[tid] = i_data[tid];

    __syncthreads();

    i_data = smem;
    //if (blockDim.x == 1024 && tid < 512)
    if (iblock_size >= 1024 && tid < 512)
        i_data[tid] += i_data[tid + 512];
    __syncthreads();

    //if (blockDim.x == 512 && tid < 256)
    if (iblock_size >= 512 && tid < 256)
        i_data[tid] += i_data[tid + 256];
    __syncthreads();

    //if (blockDim.x == 256 && tid < 128)
    if (iblock_size >= 256 && tid < 128)
        i_data[tid] += i_data[tid + 128];
    __syncthreads();

    //if (blockDim.x == 128 && tid < 64)
    if (iblock_size >= 128 && tid < 64)
        i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (tid < 32)
    {
        //volatile int * vsmem = i_data;
        volatile int * vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = i_data[0];
    }
}

template<unsigned int iblock_size>
__global__ void redunction_complete_unrolling_pairs_smem_warpshfl_template(int * input,
                                                                  int * temp, int size)
{
    extern __shared__ int smem[];

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid > size)
        return;

    int * i_data = input + blockDim.x * blockIdx.x;
    smem[tid] = i_data[tid];

    __syncthreads();

    i_data = smem;
    //if (blockDim.x == 1024 && tid < 512)
    if (iblock_size >= 1024 && tid < 512)
        i_data[tid] += i_data[tid + 512];
    __syncthreads();

    //if (blockDim.x == 512 && tid < 256)
    if (iblock_size >= 512 && tid < 256)
        i_data[tid] += i_data[tid + 256];
    __syncthreads();

    //if (blockDim.x == 256 && tid < 128)
    if (iblock_size >= 256 && tid < 128)
        i_data[tid] += i_data[tid + 128];
    __syncthreads();

    //if (blockDim.x == 128 && tid < 64)
    if (iblock_size >= 128 && tid < 64)
        i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (iblock_size >= 64 && tid < 32)
        i_data[tid] += i_data[tid + 32];
    __syncthreads();

    int local_sum = smem[tid];
    if (tid < 32)
    {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, 16);
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, 8);
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, 4);
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, 2);
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, 1);
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = local_sum;
    }
}

/*
128
                        time spent executing by the GPU
0                       23
0                       7.987648
1                       4.950496
2                       3.938240
3.2                     2.494016
3.4                     2.003584
4                       5.452000
5                       5.559872
6(only in below 64)     2.908512
6                       2.856544/2.859680/2.861440
7                       2.955232/2.958560/2.955328
8(atomic)               0.092768    (1<<20, 3.2: 0.147712)
8(+h_ref cudamemcpy)    2.322592
thrust                  0.6         (1<<20, 3.2: 0.147712)
thrust(only kernel)     0.01         (1<<20, 3.2: 0.147712)
cub                     0.59        (1<<20, 3.2: 0.147712)
cub(only kernel)        0.01        (1<<20, 3.2: 0.147712)


BDIM:512, TDIM:32
8(atomic)                                                                               0.056832    (1<<20, 3.2: 0.147712)
9(cub replace atomic，最后atomic会造成函数不能进入，只能退回cpu侧per block相加)                 0.333056    (1<<20, 3.2: 0.147712)
10(cub load replace forloop)                                                            总是0        (1<<20, 3.2: 0.147712)

    same wt/wn 2 __syncthreads
    same to include h_ref cudamemcpy or not

1024
                        time spent executing by the GPU
3.4                     2.730336
6                       6.200384

64
                        time spent executing by the GPU
3.4                     2.010336
6                       9.823040

6 baseline
128(base)   3.53
1024        7.73 usec,  shared load, wavefronts 1024(+3100.00%), % Peak 0.98(+3070.24%), bank conflicts 992(+inf%)
                        shared store, wavefronts 32(-96.88%), % Peak 0.01(-96.90%), bank conflicts 0(-100.00%)
avefronts 1024(+3100.00%), % Peak 0.98(+3070.24%), bank conflicts 992(+inf%)
                        shared store, wavefronts 32(-96.88%), % Peak 0.01(-96.90%), bank conflicts 0(-100.00%)
*/

__global__ void redunction_atomic_add(int * input,
                                      int * temp, int size)
{
    int pos_start = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_step = blockDim.x * gridDim.x;
    int thread_sum = 0;
    for(int i = pos_start; i < size; i = i + pos_step)
    {
        thread_sum += input[i];
    }
    __syncthreads();
    __shared__ int block_sum;
    block_sum = 0;
    atomicAdd(&block_sum, thread_sum);
    //printf("pos_start:%d, block_sum:%d\n", pos_start, block_sum);
    if(threadIdx.x == 0) {
        atomicAdd(temp, block_sum);
    }
}

__global__ void redunction_atomic_add_cub_for(int * input,
                                      int * temp, int size)
{
    int pos_start = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_step = blockDim.x * gridDim.x;
    int thread_sum = 0;
    for(int i = pos_start; i < size; i = i + pos_step)
    {
        thread_sum += input[i];
    }
    __syncthreads();
    //printf("pos_start:%d, thread_sum:%d\n", pos_start, thread_sum);
    typedef cub::BlockReduce<int, BDIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    //int block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    temp[blockIdx.x] = BlockReduce(temp_storage).Sum(thread_sum);
    /*
    //printf("pos_start:%d, block_sum:%d\n", pos_start, block_sum);

    if(threadIdx.x == 0) {
        //atomicAdd(temp, block_sum);
    }
    */
}
/*
int block_sum = BlockReduce(temp_storage).Sum(thread_sum);这句后面，加这3种处理，都会导致函数完全不会被调用：
1，atomicAdd
2，__syncthreads
3, printf打印
也就是用cub情况下，不能一次把求和做完，只能把按block求和的结果，copy回cpu，在cpu测加完。
*/

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, cub::BlockReduceAlgorithm ALGORITHM>
__global__ void redunction_atomic_add_cub_load(int * input,
                                      int * temp)
{
    printf("blockIdx.x:%d, threadIdx.x:%d\n", blockIdx.x, threadIdx.x);
    //int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    typedef cub::BlockReduce<int, BLOCK_THREADS, ALGORITHM> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    int data[ITEMS_PER_THREAD];
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, input, data);
    //int block_sum = BlockReduceT(temp_storage).Sum(data);
    temp[blockIdx.x] = BlockReduceT(temp_storage).Sum(data);

    /*
    if(threadIdx.x == 0) {
        atomicAdd(temp, block_sum);
    }
    */
}

void thrust_reduction(int *h_input, int size, int result)
{
    clock_t time1,time2;
    int * dptr_in;
    checkCudaErrors(cudaMalloc((void **)&dptr_in, size * sizeof(int)));
    cudaMemcpy(dptr_in, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    thrust::device_ptr<int> dptr_thr_in = thrust::device_pointer_cast(dptr_in);
    time1 = clock();
    int thrust_value = thrust::reduce(dptr_thr_in, dptr_thr_in + size, (int)0, thrust::plus<int>());
    time2 = clock();
    std::cout<<"time spent executing by the thrust_reduction: "<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    compare_results(thrust_value, result);

}

void cub_reduction(int *h_input, int size, int result)
{
    clock_t time1,time2;
    int * d_input;
    int byte_size = size * sizeof(int);
    checkCudaErrors(cudaMalloc((void **)&d_input, byte_size));
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
    int * d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, byte_size));
    void *dev_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    time1 = clock();
    cub::DeviceReduce::Sum(dev_temp_storage, temp_storage_bytes, d_input, d_output, size);
    //cub::DeviceReduce::Reduce(dev_temp_storage, temp_storage_bytes, d_input, d_input, size, cub::Sum(), 0);
    checkCudaErrors(cudaMalloc((void **)&dev_temp_storage, temp_storage_bytes));
    cub::DeviceReduce::Sum(dev_temp_storage, temp_storage_bytes, d_input, d_output, size);
    //cub::DeviceReduce::Reduce(dev_temp_storage, temp_storage_bytes, d_input, d_input, size, cub::Sum(), 0);
    cudaDeviceSynchronize();
    time2 = clock();
    cudaMemcpy(h_input, d_output, byte_size, cudaMemcpyDeviceToHost);
    int cub_value = h_input[0];
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(dev_temp_storage);
    std::cout<<"time spent executing by the cub_reduction: "<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    compare_results(cub_value, result);

}

int main(int argc, char** argv)
{
    printf("Runing neighbored pairs reduction kernel \n");

    /*
     * 0-default
     * 1-neighbored
     * 2-interleaved
     * 3-loopunrolling
     * 4-warpunrolling
     * 5-compleserolling
     * 6-compleserolling_smem
     * 7-compleserolling_smem_warpshfl
     * 8-atomic_add_wn_cpureduc
     * 9-atomic_add_wn_cpureduc_cub_for
     * 10-atomic_add_wn_cpureduc_cub_load
    */
    int method = 0;
    if (argc > 1)
        method = atoi(argv[1]);
    method = 10;
    printf("method:%d\n", method);
    int block_size = BDIM;
    /*
    if (argc > 2)
        block_size = atoi(argv[2]);
    */
    printf("block_size:%d\n", block_size);
    int rollingfactor = 2;
    if (argc > 3 && method == 3)
        rollingfactor = atoi(argv[3]);

    //int size = 1 << 29;
    int size = 1 << 20;
    printf("size:%d\n", size);
    int byte_size = size * sizeof(int);

    int gridsize = 0;
    int items_per_thread = 1;
    if(method == 3)
        gridsize = (size/block_size)/rollingfactor;
    else if(method == 8 || method == 9) {
        items_per_thread = TDIM;
        gridsize = size/block_size/items_per_thread;
    }
    else
        gridsize = size/block_size;

    int * h_input, *h_ref;
    h_input = (int*)malloc(byte_size);
    if(h_input == 0)
    {
        printf("host malloc failed, exit");
        return EXIT_FAILURE;
    }

    initialize(h_input, size, INIT_RANDOM);
    //initialize(h_input, size, INIT_ONE);

    int cpu_result = reduction_cpu(h_input, size);

    //thrust_reduction(h_input, size, cpu_result);
    //cub_reduction(h_input, size, cpu_result);
    //return 0;

    dim3 block(block_size);
    dim3 grid(gridsize);

    printf("Kernel launch parameters | grid.x : %d, block.x : %d\n", grid.x, block.x);

    int temp_array_byte_size = sizeof(int) * grid.x;
    h_ref = (int*)malloc(temp_array_byte_size);

    int *d_input, *d_temp;

    gpuErrchk(cudaMalloc(&d_input, byte_size));
    gpuErrchk(cudaMalloc(&d_temp, temp_array_byte_size));

    gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size))
    checkCudaErrors(cudaMemcpy(d_input, h_input, byte_size,
                         cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    float gpu_time = 0.0f;
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    char * kernelname;
    switch(method) {
        case 1:
            kernelname="redunction_neighbored_pairs\n";
            redunction_neighbored_pairs <<< grid, block >>>(d_input, d_temp, size);
            break;
        case 2:
            kernelname="redunction_interleaved_pairs\n";
            redunction_interleaved_pairs <<< grid, block >>>(d_input, d_temp, size);
            break;
        case 3:
            kernelname="redunction_loop_unrolling_pairs\n";
            //printf("rollingfactor : %d\n", rollingfactor);
            //reduction_unrolling_blocks2 <<< grid, block >>>(d_input, d_temp, size);
            redunction_loop_unrolling_pairs <<< grid, block >>>(d_input, d_temp, size, rollingfactor);
            break;
        case 4:
            kernelname="redunction_warp_unrolling_pairs\n";
            redunction_warp_unrolling_pairs <<< grid, block >>>(d_input, d_temp, size);
            break;
        case 5:
            kernelname="redunction_complete_unrolling_pairs\n";
            //redunction_complete_unrolling_pairs <<< grid, block >>>(d_input, d_temp, size);
            switch ( block_size)
            {
                case 1024:
                    redunction_complete_unrolling_pairs_template <1024> <<< grid, block >>>(d_input, d_temp, size);
                    break;
                case 512:
                    redunction_complete_unrolling_pairs_template <512> <<< grid, block >>>(d_input, d_temp, size);
                    break;
                case 256:
                    redunction_complete_unrolling_pairs_template <256> <<< grid, block >>>(d_input, d_temp, size);
                    break;
                case 128:
                    redunction_complete_unrolling_pairs_template <128> <<< grid, block >>>(d_input, d_temp, size);
                    break;
                case 64:
                    redunction_complete_unrolling_pairs_template <64> <<< grid, block >>>(d_input, d_temp, size);
                    break;
            }
            break;
        case 6:
            kernelname="redunction_complete_unrolling_pairs_smem\n";
            switch ( block_size)
            {
                case 1024:
                    redunction_complete_unrolling_pairs_smem_template <1024> <<< grid, block, 1024 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
                case 512:
                    redunction_complete_unrolling_pairs_smem_template <512> <<< grid, block, 512 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
                case 256:
                    redunction_complete_unrolling_pairs_smem_template <256> <<< grid, block, 256 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
                case 128:
                    redunction_complete_unrolling_pairs_smem_template <128> <<< grid, block, 128 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
                case 64:
                    redunction_complete_unrolling_pairs_template <64> <<< grid, block, 64 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
            }
            break;
        case 7:
            kernelname="redunction_complete_unrolling_pairs_smem_warpshfl_template\n";
            switch ( block_size)
            {
                case 1024:
                    redunction_complete_unrolling_pairs_smem_warpshfl_template <1024> <<< grid, block, 1024 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
                case 512:
                    redunction_complete_unrolling_pairs_smem_warpshfl_template <512> <<< grid, block, 512 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
                case 256:
                    redunction_complete_unrolling_pairs_smem_warpshfl_template <256> <<< grid, block, 256 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
                case 128:
                    redunction_complete_unrolling_pairs_smem_warpshfl_template <128> <<< grid, block, 128 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
                case 64:
                    redunction_complete_unrolling_pairs_smem_warpshfl_template <64> <<< grid, block, 64 *sizeof(int) >>>(d_input, d_temp, size);
                    break;
            }
            break;
        case 8:
            kernelname="redunction_atomic_add\n";
            redunction_atomic_add <<< grid, block >>>(d_input, d_temp, size);
            break;
        case 9:
            kernelname="redunction_atomic_add_cub_for\n";
            redunction_atomic_add_cub_for <<< grid, block >>>(d_input, d_temp, size);
            break;
        case 10:
            kernelname="redunction_atomic_add_cub_load\n";
            redunction_atomic_add_cub_load<BDIM, TDIM, cub::BLOCK_REDUCE_RAKING> <<< grid, block >>>(d_input, d_temp);
            break;
        default:
            kernelname="redunction_powed_pairs\n";
            redunction_powed_pairs <<< grid, block >>>(d_input, d_temp, size);
            break;
    }
    //cudaDeviceSynchronize();
    printf(kernelname);
    //reduction_kernel_complete_unrolling <<< grid, block >>>(d_input, d_temp, size);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter ++;
    }
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("time spent executing by the GPU: %.6f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.6f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    //if (method != 8 && method != 9 && method != 10)
    if (method != 8)
    {
        checkCudaErrors(cudaMemcpy(h_ref, d_temp, temp_array_byte_size,
                                   cudaMemcpyDeviceToHost));
    }
    else
    {
        checkCudaErrors(cudaMemcpy(h_ref, d_temp, sizeof(int),
                                   cudaMemcpyDeviceToHost));
    }

    int gpu_result = 0;
    //if (method != 8 && method != 9 && method != 10)
    if (method != 8)
    {
        for ( int i = 0; i < grid.x; i ++)
        {
            gpu_result += h_ref[i];
        }
    }
    else
        gpu_result = h_ref[0];

    compare_results(gpu_result, cpu_result);

    free(h_input);
    free(h_ref);
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_temp));

    cudaDeviceReset();
    return 0;
}