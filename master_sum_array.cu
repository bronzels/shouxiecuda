#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include "common.hpp"

#include <stdio.h>
#include <ctime>
#include <random>
using namespace std;

__global__ void sum_array_gpu(int *a, int *b, int *c, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
}

__global__ void sum_array_gpu_misaligned(int *a, int *b, int *c, int size, int offset)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int k = gid + offset;
    if(k < size)
    {
        c[gid] = a[k] + b[k];
    }
}

__global__ void sum_array_gpu_misaligned_write(int *a, int *b, int *c, int size, int offset)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int k = gid + offset;
    if(k < size)
    {
        c[k] = a[gid] + b[gid];
    }
}

inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert : %s %s %d \n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

int sync(int argc, char** argv)
{
    cudaError error;

    //int size = 1 << 22;
    int size = 1 << 27;
    int block_size = 128;

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d doesn't support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    int offset = 0;
    if (argc > 1)
        offset = atoi(argv[1]);

    unsigned int NO_BYTES = size * sizeof(int);

    //host pointers
    int *h_a, *h_b, *gpu_results, *h_c;

    //allocate memory for host pointers
    h_a = (int *)malloc(NO_BYTES);
    //cudaHostAlloc(&h_a, NO_BYTES, cudaHostAllocMapped);
    //cudaMallocManaged((void **)&h_a, NO_BYTES);
    h_b = (int *)malloc(NO_BYTES);
    //cudaHostAlloc(&h_b, NO_BYTES, cudaHostAllocMapped);
    //cudaMallocManaged((void **)&h_b, NO_BYTES);
    h_c = (int *)malloc(NO_BYTES);
    gpu_results = (int *)malloc(NO_BYTES);
    //cudaMallocManaged((void **)&gpu_results, NO_BYTES);

    //initialize host pointer
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        h_a[i] = (int)(rand() & 0xFF);
    }
    for (int i = 0; i < size; i++)
    {
        h_b[i] = (int)(rand() & 0xFF);
    }
    clock_t cpu_start, cpu_end;
    //cpu_start = clock();
    sum_arry_cpu(h_a, h_b, h_c, size);
    //cpu_end = clock();

    //device
    int *d_a, *d_b, *d_c;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    float gpu_time = 0.0f;
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    error = cudaMalloc(&d_a, NO_BYTES);
    //error = cudaHostGetDevicePointer((void **)&d_a, (void **)h_a, 0);
    if(error != cudaSuccess)
    {
        fprintf(stderr, " Error : %s \n", cudaGetErrorString(error));
    }
    gpuErrchk(cudaMalloc(&d_b, NO_BYTES));
    //gpuErrchk(cudaHostGetDevicePointer((void **)&d_b, (void **)h_b, 0));
    gpuErrchk(cudaMalloc(&d_c, NO_BYTES));

    gpuErrchk(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));

    //luanching the grid
    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x);
    clock_t gpu_start, gpu_end;
    //gpu_start = clock();

    //sum_array_gpu <<< grid, block>>> (d_a, d_b, d_c, size);
    //sum_array_gpu <<< grid, block>>> (h_a, h_b, gpu_results, size);
    //sum_array_gpu_misaligned <<< grid, block>>> (d_a, d_b, d_c, size, offset);
    sum_array_gpu_misaligned_write <<< grid, block>>> (d_a, d_b, d_c, size, offset);

    //gpu_end = clock();
    gpuErrchk(cudaDeviceSynchronize());

    /*
    printf("Sum array CPU execution time: %4.12f \n",
           (double)((double)(cpu_end - cpu_start)/CLOCKS_PER_SEC));
    printf("Sum array GPU execution time: %4.12f \n",
           (double)((double)(gpu_end - gpu_start)/CLOCKS_PER_SEC));
    */

    gpuErrchk(cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost));

//！！！所有测试
// gpu时间的start在程序最开头，测试stream overlapping是挪到cpu初始化之后，cudaMalloc之前。
// size = 1 << 22
    //only kernel, size = 1000, block = 128 best
    //128, 0.032768
    //256, 0.034496
    //512, 0.044992
    //1024, 0.099328
    //64, 0.033568

    //all, size = 1 << 22, block = 128
    //orig, 121.310081
    //zerocopy, 124.504669
    //unified, 121.955330
    //misaligned 0,
/*
Memory Throughput [%]	91.56
L1/TEX Cache Throughput [%]	24.35
L2 Cache Throughput [%]	45.39
DRAM Throughput [%]	91.56
Memory Throughput [Gbyte/second]	318.20
L1/TEX Hit Rate [%]	0
L2 Hit Rate [%]	33.36
L2 Compression Success Rate [%]	0
Mem Busy [%]	45.39
Max Bandwidth [%]	91.56
Mem Pipes Busy [%]	18.46
L2 Compression Ratio	0
*/
    //misaligned 10,
/*
Memory Throughput [%]	93.35
L1/TEX Cache Throughput [%]	27.09
L2 Cache Throughput [%]	48.18
DRAM Throughput [%]	93.35
Memory Throughput [Gbyte/second]	315.86
L1/TEX Hit Rate [%]	12.62
L2 Hit Rate [%]	35.80
L2 Compression Success Rate [%]	0
Mem Busy [%]	48.18
Max Bandwidth [%]	93.35
Mem Pipes Busy [%]	18.85
L2 Compression Ratio	0
*/
    //misaligned 22,
/*
Memory Throughput [%]	92.96
L1/TEX Cache Throughput [%]	26.91
L2 Cache Throughput [%]	47.92
DRAM Throughput [%]	92.96
Memory Throughput [Gbyte/second]	319.72
Mem Busy [%]	47.92
L1/TEX Hit Rate [%]	11.89
Max Bandwidth [%]	92.96
L2 Hit Rate [%]	35.82
Mem Pipes Busy [%]	18.77
L2 Compression Success Rate [%]	0
L2 Compression Ratio	0
*/
/*L1 disable
nvcc -Xptxas -dlcm-ca -I /usr/local/cuda/samples/cuda-samples/Common -o master_sum_array master_sum_array.cu
*/
    //misaligned 10, L1 disabled, , nvcc -Xptxas -dlcm=ca -I /usr/local/cuda/samples/cuda-samples/Common -o master_sum_array master_sum_array.cu)
/*
Memory Throughput [%]	93.02
L1/TEX Cache Throughput [%]	26.94
L2 Cache Throughput [%]	47.98
DRAM Throughput [%]	93.02
Memory Throughput [Gbyte/second]	319.51
L1/TEX Hit Rate [%]	12.62
L2 Hit Rate [%]	35.83
L2 Compression Success Rate [%]	0
Mem Busy [%]	47.98
Max Bandwidth [%]	93.02
Mem Pipes Busy [%]	18.78
L2 Compression Ratio	0
*/
    //misaligned 22, L1 disabled
/*
Compute (SM) Throughput [%]	18.87
Memory Throughput [%]	93.42
L1/TEX Cache Throughput [%]	27.06
L2 Cache Throughput [%]	48.21
DRAM Throughput [%]	93.42
Memory Throughput [Gbyte/second]	320.21
L1/TEX Hit Rate [%]	11.91
L2 Hit Rate [%]	35.83
L2 Compression Success Rate [%]	0
Mem Busy [%]	48.21
Max Bandwidth [%]	93.42
Mem Pipes Busy [%]	18.87
L2 Compression Ratio	0
*/
    //misaligned write 10(L1 not used)
/*
Memory Throughput [%]	92.65
L1/TEX Cache Throughput [%]	33.87
L2 Cache Throughput [%]	48.41
DRAM Throughput [%]	92.65
Memory Throughput [Gbyte/second]	320.44
L1/TEX Hit Rate [%]	11.62
L2 Hit Rate [%]	36.81
L2 Compression Success Rate [%]	0
Mem Busy [%]	48.41
Max Bandwidth [%]	92.65
Mem Pipes Busy [%]	18.67
L2 Compression Ratio	0
*/
    //misaligned write 22(L1 not used)
/*
Memory Throughput [%]	93.36
L1/TEX Cache Throughput [%]	34.14
L2 Cache Throughput [%]	48.83
DRAM Throughput [%]	93.36
Memory Throughput [Gbyte/second]	320.00
L1/TEX Hit Rate [%]	12.52
L2 Hit Rate [%]	36.75
L2 Compression Success Rate [%]	0
Mem Busy [%]	48.83
Max Bandwidth [%]	93.36
Mem Pipes Busy [%]	18.81
L2 Compression Ratio	0
*/
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter ++;
    }
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("time spent executing by the GPU: %.6f\n", gpu_time);

    compare_arrays(h_c, gpu_results, size);

    free(h_a);
    //cudaFreeHost(h_a);
    free(h_b);
    //cudaFreeHost(h_b);
    free(h_c);
    free(gpu_results);
    gpuErrchk(cudaFree(d_c));


    checkCudaErrors(cudaDeviceReset());
    return 0;
}

int async(int argc, char** argv)
{
    int size = 1 << 25;
    int block_size = 128;

    unsigned int NO_BYTES = size * sizeof(int);

    //host pointers
    int *h_a, *h_b, *gpu_results, *h_c;

    //allocate memory for host pointers
    h_a = (int *)malloc(NO_BYTES);
    h_b = (int *)malloc(NO_BYTES);
    h_c = (int *)malloc(NO_BYTES);
    gpu_results = (int *)malloc(NO_BYTES);

    //initialize host pointer
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        h_a[i] = (int)(rand() & 0xFF);
    }
    for (int i = 0; i < size; i++)
    {
        h_b[i] = (int)(rand() & 0xFF);
    }
    clock_t cpu_start, cpu_end;
    sum_arry_cpu(h_a, h_b, h_c, size);

    //device
    int *d_a, *d_b, *d_c;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    float gpu_time = 0.0f;
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    cudaMalloc(&d_a, NO_BYTES);
    cudaMalloc(&d_b, NO_BYTES);
    cudaMalloc(&d_c, NO_BYTES);

    int const NUM_STREAMS = 8;
    int ELEMENTS_PER_STREAM = size / NUM_STREAMS;
    int BYTES_PER_STREAM = NO_BYTES / NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    int offset = 0;

    //luanching the grid
    dim3 block(block_size);
    dim3 grid((ELEMENTS_PER_STREAM + block.x - 1) / block.x);

    for (int i = 0; i < NUM_STREAMS; i++)
    {
        offset = i * ELEMENTS_PER_STREAM;
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
        sum_array_gpu <<< grid, block, 0, streams[i]>>> (&d_a[offset], &d_b[offset], &d_c[offset], ELEMENTS_PER_STREAM);
        cudaMemcpyAsync(&gpu_results[offset], &d_c[offset], BYTES_PER_STREAM, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter ++;
    }
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("time spent executing by the GPU: %.6f\n", gpu_time);

    cudaDeviceSynchronize();
    time2 = clock();
    printf("time spent executing openCL sum_array:%f s\n", (double)(time2-time1)/CLOCKS_PER_SEC);

    compare_arrays(h_c, gpu_results, size);

    free(h_a);
    free(h_b);
    free(h_c);
    free(gpu_results);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    checkCudaErrors(cudaDeviceReset());
    return 0;
}

int main(int argc, char** argv)
{
    printf("------------sync------------");
    sync(argc, argv);
    printf("------------async------------");
    async(argc, argv);
}
/*
time spent executing openCL(only kernel) sum_array:1.209376 ms
time spent executing openCL sum_array:0.050000 s
    int size = 1 << 25;
    int const NUM_STREAMS = 8;
------------sync------------time spent executing by the GPU: 51.621441
Arrays are same
------------async------------time spent executing by the GPU: 51.985504
Arrays are same

time spent executing openCL(only kernel) sum_array:4.796416 ms
time spent executing openCL sum_array:0.200000 s
     int size = 1 << 27;
    int const NUM_STREAMS = 8;
------------sync------------time spent executing by the GPU: 204.370270
Arrays are same
------------async------------time spent executing by the GPU: 202.696701
Arrays are same

     int size = 1 << 27;
    int const NUM_STREAMS = 16;
------------sync------------time spent executing by the GPU: 200.677765
Arrays are same
------------async------------time spent executing by the GPU: 205.065094
Arrays are same

     int size = 1 << 27;
    int const NUM_STREAMS = 4;
------------sync------------time spent executing by the GPU: 202.320312
Arrays are same
------------async------------time spent executing by the GPU: 203.188385
Arrays are same
 */