#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include "common.hpp"

#include <stdio.h>
#include <ctime>
#include <random>
using namespace std;

template <typename T>
__global__ void sum_array_gpu_1d(T *a, T *b, T *c, size_t size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
}

template <typename T>
__global__ void sum_array_gpu_2d(T *a, T *b, T *c, size_t height, size_t width, size_t pitch_a, size_t pitch_b, size_t pitch_c)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gid / width;
    int col = gid % width;
    if(gid >= height * width)
        return;
    //for( int row = 0; row < height; row++) {
        T *row_a_m = (T*)((char *)a + row*pitch_a);
        T *row_b_m = (T*)((char *)b + row*pitch_b);
        T *row_c_m = (T*)((char *)c + row*pitch_c);
        //for( int j = width; j < width; j++) {
            row_c_m[col] = row_a_m[col] + row_b_m[col];
        //}
    //}
}

template <typename T>
__global__ void sum_array_gpu_3d(cudaPitchedPtr a, cudaPitchedPtr b, cudaPitchedPtr c, cudaExtent extent)
{
    size_t width = extent.width / sizeof(T);

    float *ptr_a = (T*)a.ptr;
    float *ptr_b = (T*)b.ptr;
    float *ptr_c = (T*)c.ptr;

    float *slicehead_a, *slicehead_b, *slicehead_c;
    float *rowhead_a, *rowhead_b, *rowhead_c;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= width * extent.depth * extent.height)
        return;

    int z = gid / (width * extent.height);
    int y = (gid % (width * extent.height)) / width;
    int x = (gid % (width * extent.height)) % width;

    //for( int z = 0; z < extent.depth; z++) {
        slicehead_a = (T*)((char*)ptr_a + z * a.pitch * extent.height);
        slicehead_b = (T*)((char*)ptr_b + z * b.pitch * extent.height);
        slicehead_c = (T*)((char*)ptr_c + z * c.pitch * extent.height);
        //for (int y = 0; y < extent.height; y++) {
            rowhead_a = (T*)((char*)slicehead_a + y * a.pitch);
            rowhead_b = (T*)((char*)slicehead_b + y * a.pitch);
            rowhead_c = (T*)((char*)slicehead_c + y * a.pitch);
            //for (int x = 0; x < width; x++) {
                rowhead_c[x] = rowhead_b[x] + rowhead_b[x];
            //}
        //}
    //}
}

#define BDIM 128

template <typename T>
void gpu_1d(T *h_a, T *h_b, T *h_c, const char * kernel_name, size_t w, size_t h, size_t d)
{
    size_t size = w * h * d;
    T *d_a, *d_b, *d_c;

    dim3 block(BDIM);
    dim3 grid((size + block.x - 1) / block.x);

    size_t NO_BYTES = size * sizeof(T);
    checkCudaErrors(cudaMalloc(&d_a, NO_BYTES));
    checkCudaErrors(cudaMalloc(&d_b, NO_BYTES));
    checkCudaErrors(cudaMalloc(&d_c, NO_BYTES));

    checkCudaErrors(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    float time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    sum_array_gpu_1d <<< grid, block>>> (d_a, d_b, d_c, size);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s %s time: %f\n", kernel_name, typeid(T).name(), time/1000);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_c, d_c, NO_BYTES, cudaMemcpyDeviceToHost));

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
}

template <typename T>
void gpu_2d(T *h_a, T *h_b, T *h_c, const char * kernel_name, size_t w, size_t h, size_t d)
{
    size_t size = w * h * d;
    T *d_a, *d_b, *d_c;

    dim3 block(BDIM);
    dim3 grid((size + block.x - 1) / block.x);
    size_t width = w;
    size_t height = size / width;

    size_t pitch_a, pitch_b, pitch_c;
    size_t byte_width = width*sizeof(T);
    checkCudaErrors(cudaMallocPitch(&d_a, &pitch_a, byte_width, height));
    checkCudaErrors(cudaMallocPitch(&d_b, &pitch_b, byte_width, height));
    checkCudaErrors(cudaMallocPitch(&d_c, &pitch_c, byte_width, height));

    checkCudaErrors(cudaMemcpy2D(d_a, byte_width, h_a, pitch_a, byte_width, height, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_b, byte_width, h_b, pitch_b, byte_width, height, cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    float time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    sum_array_gpu_2d <<< grid, block>>> (d_a, d_b, d_c, height, width, pitch_a, pitch_b, pitch_c);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s %s time: %f\n", kernel_name, typeid(T).name(), time/1000);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy2D(h_c, byte_width, d_c, pitch_c, byte_width, height, cudaMemcpyDeviceToHost));

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
}

template <typename T>
void gpu_3d(T *h_a, T *h_b, T *h_c, const char * kernel_name, size_t w, size_t h, size_t d)
{
    size_t size = w * h * d;
    dim3 block(BDIM);
    dim3 grid((size + block.x - 1) / block.x);
    size_t width = w;
    size_t height = h;
    size_t depth = d;

    cudaPitchedPtr          d_a, d_b, d_c;
    cudaExtent              extent;
    cudaMemcpy3DParms       cpyparm_a, cpyparm_b, cpyparm_c;

    size_t byte_width = width*sizeof(T);
    make_cudaExtent(byte_width, height, depth);
    /*
w- Width in elements when referring to array memory, in bytes when referring to linear memory
h- Height in elements
d- Depth in elements
     */
    cudaMalloc3D(&d_a, extent);
    cudaMalloc3D(&d_b, extent);
    cudaMalloc3D(&d_c, extent);

    cpyparm_a = {0};
    cpyparm_a.srcPtr = make_cudaPitchedPtr((void*)h_a, byte_width, width, height);
    cpyparm_a.dstPtr = d_a;
    cpyparm_a.extent = extent;
    cpyparm_a.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyparm_a);
    cpyparm_b = {0};
    cpyparm_b.srcPtr = make_cudaPitchedPtr((void*)h_b, byte_width, width, height);
    cpyparm_b.dstPtr = d_b;
    cpyparm_b.extent = extent;
    cpyparm_b.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cpyparm_b);
    

    cudaEvent_t start, end;
    float time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    sum_array_gpu_3d<float> <<< grid, block>>> (d_a, d_b, d_c, extent);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s %s time: %f\n", kernel_name, typeid(T).name(), time/1000);
    checkCudaErrors(cudaDeviceSynchronize());
    cpyparm_c = {0};
    cpyparm_c.srcPtr = d_c;
    cpyparm_c.dstPtr = make_cudaPitchedPtr((void*)h_c, byte_width, width, height);
    cpyparm_c.extent = extent;
    cpyparm_c.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&cpyparm_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
}

#define W 1920*16
#define H 1080*16
#define D 3*2

template <typename T>
int exec(int argc, char** argv) {
    size_t w = W;
    size_t h = H;
    size_t d = D;
    size_t size = w * h * d;
    //size_t size = 1 << 27;

    bool print_a = true;

    size_t NO_BYTES = size * sizeof(T);
    size_t total_in_sizeG = (NO_BYTES * 4) >> 30;
    //size_t转double运行时illegal instruction
    printf("Sum array dimension:(%d X %d X %d), size:%zu, bytes:%zu, bytes total:%zu G\n",w,h,d,size,NO_BYTES,total_in_sizeG);

    //host pointers
    T *h_a, *h_b, *h_cpu_results, *h_c;

/*                      cpu            avx2          avx2(32align=64)    avx512          avx512(128align)    openmp(for)    avx2+openmp(for)                                                      openmp(parallel+for)    avx2+openmp(paralle+for)  avx2+openmp(paralle+for+schedule)
*16*16*2/47G,int        4.28(4.25)     2.70(2.70)    2.72                                                    3.93(3.93)      2.72(2.69)                                                           23.860000               23.87                     64.28
                                                                                                             nowait,2.68(2.700000)
向量加复杂化(int)         15.40                                                                                 parallel+for=15.34,for=39.510000
*16*16*2/47G,float      4.31(4.16)     2.67          2.76                                                                                                                                         23.89(3.93)             23.92(23.86)
向量加复杂化(float)        4.31                                                                                 parallel+for=40.360000,for=14.390000
*/
    //allocate memory for host pointers
    /*
    h_a = (T *)aligned_alloc(64, NO_BYTES);
    h_b = (T *)aligned_alloc(64, NO_BYTES);
    h_c = (T *)aligned_alloc(64, NO_BYTES);
    h_cpu_results = (T *)aligned_alloc(64, NO_BYTES);
    */
    h_a = (T *)malloc(NO_BYTES);
    h_b = (T *)malloc(NO_BYTES);
    h_c = (T *)malloc(NO_BYTES);
    h_cpu_results = (T *)malloc(NO_BYTES);
    if(h_a == NULL || h_b == NULL || h_c == NULL || h_cpu_results == NULL) {
        printf("malloc failed, exit\n");
        exit(1);
    }

    //initialize host pointer
    initialize(h_a, size, INIT_RANDOM);
    initialize(h_b, size, INIT_RANDOM);

    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_cpu_results, size);
    cpu_end = clock();
    printf("%s %s time: %f\n", "CPU not optimized", typeid(T).name(), (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
    if ( print_a)
    {
        print_matrix(h_cpu_results, 1, 10);
        print_matrix(h_cpu_results + size / 2 - 10, 1, 10);
        print_matrix(h_cpu_results + size / 2 + 10, 1, 10);
        print_matrix(h_cpu_results + size - 10, 1, 10);
    }

    int kernel_num;
    char *kernel_name;
    void(*kernel)(T *h_a, T *h_b, T *h_c, const char * kernel_name, size_t w, size_t h, size_t d);
    kernel_num = 0;
    //for (kernel_num = 0; kernel_num < 5; kernel_num ++)
    {
        switch (kernel_num) {
            case 0:
                kernel_name = "CPU openmp+avx2";
                break;
            case 1:
                kernel_name = "CPU openmp+avx512";
                break;
            case 2:
                kernel = gpu_1d;
                kernel_name = "GPU(1d)";
                break;
            case 3:
                kernel = gpu_2d;
                kernel_name = "GPU(2d)";
                break;
            case 4:
                kernel = gpu_3d;
                kernel_name = "GPU(3d)";
                break;
        }
        printf("Start %s %s execution\n", kernel_name, typeid(T).name());
        if(kernel_num == 0 || kernel_num == 1) {
            cpu_start = clock();
            if(kernel_num == 0)
                sum_array_cpu_simd_avx2(h_a, h_b, h_c, size);
            else
                sum_array_cpu_simd_avx512(h_a, h_b, h_c, size);
            cpu_end = clock();
            printf("%s %s time: %f\n", kernel_name, typeid(T).name(), (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
            if ( print_a)
            {
                print_matrix(h_c, 1, 10);
                print_matrix(h_c + size / 2 - 10, 1, 10);
                print_matrix(h_c + size / 2 + 10, 1, 10);
                print_matrix(h_c + size - 10, 1, 10);
            }
            printf("Compare %s %s result with cpu:\n", kernel_name, typeid(T).name());
            compare_arrays(h_c, h_cpu_results, size);
            if(kernel_num == 1)
                memcpy(h_cpu_results, h_c, NO_BYTES);
        }
        else{
            int dev = 0;
            cudaSetDevice(dev);
            (*kernel)(h_a, h_b, h_c, kernel_name, w, h, d);
            if ( print_a)
            {
                print_matrix(h_c, 1, 10);
                print_matrix(h_c + size / 2 - 10, 1, 10);
                print_matrix(h_c + size / 2 + 10, 1, 10);
                print_matrix(h_c + size - 10, 1, 10);
            }
            printf("Compare %s %s result with CPU(smid):\n", kernel_name, typeid(T).name());
            compare_arrays(h_c, h_cpu_results, size);
            cudaDeviceReset();
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_cpu_results);

    return 0;
}

int main(int argc, char** argv) {
    exec<int>(argc, argv);
    //exec<float>(argc, argv);
}