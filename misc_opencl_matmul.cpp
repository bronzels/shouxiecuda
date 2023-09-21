#include <vector>
#include <iostream>
#include <fstream>
#include <string.h>

#include <CL/cl.hpp>

#include "cblas.h"

#include "cuda_common.cuh"
#include "common.hpp"
#include "cl_common.h"

using namespace std;

#define KERNEL_SOURCE_FILE "misc_opencl_matmul.cl"
#define KERNEL_NAME_MATMULL "matMul"
#define KERNEL_NAME_MATMULLMEM "matMulMem"
#define KERNEL_NAME_MATMULLMEMPAD "matMulMemPad"

#define BDIM 32
/*
#define M 3072
#define K 1024
#define N 2048
*/
#define M 96*2
#define K 32*2
#define N 64*2
#define print_a true

int main( int argc, char* argv[]) {
    int m = M;
    int k = K;
    int n = N;
    int block_x = BDIM;
    int block_y = BDIM;

    int size_a = m * k;
    int size_b = k * n;
    int size = m * n;
    int byte_size = sizeof(float) * size;
    int byte_size_a = sizeof(float) * size_a;
    int byte_size_b = sizeof(float) * size_b;

    printf("Matmul for (%d X % d) and (%d X % d) matrix with work-group size %d X %d \n",m,k,k,n,block_x,block_y);

    cl_int err = CL_SUCCESS;
    cl_int api_err;

    float *h_a = (float *)malloc(byte_size_a);
    float *h_b = (float *)malloc(byte_size_b);
    float *h_c = (float *)malloc(byte_size);
    float *h_c_cpu = (float *)malloc(byte_size);

    initialize(h_a, size_a, INIT_RANDOM);
    initialize(h_b, size_b, INIT_RANDOM);

    clock_t t_start, t_stop;

    t_start = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, h_a, k, h_b, n, 0.0, h_c_cpu, n);
    t_stop = clock();
    printf("openBLAS time: %f \n", (double)((double)(t_stop - t_start)/CLOCKS_PER_SEC));
    if ( print_a)
    {
        print_matrix(h_c_cpu, 1, 10);
        print_matrix(h_c_cpu + m * n / 2 - 10, 1, 10);
        print_matrix(h_c_cpu + m * n / 2, 1, 10);
        print_matrix(h_c_cpu + m * n - 10, 1, 10);
    }

    cl::NDRange localSize(block_x, block_y);
    //cl::NDRange globalSize(ceil(m/(float)block_x), ceil(n/(float)block_y));
    //cl::NDRange globalSize(m/block_x, n/block_y);
    //！！！这里和cuda不一样gridSize是block的数量，opencl的globalSize是绝对的threads数目。
    cl::NDRange globalSize(m, n);
    /*
    cl::NDRange localSize(block_x * block_y);
    cl::NDRange globalSize(m * n);
    */

    vector<cl::Platform> platforms;
    err |= cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];
    vector<cl::Device> devices;
    err |= platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];
    if ( err != CL_SUCCESS )
    {
        printf("cl error till device initialization, code:%d, exit\n", err);
        exit(1);
    }

    cl::Context context({device});
    //cl_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;
    //cl::CommandQueue queue(context, device, &prop, &api_err);
    cl::CommandQueue queue(context, device, nullptr, &api_err);
    if ( api_err != CL_SUCCESS )
    {
        printf("cl command queue create error, code:%d, exit\n", api_err);
        exit(1);
    }

    const string kernel_source_file(KERNEL_SOURCE_FILE);
    const size_t kernel_source_len = 1024;

    /*
    //ifstream ifsSource(kernel_source_file, ios::in);
    ifstream ifsSource(kernel_source_file, ifstream::binary);
    char cSourceCL[kernel_source_len] = {0};
    if (ifsSource)
    {
        ifsSource.read(cSourceCL, kernel_source_len);
        ifsSource.close();
    }
    else
    {
        printf("kernel source file reading error, path:%s, exit\n", KERNEL_SOURCE_FILE);
        exit(1);
    }
    string strSource(cSourceCL);
    cout << strSource;
    */
    std::fstream kernelFile(kernel_source_file);
    std::string content(
            (std::istreambuf_iterator<char>(kernelFile)),
            std::istreambuf_iterator<char>()
    );

    const char* kernelCharArray = content.c_str();
    cout << kernelCharArray;
    cl::Program program(context, kernelCharArray);
    api_err = program.build({device});
    if ( api_err  != CL_SUCCESS )
    {
        printf("cl program build error, code:%d, exit\n", api_err);
        cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        exit(1);
    }
    //cl::Kernel kernel(program, KERNEL_NAME_MATMULL, &api_err);
    cl::Kernel kernel(program, KERNEL_NAME_MATMULLMEM, &api_err);
    //cl::Kernel kernel(program, KERNEL_NAME_MATMULLMEMPAD, &api_err);
    if ( api_err != CL_SUCCESS )
    {
        printf("cl kernel initialization error, code:%d, exit\n", api_err);
        exit(1);
    }

    clock_t time1,time2;
    time1 = clock();
    cl::Buffer d_a(context, CL_MEM_READ_ONLY, byte_size_a);
    cl::Buffer d_b(context, CL_MEM_READ_ONLY, byte_size_b);
    cl::Buffer d_c(context, CL_MEM_READ_WRITE, byte_size);
    cl::Buffer d_lock(context, CL_MEM_READ_WRITE, sizeof(int));

    int lock = 0;
    err |= queue.enqueueWriteBuffer(d_a, CL_TRUE, 0,
                                    byte_size_a, h_a);
    err |= queue.enqueueWriteBuffer(d_b, CL_TRUE, 0,
                                    byte_size_b, h_b);
    err |= queue.enqueueWriteBuffer(d_lock, CL_TRUE, 0,
                                    sizeof(int), &lock);
    if ( err != CL_SUCCESS )
    {
        printf("buffer writing error, code:%d, exit\n", err);
        exit(1);
    }

    kernel.setArg(0, m);
    kernel.setArg(1, n);
    kernel.setArg(2, k);
    kernel.setArg(3, d_a);
    kernel.setArg(4, d_b);
    kernel.setArg(5, d_c);
    kernel.setArg(6, d_lock);

    printf("Start executing openCL matmul\n");
    cl::Event event;
    time1 = clock();
    err |= queue.enqueueNDRangeKernel(kernel, NULL, globalSize, localSize, NULL, &event);
    if ( err != CL_SUCCESS )
    {
        printf("enqueue error, code:%d, exit\n", err);
        exit(1);
    }
    //queue.enqueueBarrierWithWaitList(NULL, &event);
    queue.finish();
    time2 = clock();
    printf("time spent executing openCL(clock delta) mat_mul:%f s\n", (double)(time2-time1)/CLOCKS_PER_SEC);
    //！！！很不稳定，不可用，有时一直是0，有时是非常大的数
    //event.wait();
    /*
    cl_ulong time_start, time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
    printf("time spent executing openCL(opencl profiling) mat_mul:%f ms\n", (double)(time_end-time_start)/1000000);
    */

    queue.enqueueReadBuffer(d_c, CL_TRUE, 0,
                            byte_size, h_c);
    if ( print_a)
    {
        print_matrix(h_c, 1, 10);
        print_matrix(h_c + m * n / 2 - 10, 1, 10);
        print_matrix(h_c + m * n / 2, 1, 10);
        print_matrix(h_c + m * n - 10, 1, 10);
    }

    printf("Compare openCL result with CPU openblas: \n");
    compare_arrays(h_c, h_c_cpu, n, (float)1e-3);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
}

/*
Matmul for (3072 X  1024) and (1024 X  2048) matrix with work-group size 16 X 16
openBLAS time: 0.680000
260.867188 258.839325 272.200104 259.467102 262.559875 264.663086 260.845032 261.726807 252.152420 266.629303
251.444153 252.885193 263.305847 250.878708 250.923477 265.865021 264.296234 259.650787 261.617706 263.400146

254.039627 249.893036 256.307709 255.454163 253.889618 256.472992 258.126465 255.533493 248.824341 256.803040
244.015488 246.079453 251.697250 240.663818 238.964722 257.467163 257.245667 254.682449 257.227539 248.325760

249.505005 246.721252 260.700256 252.699219 254.185776 253.741287 253.699432 253.509186 247.787949 256.697876
244.231720 242.848984 256.667969 243.062561 241.962982 256.572601 260.405029 249.768417 253.517181 251.136124

// OpenCL Kernel Function
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void matMul(
        const int Mdim,
        const int Ndim,
        const int Kdim,
        __global const float *A,
        __global const float *B,
        __global float *C) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int k;
    float tmp;

    if( i < Mdim && j < Ndim) {
        tmp = 0.0;
        for (k = 0; k < Kdim; k++)
            tmp += A[Kdim * i + k] * B[Ndim * k + j];
        C[Ndim * i + j] = tmp;
    }
}

__kernel void vecAdd( __global float *a,
                      __global float *b,
                      __global float *c,
                      const unsigned int n)
{
    //Get our global thread ID
    int id = get_global_id(0);

    //Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

time spent executing openCL(only kernel) mat_mul:18446744073709.550781 ms
time spent executing openCL mat_mul:0.080000 s
260.867249 258.839294 272.200287 259.467072 262.559906 264.663055 260.845062 261.726746 252.152237 266.629395
251.444016 252.885284 263.306030 250.878876 250.923477 265.865234 264.296173 259.650665 261.617645 263.400238

254.039658 249.893311 256.307739 255.454010 253.889572 256.472931 258.126587 255.533661 248.824295 256.803040
244.015640 246.079483 251.697388 240.663788 238.964905 257.467255 257.245667 254.682236 257.227600 248.325653

249.505203 246.721344 260.700226 252.699432 254.185791 253.741119 253.699326 253.509308 247.787872 256.697815
244.231522 242.848846 256.667999 243.062469 241.962845 256.572418 260.404968 249.768341 253.517120 251.136383

Compare openCL result with CPU openblas:
Arrays are same

*/





