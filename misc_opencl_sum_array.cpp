#include <vector>
#include <iostream>
#include <string.h>

#include <CL/cl.hpp>

#include "common.cpph"

using namespace std;

const char * kernelSource =                                     "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                   \n" \
"__kernel void vecAdd( __global float *a,                        \n" \
"                      __global float *b,                        \n" \
"                      __global float *c,                        \n" \
"                      const unsigned int n)                     \n" \
"{                                                               \n" \
"   //Get our global thread ID                                   \n" \
"   int id = get_global_id(0);                                   \n" \
"                                                                \n" \
"   //Make sure we do not go out of bounds                       \n" \
"   if (id < n)                                                  \n" \
"       c[id] = a[id] + b[id];                                   \n" \
"}                                                               \n" \
"\n";

int main( int argc, char* argv[]) {
    printf(kernelSource);
    cl_int err = CL_SUCCESS;
    cl_int api_err;

    unsigned int n = 1 << 29;
    size_t bytes = n*sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_c_cpu = (float *)malloc(bytes);

    initialize(h_a, n, INIT_RANDOM);
    initialize(h_b, n, INIT_RANDOM);

    sum_array_cpu(h_a, h_b, h_c_cpu, n);
    print_array(h_c_cpu, 16);
    print_array(h_c_cpu + n/2, 16);
    print_array(h_c_cpu + n - 16, 16);

    size_t local_size = 128;
    cl::NDRange localSize(local_size);
    cl::NDRange globalSize(ceil(n/(float)local_size)*local_size);

    vector<cl::Platform> platforms;
    err |= cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];
    vector<cl::Device> devices;
    err |= platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context({device});
    cl_queue_properties prop = CL_QUEUE_PROFILING_ENABLE;
    cl::CommandQueue queue(context, device, &prop, &api_err);
    err += api_err;
    if ( err != 0 )
    {
        printf("cl env context create error, code:%d, exit\n", err);
        exit(1);
    }

    int length;

    cl_platform_info plat_info_des_list[] = {
            CL_PLATFORM_PROFILE,
            CL_PLATFORM_VERSION,
            CL_PLATFORM_NUMERIC_VERSION,
            CL_PLATFORM_NAME,
            CL_PLATFORM_VENDOR,
            CL_PLATFORM_EXTENSIONS,
            CL_PLATFORM_EXTENSIONS_WITH_VERSION,
            CL_PLATFORM_HOST_TIMER_RESOLUTION,
    };
    char plat_info_des_str_list[][255] = {
            "CL_PLATFORM_PROFILE",
            "CL_PLATFORM_VERSION",
            "CL_PLATFORM_NUMERIC_VERSION",
            "CL_PLATFORM_NAME",
            "CL_PLATFORM_VENDOR",
            "CL_PLATFORM_EXTENSIONS",
            "CL_PLATFORM_EXTENSIONS_WITH_VERSION",
            "CL_PLATFORM_HOST_TIMER_RESOLUTION",
    };
    length = sizeof(plat_info_des_list) / sizeof(plat_info_des_list[0]);
    printf("length:%d\n", length);
    for (int i = 0; i< length; i++) {
        string info;
        err = platform.getInfo(plat_info_des_list[i], &info);
        if (err != CL_SUCCESS)
            cout<<plat_info_des_str_list[i]<<": can't get"<<endl;
        else {
            cout<<plat_info_des_str_list[i]<<": "<<info<<endl;
        }
    }
    /*
CL_PLATFORM_PROFILE: FULL_PROFILE
CL_PLATFORM_VERSION: OpenCL 3.0 CUDA 12.0.139
CL_PLATFORM_NUMERIC_VERSION:
CL_PLATFORM_NAME: NVIDIA CUDA
CL_PLATFORM_VENDOR: NVIDIA Corporation
CL_PLATFORM_EXTENSIONS: can't get
CL_PLATFORM_EXTENSIONS_WITH_VERSION: NVIDIA Corporation
CL_PLATFORM_HOST_TIMER_RESOLUTION:
     */
    cl_device_info dev_info_des_list[] = {
            CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
            CL_DEVICE_BUILT_IN_KERNELS,
            CL_DEVICE_PARENT_DEVICE,
            CL_DEVICE_PARTITION_TYPE,
            CL_DEVICE_REFERENCE_COUNT,
    };
    char dev_info_des_str_list[][255] = {
            "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE",
            "CL_DEVICE_BUILT_IN_KERNELS",
            "CL_DEVICE_PARENT_DEVICE",
            "CL_DEVICE_PARTITION_TYPE",
            "CL_DEVICE_REFERENCE_COUNT",
    };
    length = sizeof(dev_info_des_list) / sizeof(dev_info_des_list[0]);
    printf("length:%d\n", length);
    for (int i = 0; i< length; i++) {
        string info;
        err = device.getInfo(dev_info_des_list[i], &info);
        if (err != CL_SUCCESS)
            cout<<dev_info_des_str_list[i]<<": can't get"<<endl;
        else {
            cout<<dev_info_des_str_list[i]<<": "<<info<<endl;
        }
    }
    /*
    device不报错，可是信息全是空的
    */

    cl::Program program(context, kernelSource);
    program.build({device});
    cl::Kernel kernel(program, "vecAdd", &err);
    if ( err != 0 )
    {
        printf("cl env initialization error, code:%d, exit\n", err);
        exit(2);
    }

    clock_t time1,time2;
    time1 = clock();
    cl::Buffer d_a(context, CL_MEM_READ_ONLY, bytes);
    cl::Buffer d_b(context, CL_MEM_READ_ONLY, bytes);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY, bytes);

    err |= queue.enqueueWriteBuffer(d_a, CL_TRUE, 0,
                                    bytes, h_a);
    err |= queue.enqueueWriteBuffer(d_b, CL_TRUE, 0,
                                    bytes, h_b);
    if ( err != 0 )
    {
        printf("buffer writing error, code:%d, exit\n", err);
        exit(3);
    }

    kernel.setArg(0, d_a);
    kernel.setArg(1, d_b);
    kernel.setArg(2, d_c);
    kernel.setArg(3, n);

    cl::Event event;
    err |= queue.enqueueNDRangeKernel(kernel, NULL, globalSize, localSize,
                                  NULL, &event);
    if ( err != 0 )
    {
        printf("enqueue error, code:%d, exit\n", err);
        exit(4);
    }
    event.wait();
    cl_ulong time_start, time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
    printf("time spent executing openCL(only kernel) sum_array:%f ms\n", (double)(time_end-time_start)/1000000);

    queue.finish();

    queue.enqueueReadBuffer(d_c, CL_TRUE, 0,
                        bytes, h_c);
    time2 = clock();
    printf("time spent executing openCL sum_array:%f s\n", (double)(time2-time1)/CLOCKS_PER_SEC);
    print_array(h_c, 16);
    print_array(h_c + n/2, 16);
    print_array(h_c + n - 16, 16);

    printf("Compare openCL result with CPU: \n");
    compare_arrays(h_c, h_c_cpu, n, (float)1e-4);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
}







