#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <string.h>

#include "common.h"

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
    int i = 0;
    size_t globalSize, localSize;
    cl_int err;
    double sum = 0;

    int n = 1 << 27;
    float *h_a;
    float *h_b;
    float *h_c;
    float *h_c_cpu;

    cl_mem d_a;
    cl_mem d_b;
    cl_mem d_c;

    cl_platform_id platform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    size_t bytes = n*sizeof(float);

    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);
    h_c_cpu = (float *)malloc(bytes);
    if ( h_a == NULL || h_b == NULL || h_c == NULL || h_c_cpu == NULL)
    {
        printf("host memory allocation error, exit\n");
        exit(1);
    }

    initialize_f(h_a, n, INIT_RANDOM);
    initialize_f(h_b, n, INIT_RANDOM);

    sum_array_cpu_f(h_a, h_b, h_c_cpu, n);
    print_array_f(h_c_cpu, 16);
    print_array_f(h_c_cpu + n/2, 16);
    print_array_f(h_c_cpu + n - 16, 16);

    localSize = 128;
    globalSize = ceil(n/(float)localSize)*localSize;

    err = clGetPlatformIDs(1, &platform, NULL);
    //err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    //改成cpu以后create context报错Segmentation fault，需要下载编译intel的驱动，把so放到路径才行
    if ( err != 0 )
    {
        printf("cl env context create error, code:%d, exit\n", err);
        exit(2);
    }
    size_t buf_len = 255;
    char info[buf_len];
    memset(info, '\0', buf_len);
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
        size_t info_put_len;
        err = clGetPlatformInfo(platform, plat_info_des_list[i], buf_len, info, &info_put_len);
        if (err != CL_SUCCESS)
            printf("%s: can't get\n", plat_info_des_str_list[i]);
        else {
            /*
            if( info_put_len < buf_len)
                info[info_put_len] = '\0';
            else
                info[buf_len-1] = '\0';
            */
            printf("%s: %s\n", plat_info_des_str_list[i], info);
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
        //memset(info, '\0', buf_len);
        memset(info, 0, buf_len);
        size_t info_put_len;
        err = clGetDeviceInfo(device_id, dev_info_des_list[i], buf_len, info, &info_put_len);
        if (err != CL_SUCCESS)
            printf("%s: can't get\n", dev_info_des_str_list[i]);
        else {
            /*
            if( info_put_len < buf_len)
                info[info_put_len] = '\0';
            else
                info[buf_len-1] = '\0';
            */
            printf("%s: %s\n", dev_info_des_str_list[i], info);
        }
    }
    /*
    device不报错，可是信息全是空的
    */

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) & kernelSource, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "vecAdd", &err);
    if ( err != 0 )
    {
        printf("cl env initialization error, code:%d, exit\n", err);
        exit(2);
    }

    clock_t time1,time2;
    time1 = clock();
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                               bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                               bytes, h_b, 0, NULL, NULL);
    if ( err != 0 )
    {
        printf("buffer writing error, code:%d, exit\n", err);
        exit(3);
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                 0, NULL, &event);
    if ( err != 0 )
    {
        printf("enqueue error, code:%d, exit\n", err);
        exit(4);
    }
    clWaitForEvents(1, &event);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    printf("time spent executing openCL(only kernel) sum_array:%f ms\n", (double)(time_end-time_start)/1000000);

    clFinish(queue);

    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                        bytes, h_c, 0, NULL, NULL);
    time2 = clock();
    printf("time spent executing openCL sum_array:%f s\n", (double)(time2-time1)/CLOCKS_PER_SEC);
    print_array_f(h_c, 16);
    print_array_f(h_c + n/2, 16);
    print_array_f(h_c + n - 16, 16);

    printf("Compare openCL result with CPU: \n");
    compare_arrays_f(h_c, h_c_cpu, n, 1e-4);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);

    cl_int status;
    status = clReleaseKernel(kernel);  // Release kernel.
    status = clReleaseProgram(program);  // Release program object.
    status = clReleaseMemObject(d_a);  // Release mem object.
    status = clReleaseMemObject(d_b);
    status = clReleaseMemObject(d_c);
    status = clReleaseCommandQueue(queue);  // Release  Command queue.
    status = clReleaseContext(context);  // Release context.
}







