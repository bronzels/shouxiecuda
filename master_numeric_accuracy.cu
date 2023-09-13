#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"

#include "helper_cuda.h"
#include "helper_functions.h"

#include <stdio.h>
#include <ctime>
#include <random>
using namespace std;

/*
cpu accuracy
float           1-8-23
double          1-11-52

gpu accuracy
*/

void float_accuracy_comparison()
{
    printf("float accuracy comparison \n");
    float a = 3.1415927f;
    float b = 3.1415928f;
    if (a == b)
    {
        printf("a is equal to b\n");
    }
    else
    {
        printf("a does not equal b\n");
    }
}

void double_accuracy_comparison()
{
    printf("\ndouble accuracy comparison \n");
    double a = 3.1415927;
    double b = 3.1415928;
    if (a == b)
    {
        printf("a is equal to b\n");
    }
    else
    {
        printf("a does not equal b\n");
    }
}

__global__ void lots_of_float_compute(float *inputs, int N, size_t niters,
                                      float *outputs)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nthreads = gridDim.x * blockDim.x;

    for(; tid < N; tid += nthreads)
    {
        size_t iter;
        float val = inputs[tid];

        for (iter = 0; iter < niters; iter++)
        {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }
        
        outputs[tid] = val;
    }
}

__global__ void lots_of_double_compute(double *inputs, int N, size_t niters,
                                      double *outputs)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nthreads = gridDim.x * blockDim.x;

    for(; tid < N; tid += nthreads)
    {
        size_t iter;
        double val = inputs[tid];

        for (iter = 0; iter < niters; iter++)
        {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }

        outputs[tid] = val;
    }
}

static void run_float_test(size_t N, int niters, int blocksPerGrid, int threadPerBlock,
                           long *to_device_clock_cyl, long *kernel_clock_cyl, long *from_device_clock_cyl,
                           float *sample, int sampleLength)
{
    int i;
    float *h_floatInputs, *h_floatOutputs;
    float *d_floatInputs, *d_floatOutputs;

    h_floatInputs = (float *)malloc(sizeof(float) * N);
    h_floatOutputs = (float *)malloc(sizeof(float) * N);
    cudaMalloc((void **)&d_floatInputs, sizeof(float) * N);
    cudaMalloc((void **)&d_floatOutputs, sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        h_floatInputs[i] = (float)i;
    }

    clock_t ops_start, ops_end;

    ops_start = clock();
    cudaMemcpy(d_floatInputs, h_floatInputs, sizeof(float)*N, cudaMemcpyHostToDevice);
    ops_end = clock();
    *to_device_clock_cyl = ops_end - ops_start;

    ops_start = clock();
    lots_of_float_compute<<<blocksPerGrid, threadPerBlock>>>(d_floatInputs, N, niters, d_floatOutputs);
    cudaDeviceSynchronize();
    ops_end = clock();
    *kernel_clock_cyl = ops_end - ops_start;

    ops_start = clock();
    cudaMemcpy(h_floatOutputs, d_floatOutputs, sizeof(float)*N, cudaMemcpyDeviceToHost);
    ops_end = clock();
    *from_device_clock_cyl = ops_end - ops_start;

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_floatOutputs[i];
    }

    cudaFree(d_floatInputs);
    cudaFree(d_floatOutputs);
    free(h_floatInputs);
    free(h_floatOutputs);
}

static void run_double_test(size_t N, int niters, int blocksPerGrid, int threadPerBlock,
                           long *to_device_clock_cyl, long *kernel_clock_cyl, long *from_device_clock_cyl,
                           double *sample, int sampleLength)
{
    int i;
    double *h_doubleInputs, *h_doubleOutputs;
    double *d_doubleInputs, *d_doubleOutputs;

    h_doubleInputs = (double *)malloc(sizeof(double) * N);
    h_doubleOutputs = (double *)malloc(sizeof(double) * N);
    cudaMalloc((void **)&d_doubleInputs, sizeof(double) * N);
    cudaMalloc((void **)&d_doubleOutputs, sizeof(double) * N);

    for (i = 0; i < N; i++)
    {
        h_doubleInputs[i] = (double)i;
    }

    clock_t ops_start, ops_end;

    ops_start = clock();
    cudaMemcpy(d_doubleInputs, h_doubleInputs, sizeof(double)*N, cudaMemcpyHostToDevice);
    ops_end = clock();
    *to_device_clock_cyl = ops_end - ops_start;

    ops_start = clock();
    lots_of_double_compute<<<blocksPerGrid, threadPerBlock>>>(d_doubleInputs, N, niters, d_doubleOutputs);
    cudaDeviceSynchronize();
    ops_end = clock();
    *kernel_clock_cyl = ops_end - ops_start;

    ops_start = clock();
    cudaMemcpy(h_doubleOutputs, d_doubleOutputs, sizeof(double)*N, cudaMemcpyDeviceToHost);
    ops_end = clock();
    *from_device_clock_cyl = ops_end - ops_start;

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_doubleOutputs[i];
    }

    cudaFree(d_doubleInputs);
    cudaFree(d_doubleOutputs);
    free(h_doubleInputs);
    free(h_doubleOutputs);
}

void gpu_float_double_elapsed()
{
    int i;
    double meanFloatToDeviceTime, meanFloatKernelTime, meanFloatFromDeviceTime;
    double meanDoubleToDeviceTime, meanDoubleKernelTime, meanDoubleFromDeviceTime;
    struct cudaDeviceProp deviceProperties;
    size_t totalMem, freeMem;
    float *floatSample;
    double *doubleSample;
    int sampleLength = 10;
    int nRuns = 5;
    int nKernelIters = 20;

    meanFloatToDeviceTime = meanFloatKernelTime = meanFloatFromDeviceTime = 0.0;
    meanDoubleToDeviceTime = meanDoubleKernelTime = meanDoubleFromDeviceTime = 0.0;

    cudaMemGetInfo(&freeMem, &totalMem);
    cudaGetDeviceProperties(&deviceProperties, 0);

    size_t N = (freeMem * 0.9 /2) / sizeof(double);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    if(blocksPerGrid > deviceProperties.maxGridSize[0])
    {
        blocksPerGrid = deviceProperties.maxGridSize[0];
    }

    printf("Runing %d blockswith %d threads/block over %lu elements\n",
           blocksPerGrid, threadsPerBlock, N);

    floatSample = (float*)malloc(sizeof(float) * sampleLength);
    doubleSample = (double *)malloc(sizeof(double) * sampleLength);

    for (i = 0; i < nRuns; i++)
    {
        long toDeviceTime, kernelTime, fromDeviceTime;

        run_float_test(N, nKernelIters, blocksPerGrid, threadsPerBlock,
                       &toDeviceTime, &kernelTime, &fromDeviceTime,
                       floatSample, sampleLength);
        meanFloatToDeviceTime += toDeviceTime;
        meanFloatKernelTime += kernelTime;
        meanFloatFromDeviceTime += fromDeviceTime;

        run_double_test(N, nKernelIters, blocksPerGrid, threadsPerBlock,
                        &toDeviceTime, &kernelTime, &fromDeviceTime,
                        doubleSample, sampleLength);
        meanDoubleToDeviceTime += toDeviceTime;
        meanDoubleKernelTime += kernelTime;
        meanDoubleFromDeviceTime += fromDeviceTime;
    }

    meanFloatToDeviceTime /= nRuns;
    meanFloatKernelTime /= nRuns;
    meanFloatFromDeviceTime /= nRuns;
    meanDoubleToDeviceTime /= nRuns;
    meanDoubleKernelTime /= nRuns;
    meanDoubleFromDeviceTime /= nRuns;

    meanFloatToDeviceTime /= CLOCKS_PER_SEC;
    meanFloatKernelTime /= CLOCKS_PER_SEC;
    meanFloatFromDeviceTime /= CLOCKS_PER_SEC;
    meanDoubleToDeviceTime /= CLOCKS_PER_SEC;
    meanDoubleKernelTime /= CLOCKS_PER_SEC;
    meanDoubleFromDeviceTime /= CLOCKS_PER_SEC;

    printf("For single-precision floating point, mean times for:\n");
    printf("  Copy to device:     %f\n", meanFloatToDeviceTime);
    printf("  Kernel execution:   %f\n", meanFloatKernelTime);
    printf("  Copy from device:   %f\n", meanFloatFromDeviceTime);
    printf("For double-precision floating point, mean times for:\n");
    printf("  Copy to device:     %f s (%0.2f x slower than signle-precision\n", meanDoubleToDeviceTime, meanDoubleToDeviceTime/meanFloatToDeviceTime);
    printf("  Kernel execution:   %f s (%0.2f x slower than signle-precision\n", meanDoubleKernelTime, meanDoubleKernelTime/meanFloatKernelTime);
    printf("  Copy from device:   %f s (%0.2f x slower than signle-precision\n", meanDoubleFromDeviceTime, meanDoubleFromDeviceTime/meanFloatFromDeviceTime);
}


__global__ void standard_kernel(float a, float *out, int iters)
{
    int i;
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (tid == 0)
    {
        float tmp;

        for (i = 0; i < iters; i++)
        {
            tmp = powf(a, 2.0f);
        }

        *out = tmp;
    }
}

__global__ void intrinsic_kernel(float a, float *out, int iters)
{
    int i;
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (tid == 0)
    {
        float tmp;

        for (i = 0; i < iters; i++)
        {
            tmp = __powf(a, 2.0f);
        }

        *out = tmp;
    }
}

__global__ void standard(float *ptr)
{
    *ptr = powf(*ptr, 2.0f);
}

__global__ void intrinsic(float *ptr)
{
    *ptr = __powf(*ptr, 2.0f);
}

void standar_intrinsic_once()
{
    float value = 23.3;
    int SIZE = sizeof(float);

    float h_val;
    float *d_val;

    cudaMalloc((void**)&d_val, SIZE);
    cudaMemcpy(d_val, &value, SIZE, cudaMemcpyHostToDevice);
    standard <<<1, 1 >>> (d_val);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_val, d_val, SIZE, cudaMemcpyDeviceToHost);
    printf("standard:%0.7f\n", h_val);

    cudaMemcpy(d_val, &value, SIZE, cudaMemcpyHostToDevice);
    intrinsic <<<1, 1 >>> (d_val);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_val, d_val, SIZE, cudaMemcpyDeviceToHost);
    printf("intrinsic:%0.6f\n", h_val);

    cudaDeviceReset();
}

void standard_intrinsic_accu()
{
	int i;
	int runs = 30;
	int iters = 1000;

	float *d_standard_out, h_standard_out;
	gpuErrchk(cudaMalloc((void **)&d_standard_out, sizeof(float)));

	float *d_intrinsic_out, h_intrinsic_out;
	gpuErrchk(cudaMalloc((void **)&d_intrinsic_out, sizeof(float)));

	float input_value = 1023.273;

	double mean_intrinsic_time = 0.0;
	double mean_standard_time = 0.0;

	clock_t ops_start, ops_end;

	for (i = 0; i < runs; i++)
	{
		//ops_start = clock();
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        StopWatchInterface *timer = NULL;
        sdkCreateTimer(&timer);
        sdkResetTimer(&timer);
        float gpu_time = 0.0f;
        sdkStartTimer(&timer);
        cudaEventRecord(start, 0);
		standard_kernel << <1, 32 >> >(input_value, d_standard_out, iters);
        cudaEventRecord(stop, 0);
        sdkStopTimer(&timer);
        unsigned long int counter = 0;
        while(cudaEventQuery(stop) == cudaErrorNotReady)
        {
            counter ++;
        }
        checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
        //printf("time spent executing by the GPU: %.6f\n", gpu_time);
        mean_standard_time += gpu_time;
        gpuErrchk(cudaDeviceSynchronize());
		//ops_end = clock();
		//mean_standard_time += ops_end - ops_start;

		//ops_start = clock();
        cudaEvent_t start2, stop2;
        checkCudaErrors(cudaEventCreate(&start2));
        checkCudaErrors(cudaEventCreate(&stop2));
        StopWatchInterface *timer2 = NULL;
        sdkCreateTimer(&timer2);
        sdkResetTimer(&timer2);
        float gpu_time2 = 0.0f;
        sdkStartTimer(&timer2);
        cudaEventRecord(start2, 0);
		intrinsic_kernel << <1, 32 >> >(input_value, d_intrinsic_out, iters);
        cudaEventRecord(stop2, 0);
        sdkStopTimer(&timer2);
        unsigned long int counter2 = 0;
        while(cudaEventQuery(stop2) == cudaErrorNotReady)
        {
            counter2 ++;
        }
        checkCudaErrors(cudaEventElapsedTime(&gpu_time2, start2, stop2));
        //printf("time spent executing by the GPU: %.6f\n", gpu_time2);
        mean_intrinsic_time += gpu_time2;
		gpuErrchk(cudaDeviceSynchronize());
		//ops_end = clock();
		//mean_intrinsic_time += ops_end - ops_start;
	}

	//mean_intrinsic_time = mean_intrinsic_time / CLOCKS_PER_SEC;
	//mean_standard_time = mean_standard_time / CLOCKS_PER_SEC;

	gpuErrchk(cudaMemcpy(&h_standard_out, d_standard_out, sizeof(float),
		cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(float),
		cudaMemcpyDeviceToHost));
	float host_value = powf(input_value, 2.0f);

	printf("Host calculated\t\t\t%.6f\n", host_value);
	printf("Standard Device calculated\t%.6f\n", h_standard_out);
	printf("Intrinsic Device calculated\t%.6f\n", h_intrinsic_out);
	printf("Host equals Standard?\t\t%s diff=%e\n",
		host_value == h_standard_out ? "Yes" : "No",
		fabs(host_value - h_standard_out));
	printf("Host equals Intrinsic?\t\t%s diff=%e\n",
		host_value == h_intrinsic_out ? "Yes" : "No",
		fabs(host_value - h_intrinsic_out));
	printf("Standard equals Intrinsic?\t%s diff=%e\n",
		h_standard_out == h_intrinsic_out ? "Yes" : "No",
		fabs(h_standard_out - h_intrinsic_out));
	printf("\n");
	printf("Mean execution time for standard function powf:    %f s\n",
		mean_standard_time);
	printf("Mean execution time for intrinsic function __powf: %f s\n",
		mean_intrinsic_time);
}
int main()
{
    /*
	float_accuracy_comparison();
	double_accuracy_comparison();
    gpu_float_double_elapsed();
    standar_intrinsic_once();
    */

    standard_intrinsic_accu();
    return 0;
}
/*
nvcc --fmad=true, faster use MAD instruction in GPU, less accurate
nvcc --fmac=false, slower, more accurate
 *
Host calculated                 1047087.625000
Standard Device calculated      1047087.687500
Intrinsic Device calculated     1047086.812500
Host equals Standard?           No diff=6.250000e-02
Host equals Intrinsic?          No diff=8.125000e-01
Standard equals Intrinsic?      No diff=8.750000e-01

Mean execution time for standard function powf:    0.130592 s
Mean execution time for intrinsic function __powf: 0.075232 s


 */
/*
nvcc --ptx -o master_numeric_accuracy.ptx master_numeric_accuracy.cu

.visible .entry _Z8standardPf(
        .param .u64 _Z8standardPf_param_0
)
{
        .reg .pred      %p<17>;
        .reg .f32       %f<104>;
        .reg .b32       %r<15>;
        .reg .b64       %rd<3>;


        ld.param.u64    %rd2, [_Z8standardPf_param_0];
        cvta.to.global.u64      %rd1, %rd2;
        mov.f32         %f15, 0f3F800000;
        cvt.rzi.f32.f32         %f16, %f15;
        add.f32         %f17, %f16, %f16;
        mov.f32         %f18, 0f40000000;
        sub.f32         %f19, %f18, %f17;
        abs.f32         %f1, %f19;
        ld.global.f32   %f2, [%rd1];
        abs.f32         %f3, %f2;
        setp.lt.f32     %p2, %f3, 0f00800000;
        mul.f32         %f20, %f3, 0f4B800000;
        selp.f32        %f21, %f20, %f3, %p2;
        selp.f32        %f22, 0fC3170000, 0fC2FE0000, %p2;
        mov.b32         %r1, %f21;
        and.b32         %r2, %r1, 8388607;
        or.b32          %r3, %r2, 1065353216;
        mov.b32         %f23, %r3;
        shr.u32         %r4, %r1, 23;
        cvt.rn.f32.u32  %f24, %r4;
        add.f32         %f25, %f22, %f24;
        setp.gt.f32     %p3, %f23, 0f3FB504F3;
        mul.f32         %f26, %f23, 0f3F000000;
        add.f32         %f27, %f25, 0f3F800000;
        selp.f32        %f28, %f27, %f25, %p3;
        selp.f32        %f29, %f26, %f23, %p3;
        add.f32         %f30, %f29, 0fBF800000;
        add.f32         %f31, %f29, 0f3F800000;
        rcp.approx.ftz.f32      %f32, %f31;
        add.f32         %f33, %f30, %f30;
        mul.f32         %f34, %f33, %f32;
        mul.f32         %f35, %f34, %f34;
        mov.f32         %f36, 0f3C4CAF63;
        mov.f32         %f37, 0f3B18F0FE;
        fma.rn.f32      %f38, %f37, %f35, %f36;
        mov.f32         %f39, 0f3DAAAABD;
        fma.rn.f32      %f40, %f38, %f35, %f39;
        mul.rn.f32      %f41, %f40, %f35;
        mul.rn.f32      %f42, %f41, %f34;
        sub.f32         %f43, %f30, %f34;
        add.f32         %f44, %f43, %f43;
        neg.f32         %f45, %f34;
        fma.rn.f32      %f46, %f45, %f30, %f44;
        mul.rn.f32      %f47, %f32, %f46;
        add.f32         %f48, %f42, %f34;
        sub.f32         %f49, %f34, %f48;
        add.f32         %f50, %f42, %f49;
        add.f32         %f51, %f47, %f50;
        add.f32         %f52, %f48, %f51;
        sub.f32         %f53, %f48, %f52;
        add.f32         %f54, %f51, %f53;
        mov.f32         %f55, 0f3F317200;
        mul.rn.f32      %f56, %f28, %f55;
        mov.f32         %f57, 0f35BFBE8E;
        mul.rn.f32      %f58, %f28, %f57;
        add.f32         %f59, %f56, %f52;
        sub.f32         %f60, %f56, %f59;
        add.f32         %f61, %f52, %f60;
        add.f32         %f62, %f54, %f61;
        add.f32         %f63, %f58, %f62;
        add.f32         %f64, %f59, %f63;
        sub.f32         %f65, %f59, %f64;
        add.f32         %f66, %f63, %f65;
        mul.rn.f32      %f67, %f18, %f64;
        neg.f32         %f68, %f67;
        fma.rn.f32      %f69, %f18, %f64, %f68;
        fma.rn.f32      %f70, %f18, %f66, %f69;
        mov.f32         %f71, 0f00000000;
        fma.rn.f32      %f72, %f71, %f64, %f70;
        add.rn.f32      %f73, %f67, %f72;
        neg.f32         %f74, %f73;
        add.rn.f32      %f75, %f67, %f74;
        add.rn.f32      %f76, %f75, %f72;
        mov.b32         %r5, %f73;
        setp.eq.s32     %p4, %r5, 1118925336;
        add.s32         %r6, %r5, -1;
        mov.b32         %f77, %r6;
        add.f32         %f78, %f76, 0f37000000;
        selp.f32        %f4, %f78, %f76, %p4;
        selp.f32        %f79, %f77, %f73, %p4;
        mov.f32         %f80, 0f3FB8AA3B;
        mul.rn.f32      %f81, %f79, %f80;
        cvt.rzi.f32.f32         %f82, %f81;
        abs.f32         %f83, %f82;
        setp.gt.f32     %p5, %f83, 0f42FC0000;
        mov.b32         %r7, %f82;
        and.b32         %r8, %r7, -2147483648;
        or.b32          %r9, %r8, 1123811328;
        mov.b32         %f84, %r9;
        selp.f32        %f85, %f84, %f82, %p5;
        mov.f32         %f86, 0fBF317218;
        fma.rn.f32      %f87, %f85, %f86, %f79;
        mov.f32         %f88, 0f3102E308;
        fma.rn.f32      %f89, %f85, %f88, %f87;
        mul.f32         %f90, %f89, 0f3FB8AA3B;
        add.f32         %f91, %f85, 0f4B40007F;
        mov.b32         %r10, %f91;
        shl.b32         %r11, %r10, 23;
        mov.b32         %f92, %r11;
        ex2.approx.ftz.f32      %f93, %f90;
        mul.f32         %f5, %f93, %f92;
        setp.eq.f32     %p6, %f5, 0f7F800000;
        mov.f32         %f101, 0f7F800000;
        @%p6 bra        $L__BB0_2;

        fma.rn.f32      %f101, %f5, %f4, %f5;

$L__BB0_2:
        setp.lt.f32     %p7, %f2, 0f00000000;
        setp.eq.f32     %p8, %f1, 0f3F800000;
        and.pred        %p1, %p7, %p8;
        setp.eq.f32     %p9, %f2, 0f00000000;
        @%p9 bra        $L__BB0_6;
        bra.uni         $L__BB0_3;

$L__BB0_6:
        add.f32         %f98, %f2, %f2;
        selp.f32        %f103, %f98, 0f00000000, %p8;
        bra.uni         $L__BB0_7;

$L__BB0_3:
        mov.b32         %r12, %f101;
        xor.b32         %r13, %r12, -2147483648;
        mov.b32         %f94, %r13;
        selp.f32        %f103, %f94, %f101, %p1;
        setp.geu.f32    %p10, %f2, 0f00000000;
        @%p10 bra       $L__BB0_7;

        cvt.rzi.f32.f32         %f96, %f18;
        setp.eq.f32     %p11, %f96, 0f40000000;
        @%p11 bra       $L__BB0_7;

        mov.f32         %f103, 0f7FFFFFFF;

$L__BB0_7:
        add.f32         %f99, %f3, 0f40000000;
        mov.b32         %r14, %f99;
        setp.lt.s32     %p13, %r14, 2139095040;
        @%p13 bra       $L__BB0_12;

        setp.gtu.f32    %p14, %f3, 0f7F800000;
        @%p14 bra       $L__BB0_11;
        bra.uni         $L__BB0_9;

$L__BB0_11:
        add.f32         %f103, %f2, 0f40000000;
        bra.uni         $L__BB0_12;

$L__BB0_9:
        setp.neu.f32    %p15, %f3, 0f7F800000;
        @%p15 bra       $L__BB0_12;

        selp.f32        %f103, 0fFF800000, 0f7F800000, %p1;

$L__BB0_12:
        setp.eq.f32     %p16, %f2, 0f3F800000;
        selp.f32        %f100, 0f3F800000, %f103, %p16;
        st.global.f32   [%rd1], %f100;
        ret;

}

.visible .entry _Z9intrinsicPf(
        .param .u64 _Z9intrinsicPf_param_0
)
{
        .reg .f32       %f<5>;
        .reg .b64       %rd<3>;


        ld.param.u64    %rd1, [_Z9intrinsicPf_param_0];
        cvta.to.global.u64      %rd2, %rd1;
        ld.global.f32   %f1, [%rd2];
        lg2.approx.f32  %f2, %f1;
        add.f32         %f3, %f2, %f2;
        ex2.approx.f32  %f4, %f3;
        st.global.f32   [%rd2], %f4;
        ret;

}
 */