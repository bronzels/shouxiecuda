#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void query_device()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if(deviceCount == 0)
    {
        printf("No CUDA support device found");
        return;
    }

    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);
    printf("Device %d: %s\n", devNo, iProp.name);
    printf("    Number of multiprocessor:                      %d\n", iProp.multiProcessorCount);
    printf("    clock rate:                                    %d\n", iProp.clockRate);
    printf("    Compute capability:                            %d.%d\n", iProp.major, iProp.minor);
    printf("    Total amount of global memory:                 %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
    printf("    Total amount of constant memory:               %4.2f KB\n", iProp.totalConstMem / 1024.0);
    printf("    Total amount of shared memory per block:       %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
    printf("    Total amount of shared memory per MP:          %4.2f KB\n", iProp.sharedMemPerMultiprocessor / 1024.0);
    printf("    Warp size:                                     %d\n", iProp.warpSize);
    printf("    Maximum number of threads per block:           %d\n", iProp.maxThreadsPerBlock);
    printf("    Maximum number of threads per multiprocessor:  %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("    Maximum number of warps per multiprocessor:    %d\n", iProp.maxThreadsPerMultiProcessor / 32);
    printf("    Maximum Grid size:                             (%d,%d,%d)\n", iProp.maxGridSize[0],iProp.maxGridSize[1],iProp.maxGridSize[2]);
    printf("    Maximum block dimension                :       (%d,%d,%d)\n", iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[02]);
    printf("    concurrentKernels:                             %d\n", iProp.concurrentKernels);
    printf("    canMapHostMemory:                              %d\n", iProp.canMapHostMemory);

    cudaSharedMemConfig pConfig;
    printf("set cudaSharedMemBankSize 8 bytes\n");
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceGetSharedMemConfig(&pConfig);
    printf("---result with Bank Mode:%s\n", pConfig == 1 ? "4-Byte" : "8-Byte");
    printf("set cudaSharedMemBankSize 4 bytes\n");
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    cudaDeviceGetSharedMemConfig(&pConfig);
    printf("---result with Bank Mode:%s\n", pConfig == 1 ? "4-Byte" : "8-Byte");

    /*
Device 0: NVIDIA GeForce RTX 3060
    Number of multiprocessor:                      28
    clock rate:                                    1777000
    Compute capability:                            8.6
    Total amount of global memory:                 12333376.00 KB
    Total amount of constant memory:               64.00 KB
    Total amount of shared memory per block:       48.00 KB
    Total amount of shared memory per MP:          100.00 KB
    Warp size:                                     32
    Maximum number of threads per block:           1024
    Maximum number of threads per multiprocessor:  1536
    Maximum number of warps per multiprocessor:    48
    Maximum Grid size:                             (2147483647,65535,65535)
    Maximum block dimension                :       (1024,1024,64)
     */
}

int main()
{
    query_device();

    return 0;
}