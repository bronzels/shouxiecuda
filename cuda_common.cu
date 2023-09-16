#include "cuda_common.cuh"
#include "common.cpph"

void checkError(cl_int error, int line) {
    if (error != CL_SUCCESS) {
        switch (error) {
            case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
            case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
            case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
            case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
            case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
            case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
            case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
            case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
            case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
            case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
            case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
            case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
            case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
            case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
            case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
            case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
            case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
            case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
            case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
            case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
            case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
            case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
            case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
            case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
            case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
            case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
            case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
            case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
            case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
            case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
            case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
            case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
            case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
            case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
            case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
            case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
            case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
            case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
            case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
            case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
            case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
            case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
            case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
            case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
            case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
            case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
            case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
            case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
            case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
            case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
            case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
            case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
            case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
            case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
            case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
            case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
            case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
            case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
            case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
            case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
            case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
            default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
        }
        exit(1);
    }
}

void ShowCudaGpuInfo()
{
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);

    std::printf("CUDA : cudaGetDeviceCount : number of CUDA devices:\t%d\n", num_gpus);
}

struct transpose_index : public thrust::unary_function<size_t, size_t>
{
    size_t m, n;

    __host__ __device__
    transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

    __host__ __device__
    size_t operator()(size_t linear_index)
    {
        size_t i = linear_index / n;
        size_t j = linear_index % n;
        return m* j + i;
    }
};

template <class T>
void thrust_transpose(T *h_mat_array, T *h_trans_array, int nx, int ny)
{
    int size = nx * ny;
    thrust::host_vector<T> v_mat_h(h_mat_array, h_mat_array + nx * ny);
    thrust::device_vector<T> v_mat_d;
    v_mat_d = v_mat_h;
    thrust::device_vector<T> v_out_d(size);

    thrust::counting_iterator<size_t> indices(0);

    clock_t time1,time2;

    time1 = clock();
    thrust::gather(
            thrust::make_transform_iterator(indices, transpose_index(nx, ny)),
            thrust::make_transform_iterator(indices, transpose_index(nx, ny)) + size,
            v_mat_d.begin(),
            v_out_d.begin()
    );
    time2 = clock();
    std::cout<<"time spent by the GPU thrust transpose: "<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    thrust::host_vector<T> v_out_h;
    v_out_h = v_out_d;
    //memcpy(h_trans_array, v_trans_h.data(), v_trans_h.size()*sizeof(int));
    float * h_out_array = v_out_h.data();
    compare_arrays(h_out_array, h_trans_array, size, float(1e-4));
    print_array(h_out_array + size/2, 16);
}
template void thrust_transpose(float *h_mat_array, float *h_trans_array, int nx, int ny);

template <class T>
void thrust_transpose(thrust::device_vector<T> &v_mat_d, thrust::device_vector<T> &v_out_d, int nx, int ny)
{
    int size = nx * ny;
    thrust::counting_iterator<size_t> indices(0);

    clock_t time1,time2;

    time1 = clock();
    thrust::gather(
            thrust::make_transform_iterator(indices, transpose_index(nx, ny)),
            thrust::make_transform_iterator(indices, transpose_index(nx, ny)) + size,
            v_mat_d.begin(),
            v_out_d.begin()
    );
    time2 = clock();
    std::cout<<"time spent by the GPU thrust transpose: "<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
}
template
void thrust_transpose(thrust::device_vector<float> &v_mat, thrust::device_vector<float> &v_mat_out, int nx, int ny);

//void query_device()
//{
//	int iDev = 0;
//	cudaDeviceProp iProp;
//	cudaGetDeviceProperties(&iProp, iDev);
//
//	printf("Device %d: %s\n", iDev, iProp.name);
//	printf("  Number of multiprocessors:                     %d\n",
//		iProp.multiProcessorCount);
//	//printf("  Number of multiprocessors:                     %d\n",
//	//	iProp.);
//	printf("  Compute capability       :                     %d.%d\n",
//		iProp.major,iProp.minor);
//	printf("  Total amount of global memory:                 %4.2f KB\n",
//		iProp.totalGlobalMem/ 1024.0);
//	printf("  Total amount of constant memory:               %4.2f KB\n",
//		iProp.totalConstMem / 1024.0);
//	printf("  Total amount of shared memory per block:       %4.2f KB\n",
//		iProp.sharedMemPerBlock / 1024.0);
//	printf("  Total amount of shared memory per MP:          %4.2f KB\n",
//		iProp.sharedMemPerMultiprocessor / 1024.0);
//	printf("  Total number of registers available per block: %d\n",
//		iProp.regsPerBlock);
//	printf("  Warp size:                                     %d\n",
//		iProp.warpSize);
//	printf("  Maximum number of threads per block:           %d\n",
//		iProp.maxThreadsPerBlock);
//	printf("  Maximum number of threads per multiprocessor:  %d\n",
//		iProp.maxThreadsPerMultiProcessor);
//	printf("  Maximum number of warps per multiprocessor:    %d\n",
//		iProp.maxThreadsPerMultiProcessor / 32);
//	printf("  Maximum Grid size                         :    (%d,%d,%d)\n",
//		iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
//	printf("  Maximum block dimension                   :    (%d,%d,%d)\n",
//		iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
//}