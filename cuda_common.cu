#include "cuda_common.cuh"
#include "common.h"

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
    compare_arrays(h_out_array, h_trans_array, size, 1e-4);
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