#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime_api.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cub/cub.cuh>

#include "helper_cuda.h"
#include "helper_functions.h"

#include "common.hpp"

using namespace cub;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
    typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>       BlockLoadT;
    typedef cub::BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_TRANSPOSE>     BlockStoreT;
    typedef cub::BlockRadixSort<int,  BLOCK_THREADS, ITEMS_PER_THREAD>                        BlockRadixSortT;

    __shared__ union {
        typename BlockLoadT::TempStorage             load;
        typename BlockStoreT::TempStorage            store;
        typename BlockRadixSortT::TempStorage        sort;
    } temp_storage;

    int thread_keys[ITEMS_PER_THREAD];
    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

    BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);
    //__syncthreads();

    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);
    //__syncthreads();

    BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void shared_BlockSortKernel(int *d_in, int *d_out)
{
    __shared__ int sharedMemoryArray[BLOCK_THREADS * ITEMS_PER_THREAD];

    typedef cub::BlockRadixSort <int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

    int thread_offset = threadIdx.x * ITEMS_PER_THREAD;
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        int element_offset = thread_offset + k;
        sharedMemoryArray[element_offset] = d_in[block_offset + element_offset];
    }
    __syncthreads();

    BlockRadixSortT(temp_storage).Sort(*static_cast<int(*)[ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryArray + thread_offset)));
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        d_out[block_offset + thread_offset + k] = sharedMemoryArray[thread_offset + k];
    }
}

int main()
{
    //每个都增加或减少2（29时超过shared size）
    /*
    const int numElemsPerArray      = 1 << 7;
    const int numArrays             = 1 << 6;
    const int N                     = numElemsPerArray * numArrays;
    const int numElemsPerThread     = 1 << 2;
     */
    const int numElemsPerArray      = 1 << 13;
    const int numArrays             = 1 << 12;
    const int N                     = numElemsPerArray * numArrays;
    const int numElemsPerThread     = 1 << 8;
    std::cout<<"numElemsPerArray:"<<numElemsPerArray<<", numArrays:"<<numArrays<<", N:"<<N<<", numElemsPerThread:"<<numElemsPerThread<<std::endl;

    thrust::device_vector<int> dv_keys(N);

    thrust::transform(  thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(N),
                        thrust::make_constant_iterator(numElemsPerArray),
                        dv_keys.begin(),
                        thrust::divides<int>() );

    thrust::host_vector<int> hv_keys = dv_keys;
    int * ptr_keys = thrust::raw_pointer_cast(hv_keys.data());
    print_matrix(ptr_keys, 1, 10);
    print_matrix(ptr_keys + N / 2, 1, 10);
    print_matrix(ptr_keys + N - 10, 1, 10);

    size_t byte_size = N * sizeof(int);

    int *h_data = (int *)malloc(byte_size);
    initialize(h_data, N, INIT_RANDOM_KEY);

    int *h_result_thrust = (int *)malloc(byte_size);
    int *h_result_cub = (int *)malloc(byte_size);

    int *d_in; checkCudaErrors(cudaMalloc(&d_in, byte_size));
    int *d_out; checkCudaErrors(cudaMalloc(&d_out, byte_size));

    checkCudaErrors(cudaMemcpy(d_in, h_data, byte_size, cudaMemcpyHostToDevice));
    cudaEvent_t start, end;
    float time;

    thrust::device_vector<int> dv_in(d_in, d_in+N);

    time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    thrust::stable_sort_by_key(dv_in.begin(), dv_in.end(), dv_keys.begin(), thrust::less<int>());
    thrust::stable_sort_by_key(dv_keys.begin(), dv_keys.end(), dv_in.begin(), thrust::less<int>());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s sort time: %.8f\n", "thrust GPU", time / 1000);
    checkCudaErrors(cudaMemcpy(h_result_thrust, thrust::raw_pointer_cast(dv_in.data()), byte_size, cudaMemcpyDeviceToHost));
    std::cout<<"thrust sort result:"<<std::endl;
    print_matrix(h_result_thrust, 1, 10);
    print_matrix(h_result_thrust + N / 2, 1, 10);
    print_matrix(h_result_thrust + N - 10, 1, 10);

    time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    BlockSortKernel<(int)N/numArrays/numElemsPerThread, numElemsPerThread><<<numArrays, numElemsPerArray/numElemsPerThread>>>(d_in, d_out);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s sort time: %.8f\n", "cub GPU", time / 1000);
    checkCudaErrors(cudaMemcpy(h_result_cub, d_out, byte_size, cudaMemcpyDeviceToHost));
    std::cout<<"cub sort result:"<<std::endl;
    print_matrix(h_result_cub, 1, 10);
    print_matrix(h_result_cub + N / 2, 1, 10);
    print_matrix(h_result_cub + N - 10, 1, 10);
    compare_arrays(h_result_cub, h_result_thrust, N);

    time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s sort time: %.8f\n", "cub smem GPU", time / 1000);
    checkCudaErrors(cudaMemcpy(h_result_cub, d_out, byte_size, cudaMemcpyDeviceToHost));
    BlockSortKernel<(int)N/numArrays/numElemsPerThread, numElemsPerThread><<<numArrays, numElemsPerArray/numElemsPerThread>>>(d_in, d_out);
    std::cout<<"cub smem sort result:"<<std::endl;
    print_matrix(h_result_cub, 1, 10);
    print_matrix(h_result_cub + N / 2, 1, 10);
    print_matrix(h_result_cub + N - 10, 1, 10);
    compare_arrays(h_result_cub, h_result_thrust, N);

    free(h_data);
    free(h_result_thrust);
    free(h_result_cub);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
/*
                1<<13           1<<17           1<<21           1<<25
thrust          0.01014813      0.01000778      0.05821849      0.82925123
cub             0.00099411      0.00214826      0.02573728      1.55115187
cub smem        0.00000170      0.00000131      0.00000173      0.00000125



 */
