#include "cuda_runtime_api.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cub/cub.cuh>

#include "common.hpp"

int main()
{
    int num_items = 1 << 27;
    int byte_size = num_items * sizeof(int);

    int *h_keys_in = (int *)malloc(byte_size);
    initialize(h_keys_in, num_items, INIT_RANDOM_KEY);
    int *h_values_in = (int *)malloc(byte_size);
    initialize(h_values_in, num_items, INIT_ONE_TO_TEN);
    /*
    std::cout<<"keys of cub input:"<<std::endl;
    print_matrix(h_keys_in, 1, 10);
    print_matrix(h_keys_in + num_items / 2, 1, 10);
    print_matrix(h_keys_in + num_items - 10, 1, 10);
    std::cout<<"values of cub input:"<<std::endl;
    print_matrix(h_values_in, 1, 10);
    print_matrix(h_values_in + num_items / 2, 1, 10);
    print_matrix(h_values_in + num_items - 10, 1, 10);
    */
    std::vector<int> v(h_keys_in, h_keys_in + num_items);
    std::set<int> s(v.begin(), v.end());
    std::cout<<"set size:"<<s.size()<<", num_items:"<<num_items<<std::endl;

    cudaEvent_t start, end;
    float time;

    int *h_keys_out = (int *)malloc(byte_size);
    int *h_values_out = (int *)malloc(byte_size);

    int *d_keys_in, *d_values_in;
    cudaMalloc((void**)&d_keys_in, byte_size);
    cudaMemcpy(d_keys_in, h_keys_in, byte_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_values_in, byte_size);
    cudaMemcpy(d_values_in, h_values_in, byte_size, cudaMemcpyHostToDevice);

    int *d_keys_out, *d_values_out;
    cudaMalloc((void**)&d_keys_out, byte_size);
    cudaMalloc((void**)&d_values_out, byte_size);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                              d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s sort time: %.8f\n", "cub GPU", time / 1000);
    cudaMemcpy(h_keys_out, d_keys_out, byte_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values_out, d_values_out, byte_size, cudaMemcpyDeviceToHost);
    std::cout<<"keys of cub sort:"<<std::endl;
    print_matrix(h_keys_out, 1, 10);
    print_matrix(h_keys_out + num_items / 2, 1, 10);
    print_matrix(h_keys_out + num_items - 10, 1, 10);
    std::cout<<"values of cub sort:"<<std::endl;
    print_matrix(h_values_out, 1, 10);
    print_matrix(h_values_out + num_items / 2, 1, 10);
    print_matrix(h_values_out + num_items - 10, 1, 10);

    thrust::device_vector<int> dv_keys_in(h_keys_in, h_keys_in + num_items);
    thrust::device_vector<int> dv_values_in(h_values_in, h_values_in + num_items);
    time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    thrust::sort_by_key(dv_keys_in.begin(), dv_keys_in.end(), dv_values_in.begin());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s sort time: %.8f\n", "thrust GPU", time / 1000);
    thrust::host_vector<int> hv_keys_in = dv_keys_in;
    thrust::host_vector<int> hv_values_in = dv_values_in;
    int *ptr_hv_keys_in = thrust::raw_pointer_cast(hv_keys_in.data());
    int *ptr_hv_values_in = thrust::raw_pointer_cast(hv_values_in.data());
    std::cout<<"keys of thrust sort:"<<std::endl;
    print_matrix(ptr_hv_keys_in, 1, 10);
    print_matrix(ptr_hv_keys_in + num_items / 2, 1, 10);
    print_matrix(ptr_hv_keys_in + num_items - 10, 1, 10);
    std::cout<<"values of thrust sort:"<<std::endl;
    print_matrix(ptr_hv_values_in, 1, 10);
    print_matrix(ptr_hv_values_in + num_items / 2, 1, 10);
    print_matrix(ptr_hv_values_in + num_items - 10, 1, 10);

    std::cout<<"Compare keys of cub sort with thrust:"<<std::endl;
    compare_arrays(ptr_hv_keys_in, h_keys_out, num_items);
    std::cout<<"Compare values of cub sort with thrust:"<<std::endl;
    compare_arrays(ptr_hv_values_in, h_values_out, num_items);

    return 0;
}
/*
                1<<10           1<<18           1<<22           1<<25           1<<27
cub             0.00441014      0.00993398      0.05933242      0.49922976
thrust          0.00271872      0.00636301      0.05689418      0.48491171



set size:1024, num_items:1024
cub GPU sort time: 0.00441014
keys of cub sort:
0 1 2 3 4 5 6 7 8 9

512 513 514 515 516 517 518 519 520 521

1014 1015 1016 1017 1018 1019 1020 1021 1022 1023

values of cub sort:
9 1 2 2 7 1 3 7 1 5

5 3 6 8 7 6 3 6 1 8

5 3 8 1 7 3 1 9 6 0

thrust GPU sort time: 0.00271872
keys of thrust sort:
0 1 2 3 4 5 6 7 8 9

512 513 514 515 516 517 518 519 520 521

1014 1015 1016 1017 1018 1019 1020 1021 1022 1023

values of thrust sort:
9 1 2 2 7 1 3 7 1 5

5 3 6 8 7 6 3 6 1 8

5 3 8 1 7 3 1 9 6 0

Compare keys of cub sort with thrust:
Arrays are same
Compare values of cub sort with thrust:
Arrays are same

 */
