#include <cstdio>
#include <iostream>

#include "cuda_runtime_api.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "common.hpp"

#define MAX_DEPTH 16
#define INSERTION_SORT 128

#define GPU_CHECK(ans)                                                         \
    { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
};

__device__ void selection_sort(unsigned int *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        unsigned min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j) {
            unsigned val_j = data[j];

            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right,
                                     int depth) {
    // 当递归的深度大于设定的MAX_DEPTH或者待排序的数组长度小于设定的阈值，直接调用简单选择排序
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *left_ptr = data + left;
    unsigned int *right_ptr = data + right;
    unsigned int pivot = data[(left + right) / 2];
    // partition
    while (left_ptr <= right_ptr) {
        unsigned int left_val = *left_ptr;
        unsigned int right_val = *right_ptr;

        while (left_val < pivot) { // 找到第一个比pivot大的
            left_ptr++;
            left_val = *left_ptr;
        }

        while (right_val > pivot) { // 找到第一个比pivot小的
            right_ptr--;
            right_val = *right_ptr;
        }

        // do swap
        if (left_ptr <= right_ptr) {
            *left_ptr++ = right_val;
            *right_ptr-- = left_val;
        }
    }

    // recursive
    int n_right = right_ptr - data;
    int n_left = left_ptr - data;
    // Launch a new block to sort the the left part.
    if (left < (right_ptr - data)) {
        cudaStream_t l_stream;
        // 设置非阻塞流
        cudaStreamCreateWithFlags(&l_stream, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, l_stream>>>(data, left, n_right,
                                                    depth + 1);
        cudaStreamDestroy(l_stream);
    }

    // Launch a new block to sort the the right part.
    if ((left_ptr - data) < right) {
        cudaStream_t r_stream;
        // 设置非阻塞流
        cudaStreamCreateWithFlags(&r_stream, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, r_stream>>>(data, n_left, right,
                                                    depth + 1);
        cudaStreamDestroy(r_stream);
    }
}

// Call the quicksort kernel from the host.
void run_qsort(unsigned int *data, unsigned int nitems) {
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    GPU_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    int left = 0;
    int right = nitems - 1;
    cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
    GPU_CHECK(cudaDeviceSynchronize());
}

// Initialize data on the host.
void initialize_data(unsigned int *dst, unsigned int nitems) {
    // Fixed seed for illustration
    srand(2047);

    // Fill dst with random values
    for (unsigned i = 0; i < nitems; i++)
        dst[i] = rand() % nitems;
}

// Verify the results.
void print_results(int n, unsigned int *results_d) {
    unsigned int *results_h = new unsigned[n];
    GPU_CHECK(cudaMemcpy(results_h, results_d, n * sizeof(unsigned),cudaMemcpyDeviceToHost));
    std::cout << "Sort data : ";
    for (int i = 1; i < n; ++i){
        std::cout << results_h[i] << " ";
        // if (results_h[i - 1] > results_h[i]) {
        //     std::cout << "Invalid item[" << i - 1 << "]: " << results_h[i - 1]
        //               << " greater than " << results_h[i] << std::endl;
        //     exit(EXIT_FAILURE);
        // }
    }
    std::cout << std::endl;
    delete[] results_h;
}

void check_results(int n, unsigned int *results_d)
{
    unsigned int *results_h = new unsigned[n];
    GPU_CHECK(cudaMemcpy(results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int i = 1 ; i < n ; ++i)
        if (results_h[i-1] > results_h[i])
        {
            std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
            exit(EXIT_FAILURE);
        }

    std::cout << "OK" << std::endl;
    delete[] results_h;
}

// Main entry point.

void sort_thrust(int num_items, int samples, unsigned int *h_data, unsigned int *h_sorted_data) {
    int byte_size = num_items * sizeof(unsigned int);
    //void (*methodfunc)(std::vector<int>&, std::vector<int>&);
    char *method_name = "Thrust GPU";
    std::vector<unsigned int> vec_out(num_items);
    clock_t time1,time2;
    time1 = clock();
    thrust::device_vector<unsigned int> d_vec(h_data, h_data+num_items);
    thrust::sort(d_vec.begin(), d_vec.end());
    unsigned int *d_data = thrust::raw_pointer_cast(d_vec.data());
    cudaMemcpy(h_sorted_data, d_data, byte_size, cudaMemcpyDeviceToHost);
    //thrust::copy(d_vec.begin(), d_vec.end(), vec_out.begin());
    time2 = clock();

    std::cout<< method_name <<" data: "<<std::endl;
    std::cout << "Validating results: ";
    check_results(num_items, thrust::raw_pointer_cast(d_vec.data()));

    std::cout<< method_name <<" sort time: ";
    printf("%.8f\n", (double)(time2-time1)/CLOCKS_PER_SEC);

    // print result
    print_results(samples, d_data);
    print_results(samples, d_data + num_items - samples);
}

void sort_cuda(int num_items, int samples, unsigned int *h_data, unsigned int *h_sorted_data) {
    // Find/set device and get device properties
    int device = 0;
    cudaDeviceProp deviceProp;
    GPU_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    if (!(deviceProp.major > 3 ||
          (deviceProp.major == 3 && deviceProp.minor >= 5))) {
        printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.",
               device, deviceProp.name);
        return;
    }
    int byte_size = num_items * sizeof(unsigned int);
    // Allocate GPU memory.
    unsigned int *d_data = 0;
    GPU_CHECK(cudaMalloc((void **)&d_data, byte_size));
    GPU_CHECK(cudaMemcpy(d_data, h_data, byte_size, cudaMemcpyHostToDevice));

    // Execute
    std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    cudaEvent_t start, end;
    float time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    run_qsort(d_data, num_items);
    cudaDeviceSynchronize();
    GPU_CHECK(cudaMemcpy(h_sorted_data, d_data, num_items * sizeof(unsigned int),cudaMemcpyDeviceToHost));
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("%s sort time: %.8f\n", "linghu CUDA GPU sort", time / 1000);

    // print result
    print_results(samples, d_data);
    print_results(samples, d_data + num_items - samples);
    // check result
    std::cout << "Validating results: ";
    check_results(num_items, d_data);
    GPU_CHECK(cudaFree(d_data));
}

int main(int argc, char **argv) {
    int num_items = 1 << 18;
    int samples = 18;
    bool verbose = true;


    // Create input data
    unsigned int *h_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
    h_data = (unsigned int *)malloc(num_items * sizeof(unsigned int));
    initialize_data(h_data, num_items);

    unsigned int *h_sorted_data_cuda = (unsigned int *)malloc(num_items * sizeof(unsigned int));
    unsigned int *h_sorted_data_thrust = (unsigned int *)malloc(num_items * sizeof(unsigned int));

    if (verbose) {
        std::cout << "Raw  data : ";
        for (int i = 0; i < samples; i++)
            std::cout << h_data[i] << " ";
        for (int i = num_items - samples; i < num_items; i++)
            std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    sort_thrust(num_items, samples, h_data, h_sorted_data_thrust);
    sort_cuda(num_items, samples, h_data, h_sorted_data_cuda);

    std::cout << "Compare CUDA sort result with Thrust: ";
    compare_arrays(h_sorted_data_cuda, h_sorted_data_thrust, num_items);

    free(h_data);
    free(h_sorted_data_thrust);
    free(h_sorted_data_cuda);

    return 0;
}

/*
编译：nvcc -o quicksort_cuda --gpu-architecture=sm_70 -rdc=truequicksort_cuda.cu


Initializing data:
Raw  data : 254 96776 232222 184566 4753 75822 74653 123544 96167 115970 156418 124307 7575 244344 106981 206481 23233 189718 29597 53050 202579 210092 116966 180468 161171 24656 65446 174593 57890 214194 129017 192694 112306 124746 67898 218070
Thrust GPU data:
Validating results: OK
Thrust GPU sort time: 0.01000000
Sort data : 2 2 3 5 6 6 7 7 8 8 9 9 10 10 12 13 13
Sort data : 262127 262128 262130 262131 262132 262132 262132 262134 262136 262136 262137 262137 262138 262139 262139 262141 262141
Running quicksort on 262144 elements
linghu CUDA GPU sort sort time: 2.20191288
Sort data : 2 2 3 5 6 6 7 7 8 8 9 9 10 10 12 13 13
Sort data : 262127 262128 262130 262131 262132 262132 262132 262134 262136 262136 262137 262137 262138 262139 262139 262141 262141
Validating results: OK
Compare CUDA sort result with Thrust: Arrays are same


                2<<18           2<<21           2<<23
thrust          0.01            0.05            0.04
cuda            2.20487499      212.51461792    时间太长放弃
*/
