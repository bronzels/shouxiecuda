#include <cstdio>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include "common.cpph"

#include "helper_cuda.h"
#include "helper_functions.h"

#include "cblas.h"

using namespace std;

#include "cuda_common.cuh"

#define BDIM 32
#define M 3072*8
#define K 1024*8
#define N 2048*8
#define print_a true
/*
#define BDIM 16
#define M 15*BDIM
#define K 5*BDIM
#define N 20*BDIM
#define print_a true
*/

#define IPAD 2

template <class T>
void matmul_cpu(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k)
{
    for(int x = 0; x < m; x ++)
    {
        for(int y = 0; y < n; y ++)
        {
            float temp = 0;
            for(int x2y = 0; x2y < k; x2y ++)
            {
                temp += *(mat_a + k * x + x2y) * (*(mat_b + n * x2y + y));
            }
            *(mat_out + n * x + y) = temp;
        }
    }
}

template <class T>
void tranpose_matrix(T * input, T * output, int m, int n)
{
    for(int x = 0; x < m; x ++)
    {
        for(int y = 0; y < n; y ++)
        {
            *(output + m * y + x) = *(input + n * x + y);
        }
    }
}

template <class T>
void matmul_cpu_blas(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k) {
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, mat_a, k, mat_b, k, 0.0, mat_out, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, mat_a, k, mat_b, n, 0.0, mat_out, n);
}

template <class T>
void matmul_thrust_2loops_inner_product(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k) {
    thrust::host_vector<T> v_a_h(mat_a, mat_a + m * n);
    thrust::host_vector<T> v_b_h(mat_b, mat_b + k * n);
    thrust::device_vector<T> v_a_d;
    v_a_d = v_a_h;
    thrust::device_vector<T> v_b_d;
    v_b_d = v_b_h;
    thrust::device_vector<T> v_out_d(m * n, 0);

    thrust::device_vector<T> v_b_d_out(k * n);
    thrust_transpose(v_b_d, v_b_d_out, k, n);

    std::cout<<"start executing by the GPU thrust_2loops_inner_product"<<std::endl;
    clock_t time1,time2;
    time1 = clock();
    //for loop + inner_product
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            v_out_d[i*k + j] = thrust::inner_product(v_a_d.begin() + k * i, v_a_d.begin() + (i+1) * k, v_b_d.begin() + k * j, 0.0f);
        }
    }
    time2 = clock();
    std::cout<<"time spent executing by the GPU thrust_2loops_inner_product: "<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
}

template <class T>
struct Dp{
        T *A, *B;
        int m,n,k;
        Dp(T *_A, T *_B, int _m, int _n, int _k): A(_A), B(_B), m(_m), n(_n), k(_k) {
            std::cout<<"m:"<<m<<", n:"<<n<<", k:"<<k<<std::endl;
        };
        __host__ __device__ T operator()(size_t idx){
            T sum = 0.0f;
            int row = idx / n;
            int col = idx % n;
            //int col = idx - (row * k);
            for (int i = 0; i < k; i++)
                sum += A[k * row + i] * B[n * i + col];
            return sum;
        }
};
template struct Dp<float>;

template <class T>
void matmul_thrust_transform_struct(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k) {
    //thrust::host_vector<T> v_a_h(mat_a, mat_a + m * n);
    //thrust::host_vector<T> v_b_h(mat_b, mat_b + k * n);
    thrust::device_vector<T> v_a_d(mat_a, mat_a + m * n);
    //v_a_d = v_a_h;
    thrust::device_vector<T> v_b_d(mat_b, mat_b + k * n);
    //v_b_d = v_b_h;
    thrust::device_vector<T> v_out_d(m * n, 0);

    thrust::device_vector<T> v_b_d_trans(k * n);
    //thrust_transpose(v_b_d, v_b_d_trans, k, n);
    //trans     1.1     diff
    //notrans   0.25    same

    std::cout<<"start executing by the GPU thrust_transform_struct"<<std::endl;
    clock_t time1,time2;
    time1 = clock();
    // transform + struct
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(m*n),
            v_out_d.begin(),
            Dp<T>(thrust::raw_pointer_cast(v_a_d.data()), thrust::raw_pointer_cast(v_b_d.data()), m, n, k));
    time2 = clock();
    std::cout<<"time spent executing by the GPU thrust_transform_struct: "<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    //thrust::host_vector<T> v_out_h(m * n, 0);
    //v_out_h = v_out_d;
    int byte_size = m*n*sizeof(T);
    cudaMemcpy(mat_out, thrust::raw_pointer_cast(v_out_d.data()), byte_size, cudaMemcpyDeviceToHost);
    //memcpy(mat_out, v_out_h.data(), byte_size);
}

template <class T>
__global__ void matmul(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < m && y < n)
    {
        float tmp = 0;
        for( int i = 0; i < k; i++)
        {
            tmp += *(mat_a + k * x + i) * (*(mat_b + n * i + y));
        }
        *(mat_out + n * x + y) = tmp;
    }
}

template <class T>
__global__ void matmul_smem(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k)
{
    __shared__ float sA[BDIM][BDIM];
    __shared__ float sB[BDIM][BDIM];

    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < m && y < n)
    {
        float tmp = 0.;
        for (int i = 0; i < k/BDIM; i ++)
        {
            sA[tx][ty] = *(mat_a + k * x + ty + i * BDIM);
            sB[tx][ty] = *(mat_b + n * (tx + i * BDIM) + y);
            __syncthreads();
            for (int j = 0; j < BDIM; j ++)
                tmp += sA[tx][j] * sB[j][ty];
            __syncthreads();
        }
        *(mat_out + n * x + y) = tmp;
    }
}

template <class T>
__global__ void matmul_smem_pad(T * mat_a, T * mat_b, T * mat_out, int m, int n, int k)
{
    __shared__ float sA[BDIM][BDIM+1];
    __shared__ float sB[BDIM][BDIM+1];

    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < m && y < n)
    {
        float tmp = 0.;
        for (int i = 0; i < k/BDIM; i ++)
        {
            sA[tx][ty] = *(mat_a + k * x + ty + i * BDIM);
            sB[tx][ty] = *(mat_b + n * (tx + i * BDIM) + y);
            __syncthreads();
            for (int j = 0; j < BDIM; j ++)
                tmp += sA[tx][j] * sB[j][ty];
            __syncthreads();
        }
        *(mat_out + n * x + y) = tmp;
    }
}

int main(int argc, char** argv)
{
	//default values for variabless
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

	printf("Matmul for (%d X % d) and (%d X % d) matrix with block size %d X %d \n",m,k,k,n,block_x,block_y);

    float * h_mat_array_a = (float*)malloc(byte_size_a);
    float * h_mat_array_b = (float*)malloc(byte_size_b);
    float * h_mat_array_mul = (float*)malloc(byte_size);
    float * h_ref = (float*)malloc(byte_size);

	//initialize matrix with integers between one and ten
    initialize(h_mat_array_a,size_a ,INIT_RANDOM);
    initialize(h_mat_array_b,size_b ,INIT_RANDOM);

    clock_t t_start, t_stop;
    /*
	//matirx transpose in CPU
    //printf("Start processing in CPU\n");
    t_start = clock();
    //matmul_cpu(h_mat_array_a, h_mat_array_b, h_mat_array_mul, m, n, k);
    t_stop = clock();
    //printf("CPU time: %f \n", (double)((double)(t_stop - t_start)/CLOCKS_PER_SEC));
    if ( print_a)
    {
        print_matrix(h_mat_array_mul, 2, 10);
        print_matrix(h_mat_array_mul + (m - 2) * n, 2, 10);
    }
    */

    //float * h_mat_array_b_t = (float*)malloc(byte_size_b);
    float * h_mat_array_out = (float*)malloc(byte_size);
    //tranpose_matrix(h_mat_array_b, h_mat_array_b_t, k, n);
    t_start = clock();
    matmul_cpu_blas(h_mat_array_a, h_mat_array_b, h_mat_array_out, m, n, k);
    t_stop = clock();
    printf("BLAS time: %f \n", (double)((double)(t_stop - t_start)/CLOCKS_PER_SEC));
    //printf("Compare CPU with %s\n", "openBLAS");
    //compare_matrixes(h_mat_array_out, h_mat_array_mul, m, n);
    if ( print_a)
    {
        print_matrix(h_mat_array_out, 2, 10);
        print_matrix(h_mat_array_out + (m - 2) * n, 2, 10);
    }
    //return 0;

    float * d_mat_array, *d_trans_array;

    float *d_mat_array_a, *d_mat_array_b, *d_mat_array_mul;
    cudaMalloc((void**)&d_mat_array_a, byte_size_a);
	cudaMalloc((void**)&d_mat_array_b, byte_size_b);
    cudaMalloc((void**)&d_mat_array_mul, byte_size);

	cudaMemcpy(d_mat_array_a, h_mat_array_a, byte_size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_array_b, h_mat_array_b, byte_size_b, cudaMemcpyHostToDevice);

	dim3 blocks(block_x, block_y);
	dim3 grid(m/block_x, n/block_y);

    void(*kernel)(float*, float*, float*, int, int, int);
	char * kernel_name;

    float * h_ref_prev = (float*)malloc(byte_size);
    memset((void *)h_ref_prev, 0, byte_size);
    for (int kernel_num = 3; kernel_num < 4; kernel_num ++)
    {
        switch (kernel_num)
        {
            case 0:
                kernel = &matmul;
                kernel_name = "GPU(global memory)";
                break;
            case 1:
                kernel = &matmul_smem;
                kernel_name = "GPU(shared memory)";
                break;
            case 2:
                kernel = &matmul_smem_pad;
                kernel_name = "GPU(shared memory padded)";
                break;
            case 3:
                kernel_name = "GPU(cuBLAS)";
                break;
        }

        printf("Start processing in %s\n", kernel_name);

        cudaMemset((void *)d_mat_array_mul, 0, byte_size);

        cudaEvent_t start, end;
        float time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        if (kernel_num != 3)
        {
            cudaEventRecord(start);
            kernel <<< grid, blocks>>> (d_mat_array_a, d_mat_array_b, d_mat_array_mul, m, n, k);
        }
        else
        {
            cublasHandle_t handle;
            cublasStatus_t status = cublasCreate(&handle);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "!!!! CUBLAS initialization error\n";
                return EXIT_FAILURE;
            }
            float alpha = 1.0f;
            float beta = 0.0f;
            cudaEventRecord(start);
            status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_mat_array_b, n, d_mat_array_a, k, &beta, d_mat_array_mul, n);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "!!!! CUBLAS kernel execution error\n";
                return EXIT_FAILURE;
            }
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        printf("%s time: %f\n", kernel_name, time/1000);

        cudaDeviceSynchronize();

        //copy the transpose memroy back to cpu
        memset((void *)h_ref, 0, byte_size);
        cudaMemcpy(h_ref, d_mat_array_mul, byte_size, cudaMemcpyDeviceToHost);
        if ( print_a)
        {
            print_matrix(h_ref, 2, 10);
            print_matrix(h_ref + (m - 2) * n, 2, 10);
        }

        //compare the CPU and GPU transpose matrix for validity
        printf("Compare CPU with %s\n", kernel_name);
        //compare_matrixes(h_ref, h_mat_array_mul, m, n);
        compare_matrixes(h_ref, h_mat_array_out, m, n, 1e-3);
        /*
        if( kernel_num != 0)
        {
            compare_matrixes(h_ref, h_ref_prev, m, n);
        }
        if( kernel_num != 2)
            memcpy(h_ref_prev, h_ref, byte_size);
        */
    }
    /*
    memset(h_mat_array_out, 0, size);
    matmul_thrust_2loops_inner_product(h_mat_array_a,  h_mat_array_b, h_mat_array_out, m, n, k);
    if ( print_a)
    {
        print_matrix(h_mat_array_out, 2, 10);
        print_matrix(h_mat_array_out + (m - 2) * n, 2, 10);
    }
    printf("Compare GPU(kernel,cublas) with matmul_thrust_2loops_inner_product:\n");
    compare_matrixes(h_ref, h_mat_array_out, m, n, 1e-3);
    */

    memset(h_ref, 0, byte_size);
    matmul_thrust_transform_struct(h_mat_array_a,  h_mat_array_b, h_ref, m, n, k);
    if ( print_a)
    {
        print_matrix(h_ref, 2, 10);
        print_matrix(h_ref + (m - 2) * n, 2, 10);
    }
    printf("Compare CPU with matmul_thrust_transform_struct:\n");
    compare_matrixes(h_ref, h_mat_array_out, m, n, 1e-3);

    free(h_mat_array_a);
    free(h_mat_array_b);
    free(h_mat_array_mul);
    free(h_ref);
    cudaFree(d_mat_array_a);
    cudaFree(d_mat_array_b);
    cudaFree(d_mat_array_mul);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}

/*
Matmul for (3072 X  1024) and (1024 X  2048) matrix with block size 16 X 16
Start processing in CPU
CPU time: 13.920000
Start processing in GPU(global memory)
GPU(global memory) time: 0.072272
Compare CPU with GPU(global memory)
Matrics are same
Start processing in GPU(shared memory)
GPU(shared memory) time: 0.051916
Compare CPU with GPU(shared memory)
Matrics are same
Start processing in GPU(shared memory padded)
GPU(shared memory padded) time: 0.024310
Compare CPU with GPU(shared memory padded)
Matrics are same


Matmul for (24576 X  8192) and (8192 X  16384) matrix with block size 32 X 32
BLAS time: 115.130000
Start processing in GPU(global memory)
GPU(global memory) time: 65.640182
Compare CPU with GPU(global memory)
Matrics are same
Start processing in GPU(shared memory)
GPU(shared memory) time: 74.491241
Compare CPU with GPU(shared memory)
Matrics are same
Start processing in GPU(shared memory padded)
GPU(shared memory padded) time: 13.259836
Compare CPU with GPU(shared memory padded)
Matrics are same


Matmul for (24576 X  8192) and (8192 X  16384) matrix with block size 32 X 32
BLAS time: 115.260000
Start processing in GPU(cuBLAS)
GPU(cuBLAS) time: 1.396380
Compare CPU with GPU(cuBLAS)
Matrics are same


matrixes: 3072 * 1024, 1024 * 2048; block: 16 * 16
                    cpu                     gpu                     gpu_smem                   gpu_smem_padded          thrust
python              23.659889698028564      0.24089908599853516     0.21659564971923828        0.21481561660766602
c++                 13.920000               0.072272                0.051916                   0.024310                 0.25

matrixes: 3072*8 * 1024*8 , 1024*8 * 2048*8; block: 32 * 32
            cpu(python:numpy, c++:openBLAS)       gpu                   gpu_smem              gpu_smem_padded           cuBLAS          pycuda(smem_padded)     cupy                       thrust
python      46.0459623336792                      68.0925920009613      88.5304069519043      78.57171964645386                         13.312400390625001      0.5030193328857422
c++         115.130000(不转置113.600000)           65.640182             74.491241             13.259836                 0.729070                                                           114.42


                            global精度        smem精度
个位数（3,1,2）*TPB          1e-6             1e-5
个位数（3,1,2）*64*TPB       1e-4             1e-4
个位数（3,1,2）*64*8*TPB     1e-4             1e-4

个位数（3,1,2）*64*TPB   openBLAS和cpu相比精度    1e-3


nsys nvprof --print-gpu-trace cmake-build-debug/misc_matmul 
[3/3] Executing 'cuda_gpu_trace' stats report

   Start (ns)    Duration (ns)  CorrId  GrdX   GrdY  GrdZ  BlkX  BlkY  BlkZ  Reg/Trd  StcSMem (MB)  DymSMem (MB)  Bytes (MB)  Throughput (MBps)  SrcMemKd  DstMemKd            Device             Ctx  Strm                                      Name                                     
 --------------  -------------  ------  -----  ----  ----  ----  ----  ----  -------  ------------  ------------  ----------  -----------------  --------  --------  ---------------------------  ---  ----  -----------------------------------------------------------------------------
 14,533,415,915     85,411,644     107                                                                               805.306          8,858.370  Pageable  Device    NVIDIA GeForce RTX 3060 (0)    1     7  [CUDA memcpy Host-to-Device]                                                 
 14,618,912,140     56,616,575     108                                                                               536.871          9,126.806  Pageable  Device    NVIDIA GeForce RTX 3060 (0)    1     7  [CUDA memcpy Host-to-Device]                                                 
 14,860,423,533      4,709,456     109                                                                             1,610.613        341,449.900  Device              NVIDIA GeForce RTX 3060 (0)    1     7  [CUDA memset]                                                                
 15,450,657,108    726,902,735   1,113  1,536     8     1   256     1     1      216         0.000         0.049                                                     NVIDIA GeForce RTX 3060 (0)    1     7  void cutlass::Kernel<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)
 16,362,957,096    166,295,895   1,119                                                                             1,610.613          9,663.676  Device    Pageable  NVIDIA GeForce RTX 3060 (0)    1     7  [CUDA memcpy Device-to-Host]                                                 




*/