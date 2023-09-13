#include <cstdio>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/gather.h>

#include "helper_cuda.h"
#include "helper_functions.h"

#include "common.h"
#include "cuda_common.cuh"

using namespace std;

#define NX 1024*8
//1024//32//
#define NY 1024*8
//1024//8//
#define BDIMX 64
//128//4//4//8//8//4//32//16//64//8//128
#define BDIMY 16
//32//8//32//64//256//32//64//16//128
#define IPAD 2

template <class T>
__global__ void copy_row(T * mat, T * transpose, int nx, int ny)
{
    int ix = blockDim.x * blockDim.x + threadIdx.x;
    int iy = blockDim.y * blockDim.y + threadIdx.y;

    if ( ix < nx && iy < ny)
    {
        transpose[nx * iy + ix] = mat[nx * iy + ix];
    }
}

template <class T>
__global__ void copy_column(T * mat, T * transpose, int nx, int ny)
{
    int ix = blockDim.x * blockDim.x + threadIdx.x;
    int iy = blockDim.y * blockDim.y + threadIdx.y;

    if ( ix < nx && iy < ny)
    {
        transpose[ny * ix + iy] = mat[ny * ix + iy];
    }
}

template <class T>
__global__ void transpose_read_row_write_column(T * mat, T * transpose, int nx, int ny)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if ( ix < nx && iy < ny)
    {
        transpose[ny * ix + iy] = mat[nx * iy + ix];
    }
}

template <class T>
__global__ void transpose_read_column_write_row(T * mat, T * transpose, int nx, int ny)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if ( ix < nx && iy < ny)
    {
        transpose[nx * iy + ix] = mat[ny * ix + iy];
    }
}

template <class T>
__global__ void transpose_unroll4_row(T * mat, T * transpose, int nx, int ny)
{
    int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    int ti = iy * nx + ix;
    int to = ix * ny + iy;

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        transpose[to] = mat[ti];
        transpose[to + ny * blockDim.x] = mat[ti + blockDim.x];
        transpose[to + ny * blockDim.x * 2] = mat[ti + blockDim.x * 2];
        transpose[to + ny * blockDim.x * 3] = mat[ti + blockDim.x * 3];
    }
}

template <class T>
__global__ void transpose_unroll4_col(T * mat, T * transpose, int nx, int ny)
{
    int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    int ti = iy * nx + ix;
    int to = ix * ny + iy;

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        transpose[ti] = mat[to];
        transpose[ti + blockDim.x] = mat[to + ny * blockDim.x];
        transpose[ti + blockDim.x * 2] = mat[to + ny * blockDim.x * 2];
        transpose[ti + blockDim.x * 3] = mat[to + ny * blockDim.x * 3];
    }
}

template <class T>
__global__ void transpose_unroll4_col2(T * mat, T * transpose, int nx, int ny)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y * 4 + threadIdx.y;

    int ti = iy * nx + ix;
    int to = ix * ny + iy;

    if (ix < nx && iy + 3 * blockDim.y < ny)
    {
        transpose[to] = mat[ti];
        transpose[to + blockDim.y] = mat[ti + ny * blockDim.y];
        transpose[to + blockDim.y * 2] = mat[ti + ny * blockDim.y * 2];
        transpose[to + blockDim.y * 3] = mat[ti + ny * blockDim.y * 3];
    }
}

template <class T>
__global__ void transpose_diagonal_row(T * mat, T * transpose, int nx, int ny)
{
    int blk_x = blockIdx.x;
    int blk_y = (blockIdx.x + blockIdx.y) % gridDim.x;

    int ix = blockIdx.x * blk_x + threadIdx.x;
    int iy = blockIdx.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        transpose[ix * ny + iy] = mat[iy * nx + ix];
    }
}

template <class T>
__global__ void transpose_diagonal_row2(T * mat, T * transpose, int nx, int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if(ix < nx && iy < ny)
    {
        transpose[ix * ny + iy] = mat[iy * nx + ix];
    }
}

template <class T>
__global__ void transpose_diagonal_col(T * mat, T * transpose, int nx, int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if(ix < nx && iy < ny)
    {
        transpose[iy * nx + ix] = mat[ix * ny + iy];
    }
}

template <class T>
__global__ void transpose_smem_orig(T * mat, T * transpose, int nx, int ny)
{
    __shared__ int tile[BDIMY][BDIMX];

    int ix, iy, in_index;

    int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    in_index = iy * nx + ix;

    _1d_index = threadIdx.y * blockDim.x + threadIdx.x;

    i_row = _1d_index / blockDim.y;
    i_col = _1d_index % blockDim.y;

    out_ix = blockIdx.y * blockDim.y + i_col;
    out_iy = blockIdx.x * blockDim.x + i_row;

    out_index = out_iy * ny + out_ix;

    if (ix < nx && iy < ny)
    {
        tile[threadIdx.y][threadIdx.x] = mat[in_index];

        __syncthreads();

        transpose[out_index] = tile[i_col][i_row];
    }
}

template <class T>
__global__ void transpose_smem_orig_pad(T * mat, T * transpose, int nx, int ny)
{
    __shared__ int tile[BDIMY][BDIMX+IPAD];

    int ix, iy, in_index;

    int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    in_index = iy * nx + ix;

    _1d_index = threadIdx.y * blockDim.x + threadIdx.x;

    i_row = _1d_index / blockDim.y;
    i_col = _1d_index % blockDim.y;

    out_ix = blockIdx.y * blockDim.y + i_col;
    out_iy = blockIdx.x * blockDim.x + i_row;

    out_index = out_iy * ny + out_ix;

    if (ix < nx && iy < ny)
    {
        tile[threadIdx.y][threadIdx.x] = mat[in_index];

        __syncthreads();

        transpose[out_index] = tile[i_col][i_row];
    }
}

template <class T>
__global__ void transpose_smem_orig_unrolling(T * mat, T * transpose, int nx, int ny)
{
    __shared__ int tile[BDIMY][2*BDIMX];

    int ix, iy, in_index;

    int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

    ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    in_index = iy * nx + ix;

    _1d_index = threadIdx.y * blockDim.x + threadIdx.x;

    i_row = _1d_index / blockDim.y;
    i_col = _1d_index % blockDim.y;

    out_ix = blockIdx.y * blockDim.y + i_col;
    out_iy = 2 * blockIdx.x * blockDim.x + i_row;

    out_index = out_iy * ny + out_ix;

    if (ix < nx && iy < ny)
    {
        tile[threadIdx.y][threadIdx.x] = mat[in_index];
        tile[threadIdx.y][threadIdx.x + blockDim.x] = mat[in_index + blockDim.x];

        __syncthreads();

        transpose[out_index] = tile[i_col][i_row];
        transpose[out_index + ny * blockDim.x] = tile[i_col][i_row + blockDim.x];
    }
    /*
    if (threadIdx.x == 2 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 1)
    {
        printf("blockDim.x: %d, blockDim.y: %d, nx: %d, ny: %d\n", blockDim.x, blockDim.y, nx, ny);
        printf("ix: %d, iy: %d, in_index: %d, _1d_index: %d, i_row: %d, i_col: %d, out_ix: %d, out_iy: %d, out_index: %d\n", ix, iy, in_index, _1d_index, i_row, i_col, out_ix, out_iy, out_index);
        //printf("mat[in_index]: %d, tile[threadIdx.y][threadIdx.x]: %d, tile[i_col][i_row]: %d, transpose[out_index]: %d\n", mat[in_index], tile[threadIdx.y][threadIdx.x], tile[i_col][i_row], transpose[out_index]);
        printf("mat[in_index]: %d, tile[threadIdx.y][threadIdx.x]: %d, transpose[out_index]: %d\n", mat[in_index], tile[threadIdx.y][threadIdx.x], transpose[out_index]);
    }
    */
}

template <class T>
__global__ void transpose_smem_orig_pad_unrolling(T * mat, T * transpose, int nx, int ny)
{
    /* master
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    //input index
    int ix, iy, in_index;

    //output index
    int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

    //ix and iy calculation for input index
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    //input index
    in_index = iy * nx + ix;

    //1D index calculation fro shared memory
    _1d_index = threadIdx.y * blockDim.x + threadIdx.x;

    //col major row and col index calcuation
    i_row = _1d_index / blockDim.y;
    i_col = _1d_index % blockDim.y;

    //coordinate for transpose matrix
    out_ix = blockIdx.y * blockDim.y + i_col;
    out_iy = blockIdx.x * blockDim.x + i_row;

    //output array access in row major format
    out_index = out_iy * ny + out_ix;

    if (ix < nx && iy < ny)
    {
        //load from in array in row major and store to shared memory in row major
        tile[threadIdx.y][threadIdx.x] = mat[in_index];

        //wait untill all the threads load the values
        __syncthreads();

        transpose[out_index] = tile[i_col][i_row];
    }
    */
    /*  me
    */
    __shared__ int tile[BDIMY][2*BDIMX + IPAD];

    int ix, iy, in_index;

    int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

    ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    in_index = iy * nx + ix;

    _1d_index = threadIdx.y * blockDim.x + threadIdx.x;

    i_row = _1d_index / blockDim.y;
    i_col = _1d_index % blockDim.y;

    out_ix = blockIdx.y * blockDim.y + i_col;
    out_iy = 2 * blockIdx.x * blockDim.x + i_row;

    out_index = out_iy * ny + out_ix;

    if (ix < nx && iy < ny)
    {
        tile[threadIdx.y][threadIdx.x] = mat[in_index];
        tile[threadIdx.y][threadIdx.x + blockDim.x] = mat[in_index + blockDim.x];

        __syncthreads();

        transpose[out_index] = tile[i_col][i_row];
        transpose[out_index + ny * blockDim.x] = tile[i_col][i_row + blockDim.x];
    }
}

template <class T>
__global__ void transpose_smem(T * mat, T * transpose, int nx, int ny)
{
    __shared__ int tile[BDIMY][BDIMX];

    int ix, iy, in_index;

    int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    in_index = iy * nx + ix;

    _1d_index = threadIdx.y * blockDim.x + threadIdx.x;

    /*
    i_row = _1d_index / blockDim.y;
    i_col = _1d_index % blockDim.y;

    out_ix = blockIdx.y * blockDim.y + i_col;
    out_iy = blockIdx.x * blockDim.x + i_row;
    */
    out_ix = blockIdx.y * blockDim.y + threadIdx.y;
    out_iy = blockIdx.x * blockDim.x + threadIdx.x;

    out_index = out_iy * ny + out_ix;

    if (ix < nx && iy < ny)
    {
        int temp = mat[in_index];
        //tile[threadIdx.y][threadIdx.x] = temp;

        //__syncthreads();

        //transpose[out_index] = tile[i_col][i_row];
        transpose[out_index] = temp;
    }
    /*
    if (threadIdx.x == 2 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 1)
    {
        printf("blockDim.x: %d, blockDim.y: %d, nx: %d, ny: %d\n", blockDim.x, blockDim.y, nx, ny);
        printf("ix: %d, iy: %d, in_index: %d, _1d_index: %d, i_row: %d, i_col: %d, out_ix: %d, out_iy: %d, out_index: %d\n", ix, iy, in_index, _1d_index, i_row, i_col, out_ix, out_iy, out_index);
        //printf("mat[in_index]: %d, tile[threadIdx.y][threadIdx.x]: %d, tile[i_col][i_row]: %d, transpose[out_index]: %d\n", mat[in_index], tile[threadIdx.y][threadIdx.x], tile[i_col][i_row], transpose[out_index]);
        printf("mat[in_index]: %d, tile[threadIdx.y][threadIdx.x]: %d, transpose[out_index]: %d\n", mat[in_index], tile[threadIdx.y][threadIdx.x], transpose[out_index]);
    }
    */
    /*
#define NX 32
#define NY 8
#define BDIMX 4
#define BDIMY 2

time spent executing by the GPU: 0.190112
blockDim.x: 4, blockDim.y: 2, nx: 32, ny: 8
ix: 2, iy: 3, in_index: 98, _1d_index: 6, i_row: 3, i_col: 0, out_ix: 2, out_iy: 3, out_index: 26
mat[in_index]: 8, tile[threadIdx.y][threadIdx.x]: 8, tile[i_col][i_row]: 7, transpose[out_index]: 7

time spent executing by the GPU: 0.177120
blockDim.x: 4, blockDim.y: 2, nx: 32, ny: 8
ix: 2, iy: 3, in_index: 98, _1d_index: 6, i_row: 0, i_col: 0, out_ix: 3, out_iy: 2, out_index: 19
mat[in_index]: 8, tile[threadIdx.y][threadIdx.x]: 8, transpose[out_index]: 8
     */

}



/* 1024 * 1024
0       Copy row                            diff    0.030400
1       Copy column                         diff    0.029952
2       Read row write column               same    0.126464
3       Read column write row               same    0.076192
4       Unroll 4 row                        same    0.119264
5       Unroll 4 col                        same    0.065984        best
6       Diagonal row                        diff
7       Unroll 4 col 2                      same    0.127808
8       Diagonal row 2                      diff
9       Diagonal col                        diff
10      SMem orig                           same    0.062944
11      SMem orig pad                       same    0.060416
12      SMem orig unrolling                 same    0.058720/0.056896/0.059232/0.064416/0.062944
13      SMem orig pad unrolling(master)     same    0.058688/0.059968/0.058464/0.058272/0.060000
13      SMem orig pad unrolling(me)         same    0.052032/0.055616/0.055584/0.053568/0.053376        new best
11      SMem                                same    0.125024
        thrust                              same    0.02
        cublas                              same    7.1616e-05

                    kernel no.5   kernel no.13   thrust      cublas
1024*1024           0.065984      0.052032       0.02        7.1616e-05
1024*8*1024*8       5.371072      9.609568       1.73        0.00164365
1024*16*1024*16     21.688736     37.767040      6.9         0.00650122

！！！这2个核函数，block的大小超过64*16时就会失败，小于这个值都是成功的
12，我根据SMem orig改写的SMem orig pad
13，SMem orig pad unrolling，不管是原作者的一维smem版本，还是我改写的二维版本

Unroll 4 col，瘦块
128*8       0.065984
8*128       0.059072
64*16       0.066592
16*64       0.059520
32*32       0.064064
4*256       0.064736
8*64        0.050144
8*32        0.049792    best
4*32        0.053760
 */


void cublas_transpose(float *h_mat_array, float *h_trans_array, int nx, int ny) {
    int size = nx * ny;
    int byte_size = size * sizeof(float);
    float *h_out_array = (float *)malloc(byte_size);
    float *d_mat_array, *d_out_array;
    cudaMalloc((float**)&d_mat_array, byte_size);
    cudaMalloc((void**)&d_out_array, byte_size);
    cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    float alpha = 1.;
    float beta  = 0.;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, nx, ny, &alpha, d_mat_array, ny, &beta, d_mat_array, ny, d_out_array, nx);
    //cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, nx, ny, &alpha, d_mat_array, ny, &beta, d_mat_array, ny, d_out_array, nx);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    std::cout<<"time spent executing by the GPU cublas: "<<time/1000.<<std::endl;
    cudaMemcpy(h_out_array, d_out_array, byte_size, cudaMemcpyDeviceToHost);

    compare_arrays(h_out_array, h_trans_array, size, 1e-4);
    print_array(h_out_array + size/2, 16);
}

int main(int argc, char** argv)
{
	//default values for variabless
	int nx = NX;
	int ny = NY;
	int block_x = BDIMX;
	int block_y = BDIMY;
	int kernel_num = 0;

	if (argc > 1)
		kernel_num = atoi(argv[1]);
    kernel_num = 13;

	int size = nx * ny;
	int byte_size = sizeof(float) * size;

	printf("Matrix transpose for %d X % d matrix with block size %d X %d \n",nx,ny,block_x,block_y);

    float * h_mat_array = (float*)malloc(byte_size);
    float * h_trans_array = (float*)malloc(byte_size);
    float * h_ref = (float*)malloc(byte_size);

	//initialize matrix with integers between one and ten
	//initialize(h_mat_array,size ,INIT_ONE_TO_TEN);
    initialize(h_mat_array,size ,INIT_RANDOM);

	//matirx transpose in CPU
	mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);

	int * d_mat_array, *d_trans_array;

	cudaMalloc((void**)&d_mat_array, byte_size);
	cudaMalloc((void**)&d_trans_array, byte_size);

	cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);

	dim3 blocks(block_x, block_y);
	dim3 grid(nx/block_x, ny/block_y);

    void(*kernel)(int*, int*, int, int);
	char * kernel_name;

	switch (kernel_num)
	{
	case 0:
		kernel = &copy_row;
		kernel_name = "Copy row   ";
		break;
	case 1 :
		kernel = &copy_column;
		kernel_name = "Copy column   ";
		break;
	case 2 :
		kernel = &transpose_read_row_write_column;
		kernel_name = " Read row write column ";
		break;
	case 3:
		kernel = &transpose_read_column_write_row;
		kernel_name = "Read column write row ";
		break;
	case 4:
		kernel = &transpose_unroll4_row;
		kernel_name = "Unroll 4 row ";
		break;
	case 5:
		kernel = &transpose_unroll4_col;
		kernel_name = "Unroll 4 col ";
		break;
	case 6:
		kernel = &transpose_diagonal_row;
		kernel_name = "Diagonal row ";
		break;
    case 7:
        kernel = &transpose_unroll4_col2;
        kernel_name = "Unroll 4 col 2";
        break;
    case 8:
        kernel = &transpose_diagonal_row2;
        kernel_name = "Diagonal row 2";
        break;
    case 9:
        kernel = &transpose_diagonal_col;
        kernel_name = "Diagonal col";
        break;
    case 10:
        kernel = &transpose_smem_orig;
        kernel_name = "SMem orig";
        break;
    case 11:
        kernel = &transpose_smem_orig_pad;
        kernel_name = "SMem orig pad";
        break;
    case 12:
        kernel = &transpose_smem_orig_unrolling;
        kernel_name = "SMem orig unrolling";
        break;
    case 13:
        kernel = &transpose_smem_orig_pad_unrolling;
        kernel_name = "SMem orig pad unrolling";
        break;
    case 14:
        kernel = &transpose_smem;
        kernel_name = "SMem";
        break;
	}

	printf(" Launching kernel %s \n",kernel_name);

	clock_t gpu_start, gpu_end;
	gpu_start = clock();

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    float gpu_time = 0.0f;
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

	kernel <<< grid, blocks>>> (d_mat_array, d_trans_array,nx, ny);

    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    unsigned long int counter = 0;
    while(cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter ++;
    }
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("time spent executing by the GPU: %.6f\n", gpu_time);

	cudaDeviceSynchronize();

	gpu_end = clock();
	print_time_using_host_clock(gpu_start, gpu_end);

	//copy the transpose memroy back to cpu
	cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost);

	//compare the CPU and GPU transpose matrix for validity
	compare_arrays(h_ref, h_trans_array, size, 1e-4);

    print_array(h_trans_array + size/2, 16);
    thrust_transpose(h_mat_array, h_trans_array, nx, ny);
    cublas_transpose(h_mat_array, h_trans_array, nx, ny);

    /*
    printf("before transpose:\n");
    print_matrix(h_mat_array, nx, ny);
    printf("after transpose:\n");
    print_matrix(h_ref, ny, nx);
    */


	cudaDeviceReset();
	return EXIT_SUCCESS;
}