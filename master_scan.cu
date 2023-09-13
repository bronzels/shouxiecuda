#include "scan.cuh"

#define BLOCK_SIZE 512
//#define BLOCK_SIZE 1024

void inclusive_scan_cpu(int *input, int *output, int size)
{
    output[0] = input[0];

    for (int i = 1; i < size; i++)
    {
        output[i] = output[i-1] + input[i];
    }
}

void exclusive_scan_cpu(int *input, int *output, int size)
{
    output[0] = 0;

    for (int i = 1; i < size; i++)
    {
        output[i] = output[i-1] + input[i-1];
    }
}

__global__ void naive_inclusive_scan_single_block(int *input, int size)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        for(int stride = 1; tid + stride < blockDim.x; stride = stride * 2)
        {
            input[tid + stride] += input[tid];
            __syncthreads();
        }
    }
}

__device__ void _efficient_exclusive_scan_single_block(int *input, int size)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        //reduction phase
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int index = (tid+1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                input[index] += input[index - stride];
            }
            __syncthreads();
        }

        // set root value to 0
        if (tid == 0)
            input[blockDim.x - 1] = 0;

        int temp = 0;

        //down sweep
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            int index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                temp = input[index - stride]; //assign left child to temp
                input[index - stride] = input[index]; //left child
                input[index] += temp; //right child
            }
            __syncthreads();
        }

    }
}

__global__ void efficient_exclusive_scan_single_block(int *input, int size) {
    _efficient_exclusive_scan_single_block(input, size);
}

__device__ void _efficient_exclusive_scan_single_block_smem(int *input, int size)
//__global__ void _efficient_exclusive_scan_single_block_smem(int *input, int size)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int shared_block[BLOCK_SIZE];

    if (gid < size)
    {
        shared_block[tid] = input[gid];

        __syncthreads();

        //reduction phase
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int index = (tid+1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                shared_block[index] += shared_block[index - stride];
            }
            __syncthreads();
        }

        // set root value to 0
        if (tid == 0)
            shared_block[blockDim.x - 1] = 0;

        int temp = 0;

        //down sweep
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            int index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                temp = shared_block[index - stride]; //assign left child to temp
                shared_block[index - stride] = shared_block[index]; //left child
                shared_block[index] += temp; //right child
            }
            __syncthreads();
        }

        input[gid] = shared_block[tid];
    }
}

__global__ void efficient_exclusive_scan_single_block_smem(int *input, int size) {
    _efficient_exclusive_scan_single_block_smem(input, size);
}
__global__ void efficient_inclusive_scan_single_block(int *input, int size)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = input[gid];
    __syncthreads();
    _efficient_exclusive_scan_single_block(input, size);
    __syncthreads();
    input[gid] += temp;
    /*

    __shared__ int shared_block[BLOCK_SIZE];

    if (gid < size)
    {
        shared_block[tid] = input[tid];
        __syncthreads();

        //reduction phase
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int index = (tid+1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                shared_block[index] += shared_block[index - stride];
            }
            __syncthreads();
        }

        //down sweep
        for (int stride = blockDim.x / 4; stride > 0; stride /= 2)
        {
            int index = (tid + 1) * 2 * stride - 1;
            if (index + stride < blockDim.x)
            {
                shared_block[index + stride] += shared_block[index];
            }
            __syncthreads();
        }

        input[tid] = shared_block[tid];
    }
    */
}

__global__ void efficient_inclusive_scan_single_block_smem(int *input, int size)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = input[gid];
    _efficient_exclusive_scan_single_block(input, size);
    //__syncthreads();
    input[gid] += temp;
    /*

    __shared__ int shared_block[BLOCK_SIZE];

    if (gid < size)
    {
        shared_block[tid] = input[tid];
        __syncthreads();

        //reduction phase
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int index = (tid+1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                shared_block[index] += shared_block[index - stride];
            }
            __syncthreads();
        }

        //down sweep
        for (int stride = blockDim.x / 4; stride > 0; stride /= 2)
        {
            int index = (tid + 1) * 2 * stride - 1;
            if (index + stride < blockDim.x)
            {
                shared_block[index + stride] += shared_block[index];
            }
            __syncthreads();
        }

        input[tid] = shared_block[tid];
    }
    */
}

__global__ void exclusive_pre_scan(int *input, int * aux, int size)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int shared_block[BLOCK_SIZE];

    if (gid < size)
    {
        shared_block[tid] = input[gid];
        int orig_temp = shared_block[tid];

        __syncthreads();

        //reduction phase
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int index = (tid+1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                shared_block[index] += shared_block[index - stride];
            }
            __syncthreads();
        }

        // set root value to 0
        if (tid == 0)
            shared_block[blockDim.x - 1] = 0;

        int temp = 0;

        //down sweep
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            int index = (tid + 1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                temp = shared_block[index - stride]; //assign left child to temp
                shared_block[index - stride] = shared_block[index]; //left child
                shared_block[index] += temp; //right child
            }
            __syncthreads();
        }

        input[gid] = shared_block[tid];

        if (tid == (blockDim.x - 1))
        {
            aux[blockIdx.x] = shared_block[tid] + orig_temp;
        }
    }
}

__global__ void inclusive_pre_scan(int *input, int * aux, int size)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int shared_block[BLOCK_SIZE];

    if (gid < size)
    {
        shared_block[tid] = input[gid];
        __syncthreads();

        //reduction phase
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int index = (tid+1) * 2 * stride - 1;
            if (index < blockDim.x)
            {
                shared_block[index] += shared_block[index - stride];
            }
            __syncthreads();
        }

        //down sweep
        for (int stride = blockDim.x / 4; stride > 0; stride /= 2)
        {
            int index = (tid + 1) * 2 * stride - 1;
            if (index + stride < blockDim.x)
            {
                shared_block[index + stride] += shared_block[index];
            }
            __syncthreads();
        }

        input[gid] = shared_block[tid];

        if (tid == (blockDim.x - 1))
        {
            aux[blockIdx.x] =  shared_block[tid];
        }
    }
}

__global__ void post_scan(int *input, int * aux, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        input[gid] += aux[blockIdx.x];
    }

}

void exec_kernel_block(dim3 grid, dim3 block, int * h_ref, int * d_input, int input_size, int byte_size)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    //naive_inclusive_scan_single_block <<<grid, block >>> (d_input, input_size);
    //efficient_exclusive_scan_single_block <<<grid, block >>> (d_input, input_size);
    //efficient_exclusive_scan_single_block_smem <<<grid, block >>> (d_input, input_size);
    //efficient_inclusive_scan_single_block <<<grid, block >>> (d_input, input_size);
    efficient_inclusive_scan_single_block_smem <<<grid, block >>> (d_input, input_size);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Kernel execution time using events : %f \n",time);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);
}

void exec_kernel_global(dim3 grid, dim3 block, int * h_output, int * h_ref, int * d_input, int input_size, int byte_size)
{
    int *h_aux, *d_aux;

	int aux_byte_size = grid.x * sizeof(int);
	cudaMalloc((void**)&d_aux , aux_byte_size);
    int *h_aux_dbg = (int *)malloc(aux_byte_size);

	h_aux = (int*)malloc(aux_byte_size);

    /*
    cudaDeviceSynchronize();
    cudaMemcpy(h_aux_dbg, d_aux, aux_byte_size, cudaMemcpyDeviceToHost);
    print_array(h_aux_dbg, grid.x);
     //512 512
    */
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    /* include
    inclusive_pre_scan<<<grid, block>>>(d_input, d_aux, input_size);
    efficient_exclusive_scan_single_block_smem<<<1, grid.x >>> (d_aux, grid.x);
    post_scan<<<grid, block>>>(d_input, d_aux, input_size);
    */

    /* exclude
    */
    exclusive_pre_scan<<<grid, block>>>(d_input, d_aux, input_size);
    efficient_exclusive_scan_single_block_smem<<<1, grid.x >>> (d_aux, grid.x);
    post_scan<<<grid, block>>>(d_input, d_aux, input_size);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Kernel execution time using events : %f \n",time);

    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_aux, d_aux, aux_byte_size, cudaMemcpyDeviceToHost);
	print_arrays_toafile(h_ref, input_size, "input_array.txt");

	for (int i = 0; i < input_size; i++)
	{
		for (int j = 0; j < i / BLOCK_SIZE ; j++)
		{
			h_ref[i] += h_aux[j];
		}
	}

	print_arrays_toafile(h_aux,grid.x, "aux_array.txt");

    //sum_aux_values << < grid, block >> > (d_input, d_aux, input_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost );
    print_arrays_toafile_side_by_side(h_ref, h_output, input_size, "scan_outputs.txt");
    cudaFree(d_aux);
}
/*
#define BLOCK_SIZE 512
 *  efficient_exclusive_scan_single_block                       0.027648
 *  efficient_exclusive_scan_single_block_smem                  0.026624
 *  efficient_exclusive_scan_single_block(__dev)                0.024320
 *  efficient_exclusive_scan_single_block_smem(__dev)           0.025600
 *  efficient_inclusive_scan_single_block                       0.024192
 *  efficient_inclusive_scan_single_block_smem                  0.026624
 *  efficient_inclusive_scan_single_block(reuse ex)             0.024576
 *  efficient_inclusive_scan_single_block_smem(reuse ex)        0.025696
 *  exec_kernel_global_exclusive(new by master)                 0.027648
 *  exec_kernel_global_inclusive(reuse ex)                      0.028672
 *
#define BLOCK_SIZE 1024
 *  efficient_exclusive_scan_single_block                       0.024352
 *  efficient_exclusive_scan_single_block_smem                  0.025600
 *  efficient_exclusive_scan_single_block(__dev)                0.025600
 *  efficient_exclusive_scan_single_block_smem(__dev)           0.026464
 *  efficient_inclusive_scan_single_block                       0.026624
 *  efficient_inclusive_scan_single_block_smem                  0.027520
 *  efficient_inclusive_scan_single_block(reuse ex)             0.026624
 *  efficient_inclusive_scan_single_block_smem(reuse ex)        0.025600
 *  exec_kernel_global_exclusive(new by master)                 0.031744
 *  exec_kernel_global_inclusive(__dev)                         0.029344
 */

int main(int argc, char**argv)
{
	printf("Scan algorithm execution starterd \n");

	int input_size = 1 << 10;//BLOCK_SIZE;//32;

	if (argc > 1)
	{
		input_size = 1 << atoi(argv[1]);
	}

	const int byte_size = sizeof(int) * input_size;

	int * h_input, *h_output, *h_ref, *h_aux;

	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	h_input = (int*)malloc(byte_size);
	h_output = (int*)malloc(byte_size);
	h_ref = (int*)malloc(byte_size);

    //initialize(h_input, input_size, INIT_ONE);
    initialize(h_input, input_size, INIT_RANDOM);
    //initialize(h_input, input_size, INIT_ONE_TO_TEN);

	cpu_start = clock();
	//inclusive_scan_cpu(h_input, h_output, input_size);
    exclusive_scan_cpu(h_input, h_output, input_size);
	cpu_end = clock();
    //print_array(h_output, input_size);
    //return 0;

	int *d_input, *d_aux;
	cudaMalloc((void**)&d_input, byte_size);

	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE);
	dim3 grid(input_size/ block.x);

    //exec_kernel_block(grid, block, h_ref, d_input, input_size, byte_size);
    exec_kernel_global(grid, block, h_output, h_ref, d_input, input_size, byte_size);

	compare_arrays(h_ref, h_output, input_size);

    free(h_input);
    free(h_output);
    free(h_ref);
    cudaFree(d_input);

	gpuErrchk(cudaDeviceReset());
	return 0;
}