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

	print_arrays_toafile(h_aux, grid.x, "aux_array.txt");

    //sum_aux_values << < grid, block >> > (d_input, d_aux, input_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost );
    print_arrays_toafile_side_by_side(h_ref, h_output, input_size, "scan_outputs.txt");
    cudaFree(d_aux);
}
/*1 << 10
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
/*
thrust
1 << 22
Kernel execution time using events : 11.396000
end datasize:4194304, is_inplace:1, is_cpu:1, is_inclusive:1, time:0.04
end datasize:4194304, is_inplace:1, is_cpu:1, is_inclusive:0, time:0.01
end datasize:4194304, is_inplace:1, is_cpu:0, is_inclusive:1, time:0.02
end datasize:4194304, is_inplace:1, is_cpu:0, is_inclusive:0, time:0.03
end datasize:4194304, is_inplace:0, is_cpu:1, is_inclusive:1, time:0.16
end datasize:4194304, is_inplace:0, is_cpu:1, is_inclusive:0, time:0.13
end datasize:4194304, is_inplace:0, is_cpu:0, is_inclusive:1, time:0.07
end datasize:4194304, is_inplace:0, is_cpu:0, is_inclusive:0, time:0.07
1 << 23
Kernel execution time using events : 5.745408
end datasize:8388608, is_inplace:1, is_cpu:1, is_inclusive:1, time:0.08
end datasize:8388608, is_inplace:1, is_cpu:1, is_inclusive:0, time:0.02
end datasize:8388608, is_inplace:1, is_cpu:0, is_inclusive:1, time:0.04
end datasize:8388608, is_inplace:1, is_cpu:0, is_inclusive:0, time:0.04
end datasize:8388608, is_inplace:0, is_cpu:1, is_inclusive:1, time:0.31
end datasize:8388608, is_inplace:0, is_cpu:1, is_inclusive:0, time:0.25
end datasize:8388608, is_inplace:0, is_cpu:0, is_inclusive:1, time:0.13
end datasize:8388608, is_inplace:0, is_cpu:0, is_inclusive:0, time:0.14

cub
1 << 22
end datasize:4194304, is_inclusive:1, time:0.02
end datasize:4194304, is_inclusive:0, time:0.01

1 << 23
end datasize:8388608, is_inclusive:1, time:0.05
end datasize:8388608, is_inclusive:0, time:0.03

*/
void thrust_scan(int *h_input, int *h_output, int size, bool is_inplace, bool is_cpu, bool is_inclusive) {
    std::cout <<"start datasize:"<<size<<", is_inplace:"<<is_inplace<<", is_cpu:"<<is_cpu<<", is_inclusive:"<<is_inclusive<< std::endl;
    int byte_size = size *sizeof(int);
    clock_t time1,time2;
    time1 = clock();
    if(is_inplace) {
        if(is_cpu) {
            if(is_inclusive)
                thrust::inclusive_scan(thrust::host, h_input, h_input + size, h_input);
            else
                thrust::exclusive_scan(thrust::host, h_input, h_input + size, h_input);
        }
        else {
            int *d_input;
            checkCudaErrors(cudaMalloc(&d_input, byte_size));
            cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
            if(is_inclusive)
                thrust::inclusive_scan(thrust::device, d_input, d_input + size, d_input);
            else
                thrust::exclusive_scan(thrust::device, d_input, d_input + size, d_input);
            cudaMemcpy(h_input, d_input, byte_size, cudaMemcpyDeviceToHost);
            cudaFree(d_input);
        }
    }
    else {
        if(is_cpu) {
            //thrust::host_vector<int> v_h_src(size);
            thrust::host_vector<int> v_h_src(h_input, h_input+size);
            //memcpy(thrust::raw_pointer_cast(v_h_src.data()), h_input, byte_size);
            thrust::host_vector<int> v_h_dst(size);
            if(is_inclusive)
                thrust::inclusive_scan(v_h_src.begin(), v_h_src.end(), v_h_dst.begin());
            else
                thrust::exclusive_scan(v_h_src.begin(), v_h_src.end(), v_h_dst.begin());
            memcpy(h_output, thrust::raw_pointer_cast(v_h_dst.data()), byte_size);
        }
        else {
            //！！！没有这回事，不增加v_h_src就会报告thrust parallel_for failed: cudaErrorInvalidResourceHandle: invalid resource handle
            //thrust::host_vector<int> v_h_src(size);
            //memcpy(thrust::raw_pointer_cast(v_h_src.data()), h_input, byte_size);
            thrust::device_vector<int> v_d_src(size);
            //v_d_src = v_h_src;
            cudaMemcpy(thrust::raw_pointer_cast(v_d_src.data()), h_input, byte_size, cudaMemcpyHostToDevice);
            thrust::device_vector<int> v_d_dst(size);
            if(is_inclusive)
                thrust::inclusive_scan(v_d_src.begin(), v_d_src.end(), v_d_dst.begin());
            else
                thrust::exclusive_scan(v_d_src.begin(), v_d_src.end(), v_d_dst.begin());
            cudaMemcpy(h_output, thrust::raw_pointer_cast(v_d_dst.data()), byte_size, cudaMemcpyDeviceToHost);
        }
    }
    time2 = clock();
    std::cout<<"end datasize:"<<size<<", is_inplace:"<<is_inplace<<", is_cpu:"<<is_cpu<<", is_inclusive:"<<is_inclusive<<", time:"<<(double)(time2 - time1) / CLOCKS_PER_SEC<<std::endl;;
}

template <typename T>
void cub_scan(T *h_input, T *h_output, int size, bool is_inclusive) {
    std::cout <<"start datasize:"<<size<<", is_inclusive:"<<is_inclusive<< std::endl;
    int byte_size = size *sizeof(T);
    clock_t time1,time2;
    time1 = clock();
    T *d_input, *d_output;
    cudaMalloc(&d_input, byte_size);
    cudaMalloc(&d_output, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
    void *dev_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    if(is_inclusive) {
        cub::DeviceScan::InclusiveSum(dev_temp_storage, temp_storage_bytes, d_input, d_output, size);
        cudaMalloc(&dev_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(dev_temp_storage, temp_storage_bytes, d_input, d_output, size);
    }
    else {
        cub::DeviceScan::ExclusiveSum(dev_temp_storage, temp_storage_bytes, d_input, d_output, size);
        cudaMalloc(&dev_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(dev_temp_storage, temp_storage_bytes, d_input, d_output, size);
    }
    cudaFree(dev_temp_storage);
    cudaFree(d_input);
    cudaMemcpy(h_output, d_output, byte_size, cudaMemcpyDeviceToHost);
    time2 = clock();
    std::cout<<"end datasize:"<<size<<", is_inclusive:"<<is_inclusive<<", time:"<<(double)(time2 - time1) / CLOCKS_PER_SEC<<std::endl;;
}
template
void cub_scan(int *h_input, int *h_output, int size, bool is_inclusive);

int main(int argc, char**argv)
{
	printf("Scan algorithm execution starterd \n");

	//int input_size = 1 << 10;//BLOCK_SIZE;//32;
    int input_size = 1 << 22;

	if (argc > 1)
	{
		input_size = 1 << atoi(argv[1]);
	}

	const int byte_size = sizeof(int) * input_size;

	int * h_input, *h_output, *h_output_inclusive, *h_output_exclusive, *h_ref, *h_aux;

	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	h_input = (int*)malloc(byte_size);
	h_output = (int*)malloc(byte_size);
    h_output_inclusive = (int*)malloc(byte_size);
	h_ref = (int*)malloc(byte_size);

    //initialize(h_input, input_size, INIT_ONE);
    initialize(h_input, input_size, INIT_RANDOM);
    //initialize(h_input, input_size, INIT_ONE_TO_TEN);

	cpu_start = clock();
	//inclusive_scan_cpu(h_input, h_output, input_size);
    exclusive_scan_cpu(h_input, h_output, input_size);
    h_output_exclusive = h_output;
    inclusive_scan_cpu(h_input, h_output_inclusive, input_size);
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

	compare_arrays(h_ref, h_output_inclusive, input_size);

    int *h_input_yat = (int *)malloc(byte_size);
    int *h_lib_output = (int *)malloc(byte_size);
    int lib = 1;
    if(lib == 0) {
        memcpy(h_input_yat, h_input, byte_size);
        thrust_scan(h_input_yat, h_lib_output, input_size, true, true, true);
        compare_arrays(h_input_yat, h_output_inclusive, input_size);
        memcpy(h_input_yat, h_input, byte_size);
        thrust_scan(h_input_yat, h_lib_output, input_size, true, true, false);
        compare_arrays(h_input_yat, h_output_exclusive, input_size);
        memcpy(h_input_yat, h_input, byte_size);
        thrust_scan(h_input_yat, h_lib_output, input_size, true, false, true);
        compare_arrays(h_input_yat, h_output_inclusive, input_size);
        memcpy(h_input_yat, h_input, byte_size);
        thrust_scan(h_input_yat, h_lib_output, input_size, true, false, false);
        compare_arrays(h_input_yat, h_output_exclusive, input_size);

        thrust_scan(h_input, h_lib_output, input_size, false, true, true);
        compare_arrays(h_lib_output, h_output_inclusive, input_size);
        thrust_scan(h_input, h_lib_output, input_size, false, true, false);
        compare_arrays(h_lib_output, h_output_exclusive, input_size);
        thrust_scan(h_input, h_lib_output, input_size, false, false, true);
        compare_arrays(h_lib_output, h_output_inclusive, input_size);
        thrust_scan(h_input, h_lib_output, input_size, false, false, false);
        compare_arrays(h_lib_output, h_output_exclusive, input_size);
    }
    else {
        cub_scan(h_input, h_lib_output, input_size, true);
        compare_arrays(h_lib_output, h_output_inclusive, input_size);
        cub_scan(h_input, h_lib_output, input_size, false);
        compare_arrays(h_lib_output, h_output_exclusive, input_size);
    }

    free(h_input);
    free(h_output);
    free(h_output_inclusive);
    free(h_ref);
    cudaFree(d_input);

    std::cout << "free h_input_yat" << std::endl;
    free(h_input_yat);
    std::cout << "free h_lib_output" << std::endl;
    free(h_lib_output);

//！！！没有这回事，不管是host/device的数据指针，如果被用作thrust::host_vector/thrust::device_vector形如 thrust::host_vector(ptr, ptr+1)的初始化,
//就会被vector的析构释放，同时如果再被free/cudaFree就会出现重复释放的问题。free会打印整个对战，cudaFree会简单illegal mem错误。

	gpuErrchk(cudaDeviceReset());
	return 0;
}