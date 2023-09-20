#include "common.hpp"
using namespace std;

#include <iterator>
#include <algorithm>
#include <iostream>
#include <dlfcn.h>

#include <omp.h>
#include <immintrin.h>

/*
gcc -march=knl -dM -E - < /dev/null | egrep "SSE|AVX" | sort
#define __AVX2__ 1
#define __AVX512CD__ 1
#define __AVX512ER__ 1
#define __AVX512F__ 1
#define __AVX512PF__ 1
#define __AVX__ 1
#define __MMX_WITH_SSE__ 1
#define __SSE2_MATH__ 1
#define __SSE2__ 1
#define __SSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#define __SSE_MATH__ 1
#define __SSE__ 1
#define __SSSE3__ 1

for i in 4fmaps 4vnniw ifma vbmi vpopcntdq ; do echo "==== $i ====" ; gcc -mavx512$i -dM -E - < /dev/null | egrep "AVX512" | sort ; done
==== 4fmaps ====
#define __AVX5124FMAPS__ 1
#define __AVX512F__ 1
==== 4vnniw ====
#define __AVX5124VNNIW__ 1
#define __AVX512F__ 1
==== ifma ====
#define __AVX512F__ 1
#define __AVX512IFMA__ 1
==== vbmi ====
#define __AVX512BW__ 1
#define __AVX512F__ 1
#define __AVX512VBMI__ 1
==== vpopcntdq ====
#define __AVX512F__ 1
#define __AVX512VPOPCNTDQ__ 1


 */
template <typename T>
void sum_array_cpu(T *a, T *b, T *c, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
        //c[i] = (pow(a[i], 2) + b[i] / 4) / 3;
    }
}
template
void sum_array_cpu(int *a, int *b, int *c, size_t size);
template
void sum_array_cpu(float *a, float *b, float *c, size_t size);

void sum_array_cpu_si_avx2(int const *a, int const *b, int *c, size_t size)
{
    /*
    int procs = omp_get_num_procs();
    printf("procs:%d\n", procs);
    //#pragma opm num_threads(procs * 2);没用
    char str_procs[5] = {0};
    sprintf(str_procs, "%d", procs);
    setenv("OMP_NUM_THREADS", str_procs, 1);
    //#pragma omp for nowait
    //#pragma omp parallel
    //omp_set_num_threads(procs);
    */
    //#pragma omp parallel for num_threads(12) schedule(static, 1024)
    for (size_t tid = 0; tid < size / 8; tid++)
    {
        //if(tid % (1 << 26) == 0) {
        //    printf("threads:%d\n", omp_get_num_threads());
        //    printf("thread_num:%d\n", omp_get_thread_num());
        //}
        __m256i aavx = _mm256_loadu_si256((__m256i*)(&a[tid * 8]));
        __m256i bavx = _mm256_loadu_si256((__m256i*)(&b[tid * 8]));
        __m256i cavx = _mm256_add_epi32(aavx, bavx);
        _mm256_storeu_si256((__m256i*)(&c[tid*8]), cavx);
    }
    //printf("threads:%d\n", omp_get_num_threads());
    //for循环结束之后打印是1
//procs:12
//threads:12
//thread_num:0
//threads:12
//thread_num:2
//threads:12
//thread_num:4
//threads:12
//thread_num:6
//threads:12
//thread_num:8
//threads:12
//thread_num:10

    //#pragma omp parallel
    //#pragma omp parallel for
    //#pragma omp parallel for private(i)//要把i提前定义，但是可能有乱序问题
    //以上都慢了6倍
    //for循环外面套上{}，慢36倍
    //for循环外面套上{}，{}上面加parallel，for前面加for，慢6倍，效果和parallel for一样
    /*
    #pragma omp parallel for num_threads(12)
    for (size_t i = 0; i < size; i++)
    {
        //if(i % (1 << 29) == 0) {
        //    printf("threads:%d\n", omp_get_num_threads());
        //    printf("thread_num:%d\n", omp_get_thread_num());
        //}
        c[i] = a[i] + b[i];
        //c[i] = (pow(a[i], 2) + b[i] / 4) / 3;
    }
    */
}

void sum_array_cpu_ps_avx2(float const *a, float const *b, float *c, size_t size)
{
    size_t loop = size / 8;
    //#pragma omp parallel for num_threads(12)
    for (size_t tid = 0; tid < loop; tid++)
    {
        __m256 aavx = _mm256_loadu_ps(&a[tid * 8]);
        __m256 bavx = _mm256_loadu_ps(&b[tid * 8]);
        __m256 cavx = _mm256_add_ps(aavx, bavx);
        _mm256_storeu_ps(&c[tid*8], cavx);
    }
    /*
    //#pragma omp for
    #pragma omp parallel for num_threads(12)
    for (size_t i = 0; i < size; i++)
    {
        //if(i % (1 << 29) == 0) {
        //    printf("threads:%d\n", omp_get_num_threads());
        //    printf("thread_num:%d\n", omp_get_thread_num());
        //}
        c[i] = a[i] + b[i];
        c[i] = (pow(a[i], 2) + b[i] / 4) / 3;
    }
    */
}

template <typename T>
void sum_array_cpu_simd_avx2(T *h_a, T *h_b, T *h_c, size_t size) {
    if(typeid(T).name() == typeid((int)1).name())
        sum_array_cpu_si_avx2((int*)h_a, (int*)h_b, (int*)h_c, size);
    else
        sum_array_cpu_ps_avx2((float*)h_a, (float*)h_b, (float*)h_c, size);
}
template
void sum_array_cpu_simd_avx2(int *h_a, int *h_b, int *h_c, size_t size);
template
void sum_array_cpu_simd_avx2(float *h_a, float *h_b, float *h_c, size_t size);

void sum_array_cpu_si_avx512(int const *a, int const *b, int *c, size_t size)
{
    size_t loop = size / 16;
    //#pragma omp parallel for num_threads(12)
    for (size_t tid = 0; tid < loop; tid++)
    {
        //__m512i aavx = _mm512_loadu_si512((__m512i*)(&a[tid * 16]));
        __m512i aavx = _mm512_load_si512((__m512i*)(&a[tid * 16]));
        //__m512i bavx = _mm512_loadu_si512((__m512i*)(&b[tid * 16]));
        __m512i bavx = _mm512_load_si512((__m512i*)(&b[tid * 16]));
        __m512i cavx = _mm512_add_epi32(aavx, bavx);
        //_mm512_storeu_si512((__m512i*)(&c[tid*16]), cavx);
        _mm512_store_si512((__m512i*)(&c[tid*16]), cavx);
    }
}

void sum_array_cpu_ps_avx512(float const *a, float const *b, float *c, size_t size)
{
    size_t loop = size / 16;
    //#pragma omp parallel for num_threads(12)
    for (size_t tid = 0; tid < loop; tid++)
    {
        __m512 aavx = _mm512_loadu_ps(&a[tid * 16]);
        __m512 bavx = _mm512_loadu_ps(&b[tid * 16]);
        __m512 cavx = _mm512_add_ps(aavx, bavx);
        _mm512_storeu_ps(&c[tid*16], cavx);
    }
}

template <typename T>
void sum_array_cpu_simd_avx512(T *h_a, T *h_b, T *h_c, size_t size) {
    if(typeid(T).name() == typeid((int)1).name())
        sum_array_cpu_si_avx512((int*)h_a, (int*)h_b, (int*)h_c, size);
    else
        sum_array_cpu_ps_avx512((float*)h_a, (float*)h_b, (float*)h_c, size);
}
template
void sum_array_cpu_simd_avx512(int *h_a, int *h_b, int *h_c, size_t size);
template
void sum_array_cpu_simd_avx512(float *h_a, float *h_b, float *h_c, size_t size);


/*
template <typename T, char *sep, int tablen>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    size_t v_size = v.size();
    //cout << "size:" << size << ", tablen:" << tablen << ", (size + tablen - 1) / tablen:" << (size + tablen - 1) / tablen << std::endl;
    if (!v.empty()) {
        for(size_t i = 0; i < (v_size + tablen - 1) / tablen; i ++)
        {
            int offset = tablen * i;
            std::vector<T> cut_vector(v.begin() + offset, v.begin() + std::min(v_size, offset + tablen));
            for (auto el : cut_vector) {
                std::cout << el << sep;
            }
            cout << std::endl;
        }
    }
    return out;
}
template
std::ostream& operator<< <", ", 5>(std::ostream& out, const std::vector<int>& v);
*/

template <typename T> void print_vec(const std::vector<T> &v, size_t start, size_t end, char *sep, int tablen) {
    size_t size = end - start;
    size_t v_size = v.size();
    //cout << "size:" << size << ", tablen:" << tablen << ", (size + tablen - 1) / tablen:" << (size + tablen - 1) / tablen << std::endl;
    if (!v.empty()) {
        for(size_t i = 0; i < (size + tablen - 1) / tablen; i ++)
        {
            /*
            int offset = TABLEN * i;
            std::vector<T> last  = v.begin() + std::min(size - 1, offset + TABLEN - 1);
            std::copy(first, last, std::ostream_iterator<T>(out, SEP));
            */
            size_t offset = start + tablen * i;
            std::vector<T> cut_vector(v.begin() + offset, v.begin() + std::min(v_size, offset + tablen));
            for (auto el : cut_vector) {
                std::cout << el << sep;
            }
            //cout << '\b';
            //cout << " ";
            cout << std::endl;
        }
    }
}
template void print_vec(const std::vector<int> &v, size_t start, size_t end, char *sep, int tablen);
template void print_vec(const std::vector<unsigned int> &v, size_t start, size_t end, char *sep, int tablen);

template <typename T>
T add(const T &a, const T &b)
{
    return a + b;
}
template int add(const int &a, const int &b);

template <typename T>
void add_bias_(std::vector<T> &v, const T &a)
{
    for (auto& pt : v) {
        pt += a;
    }
}
template void add_bias_(std::vector<int> &v, const int &a);


void launch_dummmy_kernel()
{

}

void print_array(int * input, size_t array_size)
{
	for (size_t i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%d,", input[i]);
		}
		else
		{
			printf("%d \n", input[i]);
		}
	}
}

void print_array(float * input, size_t array_size)
{
	for (size_t i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%f,", input[i]);
		}
		else
		{
			printf("%f \n", input[i]);
		}
	}
}

void print_matrix(int * matrix, size_t nx, size_t ny)
{
    for (size_t ix = 0; ix < nx; ix++)
	{
        for (size_t iy = 0; iy < ny; iy++)
		{
            printf("%d ", matrix[ny * ix + iy]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_matrix(float * matrix, size_t nx, size_t ny)
{
    for (size_t ix = 0; ix < nx; ix++)
	{
        for (size_t iy = 0; iy < ny; iy++)
		{
			printf("%f ", matrix[ny * ix + iy]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_arrays_toafile_side_by_side(float*a, float*b, size_t size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (size_t i = 0; i < size; i++) {
			file << i << " - " <<a[i] << " - " << b[i] << "\n";
		}
		file.close();
	}
}

void print_arrays_toafile_side_by_side(int*a, int*b, size_t size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (size_t i = 0; i < size; i++) {
			file << i << " - " << a[i] << " - " << b[i] << "\n";
		}
		file.close();
	}
}

void print_arrays_toafile(int*a, size_t size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (size_t i = 0; i < size; i++) {
			file << i << " - " << a[i] << "\n";
		}
		file.close();
	}
}



int* get_matrix(size_t rows, size_t columns)
{
    size_t mat_size = rows * columns;
    size_t mat_byte_size = sizeof(int)*mat_size;

	int * mat = (int*)malloc(mat_byte_size);

	for (size_t i = 0; i < mat_size; i++)
	{
		if (i % 5 == 0)
		{
			mat[i] = i;
		}
		else
		{
			mat[i] = 0;
		}
	}

	//initialize(mat,mat_size,INIT_FOR_SPARSE_METRICS);
	return mat;
}

//simple initialization
void initialize(int * input, size_t array_size,
	INIT_PARAM PARAM, int x)
{
	if (PARAM == INIT_ONE)
	{
		for (size_t i = 0; i < array_size; i++)
		{
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (size_t i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		time_t t;
		srand((unsigned)time(&t));
		for (size_t i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (size_t i = 0; i < array_size; i++)
		{
			value = rand() % 25;
			if (value < 5)
			{
				input[i] = value;
			}
			else
			{
				input[i] = 0;
			}
		}
	}
	else if (PARAM == INIT_0_TO_X)
	{
		srand(time(NULL));
		int value;
		for (size_t i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
    else if (PARAM == INIT_RANDOM_KEY)
    {
        for(size_t i = 0; i < array_size; i++)
        {
            input[i] = i;//定义一个从0到n-1不重复的数组
        }
        srand(time(0));//随机数种子以从1970年1月1日00:00:00到现在的秒数为种子
        for(size_t i = 0; i < array_size; i++) {
            int j = rand() % array_size;//使j随机取0到n-1的数
            int temp = input[i];//用temp存储第i个数
            input[i] = input[j];//将第j个数的数值赋值给第i个数
            input[j] = temp;//将原先第i个数的数值赋值给第j个数
            //这一系列操作其实就是将0到n-1中的两个数互换位置，依旧保持不重复的数值
        }
    }

}

void initialize(float * input, const size_t array_size,
	INIT_PARAM PARAM)
{
	if (PARAM == INIT_ONE)
	{
		for (size_t i = 0; i < array_size; i++)
		{
			input[i] = 1.;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (size_t i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
        uniform_real_distribution<float> u(-1, 1);
        default_random_engine e(time(NULL));
		for (size_t i = 0; i < array_size; i++)
		{
			//input[i] = u(e);
            input[i] = 0+1.0*(rand()%RAND_MAX)/RAND_MAX *(1-0);
        }
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (size_t i = 0; i < array_size; i++)
		{
			value = rand() % 25;
			if (value < 5)
			{
                uniform_real_distribution<float> u(-1, 1);
                default_random_engine e(time(NULL));
				input[i] = u(e);
			}
			else
			{
				input[i] = 0;
			}
		}
	}
}

//cpu reduction
int reduction_cpu(int * input, const size_t size)
{
	int sum = 0;
	for (size_t i = 0; i < size; i++)
	{
		sum += input[i];
	}
	return sum;
}

//cpu transpose
template <class T>
void mat_transpose_cpu(T * mat, T * transpose, size_t nx, size_t ny)
{
	for (size_t iy = 0; iy < ny; iy++)
	{
		for (size_t  ix = 0; ix < nx; ix++)
		{
			transpose[ix * ny + iy] = mat[iy * nx + ix];
		}
	}
}
template void mat_transpose_cpu(int * mat, int * transpose, size_t nx, size_t ny);
template void mat_transpose_cpu(float * mat, float * transpose, size_t nx, size_t ny);

//compare results
void compare_results(int gpu_result, int cpu_result)
{
	printf("GPU result : %d , CPU result : %d \n",
		gpu_result, cpu_result);

	if (gpu_result == cpu_result)
	{
		printf("GPU and CPU results are same \n");
		return;
	}

	printf("GPU and CPU results are different \n");
}

void compare_results(float gpu_result, float cpu_result, float precision)
{
    printf("GPU result : %f , CPU result : %f \n",
           gpu_result, cpu_result);

    if (abs(gpu_result - cpu_result) > precision)
    {
        printf("GPU and CPU results are same \n");
        return;
    }

    printf("GPU and CPU results are different \n");
}



//compare arrays
template <class T>
void compare_arrays(T * a, T * b, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");
			printf("%d - %d | %d \n", i, a[i], b[i]);
			return;
		}
	}
	printf("Arrays are same \n");
}
template
void compare_arrays(int * a, int * b, size_t size);
template
void compare_arrays(float * a, float * b, size_t size);

template <class T>
void compare_arrays(T * a, T * b, size_t size, T precision)
{
	for (size_t i = 0; i < size; i++)
	{
		if (abs(a[i] - b[i]) > precision)
		{
			printf("Arrays are different \n");
			
			return;
		}
	}
	printf("Arrays are same \n");
	
}
template
void compare_arrays(float * a, float * b, size_t size, float precision);
template
void compare_arrays(double * a, double * b, size_t size, double precision);

void compare_matrixes(int *a, int *b, size_t m, size_t n)
{
    for(size_t i=0; i<m; i++)
    {
        for(size_t j=0; j<n; j++)
        {
            int offset = n * i + j;
            int value_a = *(a + offset);
            int value_b = *(b + offset);
            if( value_a != value_b)
            {
                printf("Matrics are different \n");
                printf("(%d,%d) - %d | %d \n", i, j, value_a, value_b);
                return;
            }
        }
    }
    printf("Matrics are same \n");
}

void compare_matrixes(float *a, float *b, size_t m, size_t n, float precision)
{
    for(size_t i=0; i<m; i++)
    {
        for(size_t j=0; j<n; j++)
        {
            int offset = n * i + j;
            float value_a = *(a + offset);
            float value_b = *(b + offset);
            if( abs(value_a - value_b) > precision)
            {
                printf("Matrics are different \n");
                printf("(%d,%d) - %f | %f \n", i, j, value_a, value_b);
                return;
            }
        }
    }
    printf("Matrics are same \n");
}

void print_time_using_host_clock(clock_t start, clock_t end)
{
	printf("GPU kernel execution time : %4.6f \n",
		(double)((double)(end - start) / CLOCKS_PER_SEC));
}

void printData(char *msg, int *in, size_t size)
{
	printf("%s: ", msg);

	for (size_t i = 0; i < size; i++)
	{
		printf("%5d", in[i]);
		fflush(stdout);
	}

	printf("\n");
	return;
}

void sum_array_cpu(float* a, float* b, float *c, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}


