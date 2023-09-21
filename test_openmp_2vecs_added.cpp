#include <stdio>
#include <ctime>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <dlfcn.h>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <random>

#include <omp.h>
#include <immintrin.h>

#define W 1920*16
#define H 1080*16
#define D 3*2

//using namespace std;

enum INIT_PARAM{
    INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X,INIT_RANDOM_KEY
};

//simple initialization
void initialize(int * input, unsigned long array_size,
                INIT_PARAM PARAM)
{
    if (PARAM == INIT_ONE)
    {
        for (unsigned long i = 0; i < array_size; i++)
        {
            input[i] = 1;
        }
    }
    else if (PARAM == INIT_ONE_TO_TEN)
    {
        for (unsigned long i = 0; i < array_size; i++)
        {
            input[i] = i % 10;
        }
    }
    else if (PARAM == INIT_RANDOM)
    {
        time_t t;
        srand((unsigned)time(&t));
        for (unsigned long i = 0; i < array_size; i++)
        {
            input[i] = (int)(rand() & 0xFF);
        }
    }
    else if (PARAM == INIT_FOR_SPARSE_METRICS)
    {
        srand(time(NULL));
        int value;
        for (unsigned long i = 0; i < array_size; i++)
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
        for (unsigned long i = 0; i < array_size; i++)
        {
            input[i] = (int)(rand() & 0xFF);
        }
    }
    else if (PARAM == INIT_RANDOM_KEY)
    {
        for(unsigned long i = 0; i < array_size; i++)
        {
            input[i] = i;//定义一个从0到n-1不重复的数组
        }
        srand(time(0));//随机数种子以从1970年1月1日00:00:00到现在的秒数为种子
        for(unsigned long i = 0; i < array_size; i++) {
            int j = rand() % array_size;//使j随机取0到n-1的数
            int temp = input[i];//用temp存储第i个数
            input[i] = input[j];//将第j个数的数值赋值给第i个数
            input[j] = temp;//将原先第i个数的数值赋值给第j个数
            //这一系列操作其实就是将0到n-1中的两个数互换位置，依旧保持不重复的数值
        }
    }

}

void initialize(float * input, const unsigned long array_size,
                INIT_PARAM PARAM)
{
    if (PARAM == INIT_ONE)
    {
        for (unsigned long i = 0; i < array_size; i++)
        {
            input[i] = 1.;
        }
    }
    else if (PARAM == INIT_ONE_TO_TEN)
    {
        for (unsigned long i = 0; i < array_size; i++)
        {
            input[i] = i % 10;
        }
    }
    else if (PARAM == INIT_RANDOM)
    {
        for (unsigned long i = 0; i < array_size; i++)
        {
            //input[i] = u(e);
            input[i] = 0+1.0*(rand()%RAND_MAX)/RAND_MAX *(1-0);
        }
    }
    else if (PARAM == INIT_FOR_SPARSE_METRICS)
    {
        srand(time(NULL));
        int value;
        for (unsigned long i = 0; i < array_size; i++)
        {
            value = rand() % 25;
            if (value < 5)
            {
                input[i] = 0+1.0*(rand()%RAND_MAX)/RAND_MAX *(1-0);
            }
            else
            {
                input[i] = 0;
            }
        }
    }
}

template <class T>
void compare_arrays(T * a, T * b, unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
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
void compare_arrays(int * a, int * b, unsigned long size);
template
void compare_arrays(float * a, float * b, unsigned long size);
template
void compare_arrays(unsigned int * a, unsigned int * b, unsigned long size);

template <class T>
void compare_arrays(T * a, T * b, unsigned long size, T precision)
{
    for (unsigned long i = 0; i < size; i++)
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
void compare_arrays(float * a, float * b, unsigned long size, float precision);
template
void compare_arrays(double * a, double * b, unsigned long size, double precision);

void print_array(int * input, unsigned long array_size)
{
    for (unsigned long i = 0; i < array_size; i++)
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

void print_array(float * input, unsigned long array_size)
{
    for (unsigned long i = 0; i < array_size; i++)
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

template <typename T>
void sum_array_cpu(T *a, T *b, T *c, unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
        //c[i] = (pow(a[i], 2) + b[i] / 4) / 3;
    }
}
template
void sum_array_cpu(int *a, int *b, int *c, unsigned long size);
template
void sum_array_cpu(float *a, float *b, float *c, unsigned long size);

void sum_array_cpu(float *a, float *b, float *c, unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
        //c[i] = (pow(a[i], 2) + b[i] / 4) / 3;
    }
}

void sum_array_cpu(float *a, float *b, float *c, unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
        //c[i] = (pow(a[i], 2) + b[i] / 4) / 3;
    }
}

void sum_array_cpu_si_avx2(int const *a, int const *b, int *c, unsigned long size)
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
    /*
    //#pragma omp parallel for num_threads(12)
    //#pragma omp parallel for num_threads(12) schedule(static, 1024)
    for (unsigned long tid = 0; tid < size / 8; tid++)
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
    */

    //#pragma omp parallel
    //#pragma omp parallel for
    //#pragma omp parallel for private(i)//要把i提前定义，但是可能有乱序问题
    //以上都慢了6倍
    //for循环外面套上{}，慢36倍
    //for循环外面套上{}，{}上面加parallel，for前面加for，慢6倍，效果和parallel for一样
    #pragma omp parallel for num_threads(12)
    //#pragma omp parallel for num_threads(12) schedule(static, 1024)
    for (unsigned long i = 0; i < size; i++)
    {
        //if(i % (1 << 29) == 0) {
        //    printf("threads:%d\n", omp_get_num_threads());
        //    printf("thread_num:%d\n", omp_get_thread_num());
        //}
        c[i] = a[i] + b[i];
        //c[i] = (pow(a[i], 2) + b[i] / 4) / 3;
    }
}

void sum_array_cpu_ps_avx2(float const *a, float const *b, float *c, unsigned long size)
{
    /*
    unsigned long loop = size / 8;
    //去掉nvcc -G以后，加上schedule(static, 1024)会引起segment fault问题
    //#pragma omp parallel for num_threads(12)
    //#pragma omp parallel for num_threads(12) schedule(static, 1024)
    for (unsigned long tid = 0; tid < loop; tid++)
    {
        __m256 aavx = _mm256_loadu_ps(&a[tid * 8]);
        __m256 bavx = _mm256_loadu_ps(&b[tid * 8]);
        __m256 cavx = _mm256_add_ps(aavx, bavx);
        _mm256_storeu_ps(&c[tid*8], cavx);
    }
    */
    //#pragma omp for
    #pragma omp parallel for num_threads(12)
    //#pragma omp parallel for num_threads(12) schedule(static, 1024)
    for (unsigned long i = 0; i < size; i++)
    {
        //if(i % (1 << 29) == 0) {
        //    printf("threads:%d\n", omp_get_num_threads());
        //    printf("thread_num:%d\n", omp_get_thread_num());
        //}
        c[i] = a[i] + b[i];
        //c[i] = (pow(a[i], 2) + b[i] / 4) / 3;
    }
}

template <typename T>
void sum_array_cpu_simd_avx2(T *h_a, T *h_b, T *h_c, unsigned long size) {
    if(typeid(T).name() == typeid((int)1).name())
        sum_array_cpu_si_avx2((int*)h_a, (int*)h_b, (int*)h_c, size);
    else
        sum_array_cpu_ps_avx2((float*)h_a, (float*)h_b, (float*)h_c, size);
}
template
void sum_array_cpu_simd_avx2(int *h_a, int *h_b, int *h_c, unsigned long size);
template
void sum_array_cpu_simd_avx2(float *h_a, float *h_b, float *h_c, unsigned long size);

template <typename T>
int exec(int argc, char** argv) {
    unsigned long w = W;
    unsigned long h = H;
    unsigned long d = D;
    unsigned long size = w * h * d;
    //unsigned long size = 1 << 27;

    bool print_a = true;

    unsigned long NO_BYTES = size * sizeof(T);
    unsigned long total_in_sizeG_cpu = (NO_BYTES * 4) >> 30;
    unsigned long total_in_sizeG_gpu = (NO_BYTES * 3) >> 30;
    //unsigned long转double运行时illegal instruction
    printf("Sum array dimension:(%d X %d X %d), size:%zu, bytes:%zu, bytes total in cpu:%zu G, bytes total in gpu:%zu G\n",
           w, h, d, size, NO_BYTES, total_in_sizeG_cpu, total_in_sizeG_gpu);

    //host pointers
    T *h_a, *h_b, *h_cpu_results, *h_c;

    //allocate memory for host pointers
    /*
    h_a = (T *)aligned_alloc(sizeof(__m256), NO_BYTES);
    h_b = (T *)aligned_alloc(sizeof(__m256), NO_BYTES);
    h_c = (T *)aligned_alloc(sizeof(__m256), NO_BYTES);
    h_cpu_results = (T *)aligned_alloc(sizeof(__m256), NO_BYTES);
    h_a = (T *)aligned_alloc(sizeof(__m512), NO_BYTES);
    h_b = (T *)aligned_alloc(sizeof(__m512), NO_BYTES);
    h_c = (T *)aligned_alloc(sizeof(__m512), NO_BYTES);
    h_cpu_results = (T *)aligned_alloc(sizeof(__m512), NO_BYTES);
    */
    //！！！align内存后，gpu3d会出错，非法内存
    h_a = (T *) malloc(NO_BYTES);
    h_b = (T *) malloc(NO_BYTES);
    h_c = (T *) malloc(NO_BYTES);
    h_cpu_results = (T *) malloc(NO_BYTES);
    if (h_a == NULL || h_b == NULL || h_c == NULL || h_cpu_results == NULL) {
        printf("malloc failed, exit\n");
        exit(1);
    }

    //initialize host pointer
    initialize(h_a, size, INIT_RANDOM);
    initialize(h_b, size, INIT_RANDOM);

    printf("Start %s type:%s execution\n", "CPU not optimized", typeid(T).name());
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_cpu_results, size);
    cpu_end = clock();
    printf("%s %s time: %f\n", "CPU not optimized", typeid(T).name(), (double) (cpu_end - cpu_start) / CLOCKS_PER_SEC);
    if (print_a) {
        print_array(h_cpu_results, 10);
        print_array(h_cpu_results + size / 2 - 10, 10);
        print_array(h_cpu_results + size / 2 + 10, 10);
        print_array(h_cpu_results + size - 10, 10);
    }

    cpu_start = clock();
    sum_array_cpu_simd_avx2(h_a, h_b, h_c, size);
    cpu_end = clock();
    printf("%s %s time: %f\n", "CPU avx2+openmp", typeid(T).name(), (double)(cpu_end - cpu_start)/CLOCKS_PER_SEC);
    if ( print_a)
    {
        print_array(h_c, 10);
        print_array(h_c + size / 2 - 10, 10);
        print_array(h_c + size / 2 + 10, 10);
        print_array(h_c + size - 10, 10);
    }
    printf("Compare %s %s result with cpu:\n", "CPU avx2+openmp", typeid(T).name());
    compare_arrays(h_c, h_cpu_results, size);

}

int main(int argc, char** argv) {
    exec<int>(argc, argv);
    //exec<float>(argc, argv);
}
/*
                cpu                 openmp              opemp+avx2
int
float

 */