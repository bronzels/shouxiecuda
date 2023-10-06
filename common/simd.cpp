#include <iterator>
#include <immintrin.h>
#include <stdlib.h>

/*
cat /proc/cpuinfo | grep name | cut -f 2 -d: | uniq -c
     12  12th Gen Intel(R) Core(TM) i5-12400F
！！！12代酷睿不支持avx512

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
    //#pragma omp parallel for num_threads(12)
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
    //去掉nvcc -G以后，加上schedule(static, 1024)会引起segment fault问题
    //#pragma omp parallel for num_threads(12)
    //#pragma omp parallel for num_threads(12) schedule(static, 1024)
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
