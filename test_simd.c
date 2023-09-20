#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>
int main()
{
    long i, N = 160000000;
    int * A = (int *)aligned_alloc(sizeof(__m256), sizeof(int) * N);
    int * B = (int *)aligned_alloc(sizeof(__m256), sizeof(int) * N);
    int * C = (int *)aligned_alloc(sizeof(__m256), sizeof(int) * N);

    int * E = (int *)aligned_alloc(sizeof(__m512), sizeof(int) * N);
    int * F = (int *)aligned_alloc(sizeof(__m512), sizeof(int) * N);
    int * G = (int *)aligned_alloc(sizeof(__m512), sizeof(int) * N);

    srand(time(0));

    for(i=0;i<N;i++)
    {
        A[i] = rand();
        B[i] = rand();
        E[i] = rand();
        F[i] = rand();
    }

    double time = omp_get_wtime();
    for(i=0;i<N;i++)
    {
        C[i] = A[i] + B[i];
    }
    time = omp_get_wtime() - time;
    printf("General Time taken %lf\n", time);

    __m256i A_256_VEC, B_256_VEC, C_256_VEC;
    time = omp_get_wtime();
    for(i=0;i<N;i+=8)
    {
        A_256_VEC = _mm256_load_si256((__m256i *)&A[i]);
        B_256_VEC = _mm256_load_si256((__m256i *)&B[i]);
        C_256_VEC = _mm256_add_epi32(A_256_VEC, B_256_VEC);
        _mm256_store_si256((__m256i *)&C[i],C_256_VEC);
    }
    time = omp_get_wtime() - time;
    printf("AVX2 Time taken %lf\n", time);

    free(A);
    free(B);
    free(C);

    __m512i A_512_VEC, B_512_VEC, C_512_VEC;
    time = omp_get_wtime();
    for(i=0;i<N;i+=16)
    {
        A_512_VEC = _mm512_load_si512((__m512i *)&E[i]);
        B_512_VEC = _mm512_load_si512((__m512i *)&F[i]);
        C_512_VEC = _mm512_add_epi32(A_512_VEC, B_512_VEC);
        _mm512_store_si512((__m512i *)&G[i],C_512_VEC);
    }
    time = omp_get_wtime() - time;
    printf("AVX512 Time taken %lf\n", time);

    for(i=0;i<N;i++)
    {
        if(G[i] != E[i] + F[i])
        {
            printf("Not Matched !!!\n");
            break;
        }
    }
    free(E);
    free(F);
    free(G);

    return 1;
}