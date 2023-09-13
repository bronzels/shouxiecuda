#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/equal.h>
#include <thrust/iterator/constant_iterator.h>
#include <cublas_v2.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <cstdlib>
#define USECPSEC 1000000ULL

long long dtime_usec(unsigned long long start){

    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

struct dp
{
    float *A, *B;
    int m,n,r;
    dp(float *_A, float *_B, int _m, int _n, int _r): A(_A), B(_B), m(_m), n(_n), r(_r) {};
    __host__ __device__
    float operator()(size_t idx){
        float sum = 0.0f;
        int row = idx/r;
        int col = idx - (row*r); // cheaper modulo
        for (int i = 0; i < m; i++)
            sum += A[col + row*i] * B[col + row*i];
        return sum;}
};

const int dsd = 200;
int main(int argc, char *argv[]){
    int ds = dsd;
    if (argc > 1) ds = atoi(argv[1]);
    const int n = ds;
    const int m = ds;
    const int r = ds;
    // data setup
    thrust::device_vector<float> data(n*m,1);
    thrust::device_vector<float> other(m*r,1);
    thrust::device_vector<float> result(n*r,0);
    // method 1
    //let's pretend that other is (already) transposed for efficient memory access by thrust
    // therefore each dot-product is formed using a row of data and a row of other
    long long dt = dtime_usec(0);
    if (ds < 201){
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < r;++j)
            {
                result[i*r+ j] = thrust::inner_product(data.begin()+(i*m), data.begin()+((i+1)*m),other.begin()+(j*m), 0.0f);
            }
        }
        cudaDeviceSynchronize();
        dt = dtime_usec(dt);
        if (thrust::equal(result.begin(), result.end(), thrust::constant_iterator<float>(m)))
            std::cout << "method 1 time: " << dt/(float)USECPSEC << "s" << std::endl;
        else
            std::cout << "method 1 failure" << std::endl;
    }
    thrust::fill(result.begin(), result.end(), 0);
    cudaDeviceSynchronize();
// method 2
    //let's pretend that data is (already) transposed for efficient memory access by thrust
    // therefore each dot-product is formed using a column of data and a column of other
    dt = dtime_usec(0);
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n*r), result.begin(), dp(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(other.data()), m, n, r));
    cudaDeviceSynchronize();
    dt = dtime_usec(dt);
    if (thrust::equal(result.begin(), result.end(), thrust::constant_iterator<float>(m)))
        std::cout << "method 2 time: " << dt/(float)USECPSEC << "s" << std::endl;
    else
        std::cout << "method 2 failure" << std::endl;
// method 3
    // once again, let's pretend the data is ready to go for CUBLAS
    cublasHandle_t h;
    cublasCreate(&h);
    thrust::fill(result.begin(), result.end(), 0);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaDeviceSynchronize();
    dt = dtime_usec(0);
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, n, r, m, &alpha, thrust::raw_pointer_cast(data.data()), n, thrust::raw_pointer_cast(other.data()), m, &beta, thrust::raw_pointer_cast(result.data()), n);
    cudaDeviceSynchronize();
    dt = dtime_usec(dt);
    if (thrust::equal(result.begin(), result.end(), thrust::constant_iterator<float>(m)))
        std::cout << "method 3 time: " << dt/(float)USECPSEC << "s" << std::endl;
    else
        std::cout << "method 3 failure" << std::endl;
}
