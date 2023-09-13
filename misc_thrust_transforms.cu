#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include "cublas_v2.h"
#include <iostream>
#include <algorithm>

#include "common.h"

void saxpy_slow(float A, thrust::device_vector<float>&X, thrust::device_vector<float>& Y)
{
    thrust::device_vector<float> temp(X.size());
    thrust::fill(temp.begin(), temp.end(), A);
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
    float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};
void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

int main(void)
{
    thrust::device_vector<float> X(10);
    thrust::device_vector<float> Y(10);
    thrust::device_vector<float> Z(10);

    thrust::sequence(X.begin(), X.end());
    //thrust::generate(X.begin(), X.end(), rand); //不能在device上使用

    thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<float>());
    thrust::host_vector<float> Y_h = Y;
    print_array(Y_h.data(), 10);

    thrust::fill(Z.begin(), Z.end(), 2);
    thrust::host_vector<float> Z_h = Z;
    print_array(Z_h.data(), 10);

    thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), [=] __device__(float x, float y){
        return pow(x, 2) + pow(y, 2);
    });
    Y_h = Y;
    print_array(Y_h.data(), 10);

    thrust::replace(Y.begin(), Y.end(), 1, 10);
    Y_h = Y;
    print_array(Y_h.data(), 10);

    float A = 10.;
    int size = 1 << 29;
    thrust::host_vector<float> Xs_h(size);
    std::generate(Xs_h.begin(), Xs_h.end(), drand48);
    thrust::device_vector<float> Xs(size);
    Xs = Xs_h;
    thrust::host_vector<float> Ysa_h(size);
    std::generate(Ysa_h.begin(), Ysa_h.end(), drand48);
    thrust::device_vector<float> Yslow(size);
    Yslow = Ysa_h;
    thrust::device_vector<float> Yfast(size);
    Yfast = Ysa_h;

    clock_t time1, time2;

    time1 = clock();
    saxpy_slow(A, Xs, Yslow);
    time2 = clock();
    std::cout<<"datasize: " << size << ", saxpy_slow time: ";
    std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;

    time1 = clock();
    saxpy_fast(A, Xs, Yfast);
    time2 = clock();
    std::cout<<"datasize: " << size << ", saxpy_fast time: ";
    std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;

    bool printfa = true;

    thrust::host_vector<float> Yslow_h = Yslow;
    float *dataYslow = Yslow_h.data();
    if (printfa)
        print_array(dataYslow + size / 2 , 32);
    thrust::host_vector<float> Yfast_h = Yfast;
    float *dataYfast = Yfast_h.data();
    if (printfa)
        print_array(dataYfast + size / 2 , 32);
    std::vector<float> v_slow(dataYslow, dataYslow + size -1);
    std::vector<float> v_fast(dataYfast, dataYfast + size -1);
    std::cout << "Compare slow with fast:" << std::endl;
    if(v_slow == v_fast)
        std::cout << "same" << std::endl;
    else
        std::cout << "different" << std::endl;

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    float *dev_a, *dev_b;
    cudaMalloc(&dev_a, sizeof(float) * size);
    cudaMalloc(&dev_b, sizeof(float) * size);
    float *a = Xs_h.data();
    float *b = Ysa_h.data();
    cublasSetVector(size, sizeof(float), a, 1, dev_a, 1);
    cublasSetVector(size, sizeof(float), b, 1, dev_b, 1);
    //time1 = clock();
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    cublasSaxpy_v2(handle, size, &A, dev_a, 1, dev_b, 1);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    //printf("Kernel execution time using events : %f \n",time);
    float *blasout = (float*)malloc(size *sizeof(float));
    cublasGetVector(size, sizeof(float), dev_b, 1, blasout, 1);
    //time2 = clock();
    //1<<27，clock计算cublas执行时间为0，包括了getvector以后是0.1
    std::cout<<"datasize: " << size << ", cublas time: ";
    //std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    std::cout<<time/1000.<<std::endl;
    if (printfa)
        print_array(blasout + size / 2 , 32);
    std::vector<float> v_blasout(blasout, blasout+size);
    std::cout << "Compare blas with fast:" << std::endl;
    compare_arrays(blasout, dataYfast, size, 1e-4);

    return 0;
}
/*
datasize: 4194304, saxpy_slow time: 0.12
datasize: 4194304, saxpy_fast time: 0.12
same

datasize: 33554432, saxpy_slow time: 0.88
datasize: 33554432, saxpy_fast time: 0.35
same

datasize: 134217728, saxpy_slow time: 3.53
datasize: 134217728, saxpy_fast time: 1.39
same
                            saxpy_slow              saxpy_fast                  cublas
datasize: 4194304           0.12                    0.12
datasize: 33554432          0.88                    0.35
datasize: 134217728         3.53                    1.39                        0.00479789
datasize: 536870912         14.01                   5.55                        0.0191455
 */