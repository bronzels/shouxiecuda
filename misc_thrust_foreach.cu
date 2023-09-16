#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <iostream>

#include "common.hpp"
#include "dotmethod.h"

struct myFunctor
{
    const int *m_vec1;
    const int *m_vec2;
    int *m_result;
    size_t v1size;
    int method;
    myFunctor(thrust::device_vector<int> const& vec1, thrust::device_vector<int> const& vec2, thrust::device_vector<int>& result, int method_input)
    {
        m_vec1 = thrust::raw_pointer_cast(vec1.data());
        m_vec2 = thrust::raw_pointer_cast(vec2.data());
        m_result = thrust::raw_pointer_cast(result.data());
        v1size = vec1.size();
        method = method_input;
    }

    __host__ __device__
    void operator()(const size_t x) const
    {
        size_t i = x%v1size;
        size_t j = x/v1size;
        if(method == DotMethod::Enum::plus)
                m_result[i + j * v1size] = m_vec1[i] + m_vec2[j];
        else if(method == DotMethod::Enum::minus)
                m_result[i + j * v1size] = m_vec1[i] - m_vec2[j];
        else if(method == DotMethod::Enum::equal)
                m_result[i + j * v1size] = m_vec1[i] == m_vec2[j] ? 1 : 0;
        else if(method == DotMethod::Enum::multiply)
                m_result[i + j * v1size] = m_vec1[i] * m_vec2[j];
    }
};

int main()
{
    const int N = 2;
    const int M = 3;
    thrust::host_vector<int> vec1_host(N);
    thrust::host_vector<int> vec2_host(M);
    vec1_host[0] = 1;
    vec1_host[1] = 5;
    vec2_host[0] = -3;
    vec2_host[1] = 42;
    vec2_host[2] = 1;

    thrust::device_vector<int> vec1_dev = vec1_host;
    thrust::device_vector<int> vec2_dev = vec2_host;

    thrust::device_vector<int> result_dev(vec1_host.size() * vec2_host.size());

    for (int i = DotMethod::Enum::plus; i <= DotMethod::Enum::multiply; i ++)
    {
        thrust::for_each_n(thrust::device, thrust::counting_iterator<size_t>(0), (N*M),
                           myFunctor(vec1_dev, vec2_dev, result_dev, i));
        thrust::host_vector<int> result_host = result_dev;
        print_matrix(result_host.data(), M, N);
    }

    return 0;
}