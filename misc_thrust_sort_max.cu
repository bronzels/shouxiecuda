#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <vector>
#include <ctime>

#include "helper_cuda.h"
#include "helper_functions.h"

#include "common.cpph"

int thrust_example_sort()
{
    thrust::host_vector<int> h_vec(10000*20);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    std::vector<int> vec(h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), vec.begin());

    thrust::device_vector<int> d_vec=h_vec;
    clock_t time1,time2;

    time1 = clock();
    thrust::sort(d_vec.begin(), d_vec.end());
    time2 = clock();
    std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;

    time1 = clock();
    std::sort(vec.begin(),vec.end());
    time2 = clock();
    std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;

    time1 = clock();
    thrust::sort(h_vec.begin(), h_vec.end());
    time2 = clock();
    std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;

    return 0;
}
/*
0.01
0.04
0.14
thrust host_vector的性能比std要差
 */

void thrust_example_counting_iterator()
{
    // 这个例子计算序列中所有非零值的索引

    // 零和非零值的序列
    thrust::device_vector<int> stencil(8);
    stencil[0] = 0;
    stencil[1] = 1;
    stencil[2] = 1;
    stencil[3] = 0;
    stencil[4] = 0;
    stencil[5] = 1;
    stencil[6] = 0;
    stencil[7] = 1;

    // 非零索引的存储
    thrust::device_vector<int> indices(8);

    // 计数迭代器定义一个序列[0,8)
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + 8;

    // 计算非零元素的索引
    typedef thrust::device_vector<int>::iterator IndexIterator;

    IndexIterator indices_end = thrust::copy_if(first, last,
                                                stencil.begin(),
                                                indices.begin(),
                                                thrust::identity<int>());
    // 索引现在包含[1,2,5,7]

    // 打印结果
    std::cout << "found " << (indices_end - indices.begin()) << " nonzero values at indices:\n";
    thrust::copy(indices.begin(), indices_end, std::ostream_iterator<int>(std::cout, "\n"));
}

int main(void)
{
    thrust_example_sort();
    //thrust_example_counting_iterator();
    //return 0;
    //int subsize = 1 << 18;
    unsigned int size = 1 << 21;
    std::cout << "Data size: " << size << std::endl;
    //int size = 33;
    char *sep = ", ";

    bool printa = true;

    std::vector<int> vec_in(size);
    std::generate(vec_in.begin(), vec_in.end(), rand);
    std::vector<int> vec_ref(size);

    std::vector<int> vec_out(size);

    clock_t time1, time2;

    time1 = clock();
    std::copy(vec_in.begin(), vec_in.end(), vec_ref.begin());
    //std::sort(vec_ref.begin(), vec_ref.end());
    time2 = clock();
    std::cout<<"C++ sort time: ";
    std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    //std::cout<< "CPU std sorted data: "<<std::endl;
    if(printa) {
        print_vec(vec_ref, 0 * 16, 2 * 16, sep, 16);
        print_vec(vec_ref, 15 * 16, 15 * 16, sep, 16);
    }

    int max_value_ref = 0;
    int max_value_out = -1;

    time1 = clock();
    std::vector<int>::iterator iter = std::max_element(vec_in.begin(), vec_in.end());
    int position = iter - vec_in.begin();
    max_value_ref = * iter;
    time2 = clock();
    std::cout<<"C++ max time: ";
    std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    std::cout<< "CPU std max data: " << max_value_ref << " ,at: " << position <<std::endl;

    char *method_name;
    //void (*methodfunc)(std::vector<int>&, std::vector<int>&);
    thrust::device_vector<int> d_vec(vec_in.size());
    method_name = "Thrust GPU";
    time1 = clock();
    thrust::copy(vec_in.begin(), vec_in.end(), d_vec.begin());
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), vec_out.begin());
    time2 = clock();

    if(printa)
    {
        std::cout<< method_name <<" data: "<<std::endl;
        //print_vec(vec_out, 13, 23, sep, 8);
        print_vec(vec_out, 0*16, 2*16, sep, 16);
        print_vec(vec_out, (16-1)*16, 16*16, sep, 16);
    }
    std::cout<< method_name <<" sort time: ";
    std::cout<< (double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    std::cout<< method_name <<" sort result compare: ";
    if(vec_ref == d_vec)
        std::cout<< "same"<<std::endl;
    else
        std::cout<< "different"<<std::endl;

    time1 = clock();
    /*
    thrust::copy(vec_in.begin(), vec_in.end(), d_vec.begin());
    thrust::device_vector<int>::iterator iter_gpu = thrust::max_element(d_vec.begin(), d_vec.end());
    int position_gpu = iter_gpu - d_vec.begin();
    max_value_out = * iter_gpu;
    */
    int *ptr_in = vec_in.data();
    int *dptr_in;
    checkCudaErrors(cudaMalloc((void **)&dptr_in, size * sizeof(int)));
    cudaMemcpy(dptr_in, ptr_in, size * sizeof(int), cudaMemcpyHostToDevice);
    thrust::device_ptr<int> dptr_thr_in = thrust::device_pointer_cast(dptr_in);
    thrust::device_ptr<int> iter_gpu = thrust::max_element(dptr_thr_in, dptr_thr_in + size - 1);
    int position_gpu = iter_gpu - dptr_thr_in;
    max_value_out = * iter_gpu;
    time2 = clock();
    std::cout<< method_name << " max time: ";
    std::cout<<(double)(time2-time1)/CLOCKS_PER_SEC<<std::endl;
    std::cout<< method_name << " max data: " << max_value_ref << " ,at: " << position_gpu <<std::endl;

    std::cout<< method_name <<" max result compare: ";
    if(max_value_ref == max_value_out && position == position_gpu)
        std::cout<< "same"<<std::endl;
    else
        std::cout<< "different"<<std::endl;

}
/*
Data size: 4194304
C++ sort: 0.83
CPU std data:
614, 858, 880, 1210, 2126, 2333, 2508, 2552, 3722, 4125, 4686, 4855, 5677, 5712, 6517, 6700,
6900, 7091, 8204, 8205, 10014, 10991, 11425, 11473, 12114, 12352, 13590, 13866, 13988, 15553, 16174, 16467,
2147471975, 2147471995, 2147474548, 2147474987, 2147476105, 2147476900, 2147477011, 2147477294, 2147478292, 2147478980, 2147479638, 2147479881, 2147480021, 2147481194, 2147481384, 2147482567,
Thrust GPU data:
614, 858, 880, 1210, 2126, 2333, 2508, 2552, 3722, 4125, 4686, 4855, 5677, 5712, 6517, 6700,
6900, 7091, 8204, 8205, 10014, 10991, 11425, 11473, 12114, 12352, 13590, 13866, 13988, 15553, 16174, 16467,
2147471975, 2147471995, 2147474548, 2147474987, 2147476105, 2147476900, 2147477011, 2147477294, 2147478292, 2147478980, 2147479638, 2147479881, 2147480021, 2147481194, 2147481384, 2147482567,
Thrust GPU sort time: 0.06
Thrust GPU sort result compare: same


Data size: 67108864
C++ sort time: 16.44
CPU std sorted data:
C++ max time: 0.3
CPU std max data: 2147483611 ,at: 13068230
Thrust GPU sort time: 0.88
Thrust GPU sort result compare: same
Thrust GPU max time: 0.76
Thrust GPU max data: 2147483611 ,at: 13068230
Thrust GPU max result compare: same


                     all copy in(22)    only sort(22)    all copy in(26)    only sort(26)
std sort             0.85               0.85             15.81              15.85
thrust sort          0.64               0.06             1.56               0.83

all copied
                            26                 29               31
std max                     0.3                2.27             9.37
thrust max                  0.76               6.02             23.96
thrust max(device ptr)      0.66               5.16             cudaMalloc失败

 */