#include "common.cpph"
using namespace std;

#include <iterator>
#include <algorithm>
#include <iostream>
#include <dlfcn.h>

template <typename T>
void sum_arry_cpu(T *a, T *b, T *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}
template
void sum_arry_cpu(int *a, int *b, int *c, int size);
template
void sum_arry_cpu(float *a, float *b, float *c, int size);

/*
template <typename T, char *sep, int tablen>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    int v_size = v.size();
    //cout << "size:" << size << ", tablen:" << tablen << ", (size + tablen - 1) / tablen:" << (size + tablen - 1) / tablen << std::endl;
    if (!v.empty()) {
        for(int i = 0; i < (v_size + tablen - 1) / tablen; i ++)
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

template <typename T> void print_vec(const std::vector<T> &v, int start, int end, char *sep, int tablen) {
    int size = end - start;
    int v_size = v.size();
    //cout << "size:" << size << ", tablen:" << tablen << ", (size + tablen - 1) / tablen:" << (size + tablen - 1) / tablen << std::endl;
    if (!v.empty()) {
        for(int i = 0; i < (size + tablen - 1) / tablen; i ++)
        {
            /*
            int offset = TABLEN * i;
            std::vector<T> last  = v.begin() + std::min(size - 1, offset + TABLEN - 1);
            std::copy(first, last, std::ostream_iterator<T>(out, SEP));
            */
            int offset = start + tablen * i;
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
template void print_vec(const std::vector<int> &v, int start, int end, char *sep, int tablen);
template void print_vec(const std::vector<unsigned int> &v, int start, int end, char *sep, int tablen);

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

void print_array(int * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
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

void print_array(float * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
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

void print_matrix(int * matrix, int nx, int ny)
{
    for (int ix = 0; ix < nx; ix++)
	{
        for (int iy = 0; iy < ny; iy++)
		{
            printf("%d ", matrix[ny * ix + iy]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_matrix(float * matrix, int nx, int ny)
{
    for (int ix = 0; ix < nx; ix++)
	{
        for (int iy = 0; iy < ny; iy++)
		{
			printf("%f ", matrix[ny * ix + iy]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_arrays_toafile_side_by_side(float*a, float*b, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " <<a[i] << " - " << b[i] << "\n";
		}
		file.close();
	}
}

void print_arrays_toafile_side_by_side(int*a, int*b, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " << a[i] << " - " << b[i] << "\n";
		}
		file.close();
	}
}

void print_arrays_toafile(int*a, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " << a[i] << "\n";
		}
		file.close();
	}
}



int* get_matrix(int rows, int columns)
{
	int mat_size = rows * columns;
	int mat_byte_size = sizeof(int)*mat_size;

	int * mat = (int*)malloc(mat_byte_size);

	for (int i = 0; i < mat_size; i++)
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
void initialize(int * input, const unsigned int array_size,
	INIT_PARAM PARAM, int x)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		time_t t;
		srand((unsigned)time(&t));
		for (int i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
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
		for (int i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
}

void initialize(float * input, const unsigned int array_size,
	INIT_PARAM PARAM)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = 1.;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
        uniform_real_distribution<float> u(-1, 1);
        default_random_engine e(time(NULL));
		for (int i = 0; i < array_size; i++)
		{
			//input[i] = u(e);
            input[i] = 0+1.0*(rand()%RAND_MAX)/RAND_MAX *(1-0);
        }
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
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
int reduction_cpu(int * input, const int size)
{
	int sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += input[i];
	}
	return sum;
}

//cpu transpose
template <class T>
void mat_transpose_cpu(T * mat, T * transpose, int nx, int ny)
{
	for (int  iy = 0; iy < ny; iy++)
	{
		for (int  ix = 0; ix < nx; ix++)
		{
			transpose[ix * ny + iy] = mat[iy * nx + ix];
		}
	}
}
template void mat_transpose_cpu(int * mat, int * transpose, int nx, int ny);
template void mat_transpose_cpu(float * mat, float * transpose, int nx, int ny);

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
void compare_arrays(T * a, T * b, int size)
{
	for (int  i = 0; i < size; i++)
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
void compare_arrays(int * a, int * b, int size);
template
void compare_arrays(unsigned int * a, unsigned int * b, int size);

template <class T>
void compare_arrays(T * a, T * b, int size, T precision)
{
	for (int i = 0; i < size; i++)
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
void compare_arrays(float * a, float * b, int size, float precision);
template
void compare_arrays(double * a, double * b, int size, double precision);

void compare_matrixes(int *a, int *b, int m, int n)
{
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
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

void compare_matrixes(float *a, float *b, int m, int n, float precision)
{
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
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

void printData(char *msg, int *in, const int size)
{
	printf("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf("%5d", in[i]);
		fflush(stdout);
	}

	printf("\n");
	return;
}

void sum_array_cpu(float* a, float* b, float *c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}


