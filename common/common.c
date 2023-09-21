#include "common.h"

void sum_array_cpu_i(int *a, int *b, int *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void sum_array_cpu_f(float *a, float *b, float *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void print_array_i(int * input, const int array_size)
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

void print_array_f(float * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%f,", *(input + i));
		}
		else
		{
			printf("%f \n", *(input + i));
		}
	}
}

//simple initialization
void initialize_i(int * input, const int array_size,
      enum INIT_PARAM PARAM, int x)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
            *(input + i) = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
            *(input + i) = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		time_t t;
		srand((unsigned)time(&t));
		for (int i = 0; i < array_size; i++)
		{
            *(input + i) = (int)(rand() & 0xFF);
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
                *(input + i) = value;
			}
			else
			{
                *(input + i) = 0;
			}
		}
	}
	else if (PARAM == INIT_0_TO_X)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
            *(input + i) = (int)(rand() & 0xFF);
		}
	}
}

void initialize_f(float * input, const int array_size,
	enum INIT_PARAM PARAM)
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
        time_t t;
        srand((unsigned)time(&t));
        for (int i = 0; i < array_size; i++)
        {
            input[i] = 0+1.0*(rand()%RAND_MAX)/RAND_MAX *(1-0);
        }
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
        printf("not supported\n");
        exit(1);
	}
}

void compare_arrays_i(int * a, int * b, int size)
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

void compare_arrays_f(float * a, float * b, int size, float precision)
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



