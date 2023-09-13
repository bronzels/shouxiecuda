#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "cuda_common.cuh"

void scan_inclusive_cpu(float*, float*, int);

__global__ void scan_inclusive_gpu(float*, float*, int);