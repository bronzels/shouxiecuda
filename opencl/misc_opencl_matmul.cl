// OpenCL Kernel Function
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#define DEBUGIT
__kernel void matMul(
        const int Mdim,
        const int Ndim,
        const int Kdim,
        __global const float *A,
        __global const float *B,
        __global float *C) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int k;
    float tmp;

    if( i < Mdim && j < Ndim) {
        tmp = 0.0;
        for (k = 0; k < Kdim; k++)
            tmp += A[Kdim * i + k] * B[Ndim * k + j];
        C[Ndim * i + j] = tmp;
    }
}

#define BDIM 32
#define BDIM_plus1 33

__kernel void matMulMem(
        const int Mdim,
        const int Ndim,
        const int Kdim,
        __global const float *A,
        __global const float *B,
        __global float *C,
        __global int *ptrlock) {
    __local float sA[BDIM][BDIM];
    __local float sB[BDIM][BDIM];
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    /*
    int x = get_global_id(0);
    int y = get_global_id(1);
    int x = get_local_size(0) * get_group_id(0) + tx;
    int y = get_local_size(1) * get_group_id(1) + ty;
    */
    int x = BDIM * get_group_id(0) + tx;
    int y = BDIM * get_group_id(1) + ty;

    if( x < Mdim && y < Ndim) {
        const int numTiles = Kdim / BDIM;
#ifdef DEBUGIT

        //printf("x:%d, y:%d, tx:%d, ty:%d, Mdim:%d, Ndim:%d, Kdim:%d, numTiles:%d\n", x, y, tx, ty, Mdim, Ndim, Kdim, numTiles);
#endif
        float tmp = 0.0;
        int i = 0;
        //while ( i < numTiles)
        for (;i < numTiles; i++)
        {
#ifdef DEBUGIT
            //printf("before LM sync, x:%d, y:%d, i:%d, sA[tx][ty]:%d, sB[tx][ty]:%d, tmp:%d\n", x, y, i, sA[tx][ty], sB[tx][ty], tmp);
#endif
            sA[tx][ty] = A[Kdim * x + BDIM * i + ty];
            sB[tx][ty] = B[Ndim * (BDIM * i + tx) + y];
#ifdef DEBUGIT
            //printf("after LM sync, x:%d, y:%d, i:%d, sA[tx][ty]:%d, sB[tx][ty]:%d\n", x, y, i, sA[tx][ty], sB[tx][ty]);
#endif

            //work_group_barrier(CLK_LOCAL_MEM_FENCE);
            //work_group_barrier(CLK_GLOBAL_MEM_FENCE);
            //barrier(CLK_LOCAL_MEM_FENCE);
            //barrier(CLK_GLOBAL_MEM_FENCE);
            atomic_inc(ptrlock);

            int j = 0;
            for (; j < Kdim; j++) {
#ifdef DEBUGIT
                //printf("x:%d, y:%d, i:%d, j:%d, sA[tx][j]:%d, sB[j][ty]:%d, tmp:%d\n", x, y, i, j, sA[tx][j], sB[j][ty], tmp);
#endif
                tmp += sA[tx][j] * sB[j][ty];
            }
#ifdef DEBUGIT
            //printf("x:%d, y:%d, i:%d, tmp:%d\n", x, y, i, tmp);
#endif

            //work_group_barrier(CLK_LOCAL_MEM_FENCE);
            //work_group_barrier(CLK_GLOBAL_MEM_FENCE);
            //barrier(CLK_LOCAL_MEM_FENCE);
            //barrier(CLK_GLOBAL_MEM_FENCE);
            atomic_dec(ptrlock);
        }
        C[Ndim * x + y] = tmp;
#ifdef DEBUGIT
        printf("x:%d, y:%d, tx:%d, ty:%d, tmp:%d, C[Ndim * x + y]:%d\n", x, y, tx, ty, tmp, C[Ndim * x + y]);
#endif
    }
}

__kernel void matMulMemPad(
        const int Mdim,
        const int Ndim,
        const int Kdim,
        __global const float *A,
        __global const float *B,
        __global float *C) {
    __local float sA[BDIM][BDIM_plus1];
    __local float sB[BDIM][BDIM_plus1];

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int x = BDIM * get_group_id(0) + tx;
    int y = BDIM * get_group_id(1) + ty;

    if( x < Mdim && y < Ndim) {
        const int numTiles = Kdim / BDIM;
        float tmp = 0.0;
        int i = 0;
        for (;i < numTiles; i++)
        {
            sA[tx][ty] = A[Kdim * x + BDIM * i + ty];
            sB[tx][ty] = B[Ndim * (BDIM * i + tx) + y];
            work_group_barrier(CLK_LOCAL_MEM_FENCE);

            int j = 0;
            for (; j < Kdim; j++) {
                tmp += sA[tx][j] * sB[j][ty];
            }
            work_group_barrier(CLK_LOCAL_MEM_FENCE);
        }
        C[Ndim * x + y] = tmp;
    }
}

