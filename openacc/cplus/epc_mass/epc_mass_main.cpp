#include <stdio.h>
#include <time.h>

#include "epc_mass_sum.h"

#define NCELLS 1<<30
static double H[NCELLS], dx[NCELLS], dy[NCELLS];
static int celltype[NCELLS];

int main(int argc, char *argv[]) {
    double summer;
    for (size_t ic=0; ic<NCELLS; ic++) {
        H[ic] = 10.0;
        dx[ic] = 0.5;
        dy[ic] = 0.5;
        celltype[ic] = REAL_CELL;
    }
    H[NCELLS/2] = 20.0;

    clock_t start, end;

    start = clock();
    summer = mass_sum(NCELLS, celltype, H, dx, dy);
    end = clock();
    printf("no op, Mass Sum is %lf, time taken is %lf s\n", summer, (double)(end - start)/CLOCKS_PER_SEC);

    start = clock();
    summer = mass_sum_omp(NCELLS, celltype, H, dx, dy);
    end = clock();
    printf("omp, Mass Sum is %lf, time taken is %lf s\n", summer, (double)(end - start)/CLOCKS_PER_SEC);

    start = clock();
    summer = mass_sum_acc(NCELLS, celltype, H, dx, dy);
    end = clock();
    printf("acc, Mass Sum is %lf, time taken is %lf s\n", summer, (double)(end - start)/CLOCKS_PER_SEC);
}
/*
no op, Mass Sum is 335544322.500000, time taken is 0.260000 s
omp, Mass Sum is 335544322.500000, time taken is 1.550000 s
acc, Mass Sum is 335544322.500000, time taken is 0.440000 s

                    no op           omp         acc
1<<27, debug        0.26            1.55        0.44
1<<27, release      0.15            1.53        0.31
1<<30, debug        2.14            12.24       2.26
1<<30, release      1.30            12.07       1.47

没有-fno-tree-vectorize，release会有段错误
 */