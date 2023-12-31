#include "epc_mass_sum.h"
#define REAL_CELL 1

double mass_sum(int ncells, int* restrict celltype,
                    double* restrict H, double* restrict dx, double* restrict dy) {
    double summer = 0.0;
    for (int ic=0; ic<ncells; ic++) {
        if (celltype[ic] == REAL_CELL) {
            summer += H[ic]*dx[ic]*dy[ic];
        }
    }
    return(summer);
}

double mass_sum_omp(int ncells, int* restrict celltype,
                    double* restrict H, double* restrict dx, double* restrict dy) {
    double summer = 0.0;
#pragma omp parallel for reduction(+:summer)
    for (int ic=0; ic<ncells; ic++) {
        if (celltype[ic] == REAL_CELL) {
            summer += H[ic]*dx[ic]*dy[ic];
        }
    }
    return(summer);
}

double mass_sum_acc(int ncells, int* restrict celltype,
                double* restrict H, double* restrict dx, double* restrict dy) {
    double summer = 0.0;
#pragma acc parallel loop reduction(+:summer)
    for (int ic=0; ic<ncells; ic++) {
        if (celltype[ic] == REAL_CELL) {
            summer += H[ic]*dx[ic]*dy[ic];
        }
    }
    return(summer);
}
