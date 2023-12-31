#ifndef _MASS_SUM_H
#define _MASS_SUM_H
#define REAL_CELL 1

double mass_sum(int ncells, int *celltype, double *H, double *dx, double *dy);
double mass_sum_omp(int ncells, int *celltype, double *H, double *dx, double *dy);
double mass_sum_acc(int ncells, int *celltype, double *H, double *dx, double *dy);
#endif