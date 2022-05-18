#ifndef UTIL_H
#define UTIL_H 1

#include <iostream>
#include <string>
#ifndef SERIAL
#include <mpi.h>
#endif

#ifdef DEBUG
#define OUTLINE std::cout << __LINE__ << std::endl;
#define DEBUGPRINT(...) printf(__VA_ARGS__);
#else
#define OUTLINE
#define DEBUGPRINT(...)
#endif
#define START_TIMING start = MPI_Wtime();
#define END_TIMING end = MPI_Wtime();
#define GET_TIME END_TIMING duration = getTime(start, end);

typedef struct {
    int row_num;
    int col_num;
    int nnz;
    int *ptr, *idx;
    double *val, *vec, *result;
}CSR;

void mallocAll(CSR &s);
void freeAll(CSR &s);
CSR input();
void printResult(double *result, int row_num);
void initVec(CSR &s);

#endif
