#include <iostream>
#include <string>
#include "util.h"
#include <chrono>
#ifndef SERIAL
#include <mpi.h>
#endif

using namespace std;

void CSRSpMV(CSR &s, int ptr_start) {
    for (int i = 0; i < s.row_num; i++) {
        s.result[i] = 0;
        for (int k = s.ptr[i]-ptr_start; k < s.ptr[i+1]-ptr_start; k++) {
            int col = s.idx[k];
            s.result[i] += s.val[k]*s.vec[col];
        }
    }
}

double getTime(double start, double end) {
    return end-start;
}

CSR serialSpMV() {
    CSR s = input();
    initVec(s);
    double *result = (double *)malloc(sizeof(double)*s.row_num);
    s.result = result;
    CSRSpMV(s, 0);
    printResult(result, s.row_num);
    free(result);
    freeAll(s);
    return s;
}

int countNNZ(CSR &s, int i, int j) {
    return s.ptr[j] - s.ptr[i];
}

#ifndef SERIAL
double MPICSRSpMV(int argc, char **argv, int nprocs, int myrank) {
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    double start, end, duration, total_start, total_end;
    double t_cal, t_gather;
    CSR s = input();
    initVec(s);
    int col_num = s.col_num;
    int row_num = s.row_num;
    int r_procs = s.row_num/nprocs;
    double *local_res;
    double *res = (double*)malloc(sizeof(double)*(s.row_num));
    int *idx = s.idx+s.ptr[myrank*r_procs];
    double *val = s.val+s.ptr[myrank*r_procs];
    int *ptr = s.ptr+myrank*r_procs;
    int left = s.row_num - myrank*r_procs;
    MPI_Barrier(MPI_COMM_WORLD);
    total_start = MPI_Wtime();
    START_TIMING
    if (myrank == nprocs-1) {  //  master
        int nnz = countNNZ(s, myrank*r_procs, s.row_num);
        local_res = (double *)malloc(sizeof(double)*left);
        CSR local_s = {left, col_num, nnz, ptr, idx, val, s.vec, local_res};
        CSRSpMV(local_s, ptr[0]);
        freeAll(s);
    } else {
        int nnz = countNNZ(s, myrank*r_procs, (myrank+1)*r_procs);
        local_res = (double *)malloc(sizeof(double)*r_procs);
        CSR local_s = {r_procs, col_num, nnz, ptr, idx, val, s.vec, local_res};
        CSRSpMV(local_s, ptr[0]);
        freeAll(s);
    }
    GET_TIME
    t_cal = duration;
    START_TIMING
    MPI_Gather(local_res, r_procs, MPI_DOUBLE, res, r_procs, MPI_DOUBLE, nprocs-1, MPI_COMM_WORLD);
    if (myrank == nprocs-1) {
        GET_TIME
        t_gather = duration;
        for (int i = nprocs*r_procs; i < row_num; i++) {
            res[i] = local_res[i-myrank*r_procs];
        }
        total_end = MPI_Wtime();
        double total_duration = total_end - total_start;
        fprintf(stderr, "cal%%:%f gather%%:%f\n", t_cal/total_duration, t_gather/total_duration);
        printResult(res, row_num);
        return total_end - total_start;
    }
    return 0;
}
#endif

int main(int argc, char **argv)
{
    #ifndef SERIAL
    MPI_Init(&argc, &argv);
    int myrank;
    double duration = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int nprocs = stoi(argv[1]);
    #endif
    for (int i = 0; i < 3; i++) {
        #ifndef SERIAL
        duration += MPICSRSpMV(argc, argv, nprocs, myrank);
        #else
        serialSpMV();
        #endif
    }
    #ifndef SERIAL
    if (myrank == nprocs-1)
        cerr << duration << "s" << endl;
    MPI_Finalize();
    #endif
    return 0;
}