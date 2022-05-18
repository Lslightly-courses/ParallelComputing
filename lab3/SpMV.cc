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

typedef struct Statistic {
    double total_time;
    double collect_per;
    double cal_per;
}Statistic;

#ifndef SERIAL
Statistic MPICSRSpMV(int argc, char **argv, int nprocs, int myrank) {
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    double start, end, duration, total_start, total_end;
    double t_cal, t_collect;
    double tag_res = 1;
    CSR s = input();
    initVec(s);
    int col_num = s.col_num;
    int row_num = s.row_num;
    int r_procs = s.row_num/nprocs;
    double *local_res;
    double *res;
    int *idx = s.idx+s.ptr[myrank*r_procs];
    double *val = s.val+s.ptr[myrank*r_procs];
    int *ptr = s.ptr+myrank*r_procs;
    int left = row_num - myrank*r_procs;
    MPI_Barrier(MPI_COMM_WORLD);
    total_start = MPI_Wtime();
    START_TIMING
    if (myrank == nprocs-1) {  //  master
        res = (double*)malloc(sizeof(double)*(s.row_num));
        int nnz = countNNZ(s, myrank*r_procs, row_num);
        local_res = (double *)malloc(sizeof(double)*left);
        CSR local_s = {left, col_num, nnz, ptr, idx, val, s.vec, local_res};
        CSRSpMV(local_s, ptr[0]);
        for (int i = (nprocs-1)*r_procs; i < row_num; i++) {
            res[i] = local_res[i-(nprocs-1)*r_procs];
        }
    } else {
        int nnz = countNNZ(s, myrank*r_procs, (myrank+1)*r_procs);
        local_res = (double *)malloc(sizeof(double)*r_procs);
        CSR local_s = {r_procs, col_num, nnz, ptr, idx, val, s.vec, local_res};
        CSRSpMV(local_s, ptr[0]);
        MPI_Send(local_res, r_procs, MPI_DOUBLE, nprocs-1, tag_res, MPI_COMM_WORLD);
        free(local_res);
    }
    freeAll(s);
    GET_TIME
    t_cal = duration;
    START_TIMING
    if (myrank == nprocs-1) {
        MPI_Status status;
        for (int i = 0; i < nprocs-1; i++) {
            MPI_Recv(res+i*r_procs, r_procs, MPI_DOUBLE, i, tag_res, MPI_COMM_WORLD, &status);
        }
        GET_TIME
        t_collect = duration;
        total_end = MPI_Wtime();
        double total_duration = total_end - total_start;
        printResult(res, row_num);
        free(res);
        free(local_res);
        return {total_duration, t_collect/total_duration, t_cal/total_duration};
    }
    return {};
}
#endif

int main(int argc, char **argv)
{
    int LOOPS = stoi(argv[1]);
    #ifndef SERIAL
        MPI_Init(&argc, &argv);
        int myrank;
        double duration = 0;
        double cal_per, collect_per = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        int nprocs = stoi(argv[2]);
    #endif
    for (int i = 0; i < LOOPS; i++) {
        #ifndef SERIAL
            auto tmp_result = MPICSRSpMV(argc, argv, nprocs, myrank);
            duration += tmp_result.total_time;
            cal_per += tmp_result.cal_per;
            collect_per += tmp_result.collect_per;
        #else
            serialSpMV();
        #endif
    }
    #ifndef SERIAL
        if (myrank == nprocs-1)
            fprintf(stderr, "%f %f%% %f%%\n", duration/LOOPS, cal_per/LOOPS*100, collect_per/LOOPS*100);
        MPI_Finalize();
    #endif
    return 0;
}