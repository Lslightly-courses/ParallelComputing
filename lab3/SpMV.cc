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

void cutRows(int cut[], int nprocs, int r_procs, CSR &s) {
    int row_num = s.row_num;
    cut[0] = 0;
    int row = 0;
    for (int i = 1; i <= nprocs; i++) {
        while (i*r_procs > s.ptr[row]) {
            row++;
        }
        cut[i] = row;
    }
    cut[nprocs] = s.row_num;
}

#ifndef SERIAL
Statistic MPICSRSpMV(int argc, char **argv, int nprocs, int myrank) {
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    double start, end, duration, total_start, total_end;
    double t_cal, t_collect, total_duration = 1;
    double tag_res = 1;
    CSR s = input();
    initVec(s);
    int col_num = s.col_num;
    int row_num = s.row_num;
    int r_procs = s.nnz/nprocs;
    double *local_res;
    double *res;
    int *cut = (int *)malloc(sizeof(int)*(nprocs+1));
    cutRows(cut, nprocs, r_procs, s);
    int *ptr = s.ptr+cut[myrank];
    int *idx = s.idx+*ptr;
    double *val = s.val+*ptr;
    int left = cut[nprocs] - cut[nprocs-1];
    MPI_Barrier(MPI_COMM_WORLD);
    total_start = MPI_Wtime();
    START_TIMING
    if (myrank == nprocs-1) {  //  master
        res = (double*)malloc(sizeof(double)*(s.row_num));
        int nnz = countNNZ(s, cut[myrank], cut[myrank+1]);
        local_res = (double *)malloc(sizeof(double)*left);
        CSR local_s = {left, col_num, nnz, ptr, idx, val, s.vec, local_res};
        CSRSpMV(local_s, ptr[0]);
        for (int i = cut[myrank]; i < cut[myrank+1]; i++) {
            res[i] = local_res[i-cut[myrank]];
        }
    } else {
        int nnz = countNNZ(s, cut[myrank],  cut[myrank+1]);
        int local_row_num = cut[myrank+1]-cut[myrank];
        local_res = (double *)malloc(sizeof(double)*(local_row_num));
        CSR local_s = {local_row_num, col_num, nnz, ptr, idx, val, s.vec, local_res};
        CSRSpMV(local_s, ptr[0]);
        MPI_Send(local_res, local_row_num, MPI_DOUBLE, nprocs-1, tag_res, MPI_COMM_WORLD);
    }
    freeAll(s);
    GET_TIME
    t_cal = duration;
    START_TIMING
    if (myrank == nprocs-1) {
        MPI_Status status;
        for (int i = 0; i < nprocs-1; i++) {
            MPI_Recv(res+cut[i], cut[i+1]-cut[i], MPI_DOUBLE, i, tag_res, MPI_COMM_WORLD, &status);
        }
        GET_TIME
        t_collect = duration;
        total_end = MPI_Wtime();
        total_duration = total_end - total_start;
        printResult(res, row_num);
        free(res);
    }
    free(cut);
    free(local_res);
    return {total_duration, t_collect/total_duration, t_cal/total_duration};
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
            if (myrank == nprocs-1) {
                duration += tmp_result.total_time;
                cal_per += tmp_result.cal_per;
                collect_per += tmp_result.collect_per;
            }
        #else
            serialSpMV();
        #endif
    }
    #ifndef SERIAL
        if (myrank == nprocs-1)
            //  这里进行取平均
            fprintf(stderr, "%f %f%% %f%%\n", duration/LOOPS*1000, cal_per/LOOPS*100, collect_per/LOOPS*100);
        MPI_Finalize();
    #endif
    return 0;
}