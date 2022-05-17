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
    CSRSpMV(s, 0);
    printResult(s);
    freeAll(s);
    return s;
}

int countNNZ(CSR &s, int i, int j) {
    return s.ptr[j] - s.ptr[i];
}

#ifndef SERIAL
void MPICSRSpMV(int argc, char **argv, int nprocs) {
    int myrank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int tag_args = 0, tag_ints_pack = 1, tag_doubles_pack = 2, tag_result = 8;
    double start, end, duration;
    double t_idx_ptr = 0, t_val_vec = 0, t_send[3] = {0}, t_cal = 0, t_recv;
    if (myrank == nprocs-1) {  //  master
        int i;
        CSR s = input();
        initVec(s);

        double total_start = MPI_Wtime();

        int r_procs = s.row_num/nprocs;
        MPI_Request args_reqs;
        MPI_Request int_reqs;
        MPI_Request double_reqs;
        for (i = 0; i < nprocs-1; i++) {
            int args[5];
            int nnz = countNNZ(s, i*r_procs, (i+1)*r_procs);

            args[0] = nnz;
            args[1] = r_procs;
            args[2] = s.col_num;

            START_TIMING
            int ints_pack_size;
            MPI_Pack_size(nnz+r_procs+1, MPI_INT, MPI_COMM_WORLD, &ints_pack_size);
            auto ints_buffer = (int *)malloc(ints_pack_size);
            int int_pos = 0;
            MPI_Pack(s.idx+s.ptr[i*r_procs], nnz, MPI_INT, ints_buffer, ints_pack_size, &int_pos, MPI_COMM_WORLD);
            MPI_Pack(s.ptr+i*r_procs, r_procs+1, MPI_INT, ints_buffer, ints_pack_size, &int_pos, MPI_COMM_WORLD);
            args[3] = int_pos;
            GET_TIME
            t_idx_ptr += duration;
            
            START_TIMING
            int doubles_pack_size;
            MPI_Pack_size(nnz+s.col_num, MPI_DOUBLE, MPI_COMM_WORLD, &doubles_pack_size);
            auto doubles_buffer = (double *)malloc(doubles_pack_size);
            int double_pos = 0;
            MPI_Pack(s.val+s.ptr[i*r_procs], nnz, MPI_DOUBLE, doubles_buffer, doubles_pack_size, &double_pos, MPI_COMM_WORLD);
            MPI_Pack(s.vec, s.col_num, MPI_DOUBLE, doubles_buffer, doubles_pack_size, &double_pos, MPI_COMM_WORLD);
            args[4] = double_pos;
            GET_TIME
            t_val_vec += duration;

            MPI_Status status;
            START_TIMING
            MPI_Send(args, 5, MPI_INT, i, tag_args, MPI_COMM_WORLD);
            GET_TIME
            t_send[0] += duration;
            START_TIMING
            MPI_Send(ints_buffer, int_pos, MPI_PACKED, i, tag_ints_pack, MPI_COMM_WORLD);
            GET_TIME
            t_send[1] += duration;
            START_TIMING
            MPI_Send(doubles_buffer, double_pos, MPI_PACKED, i, tag_doubles_pack, MPI_COMM_WORLD);
            GET_TIME
            t_send[2] += duration;
        }
        int nnz = countNNZ(s, i*r_procs, (i+1)*r_procs);
        int *idx = s.idx+s.ptr[i*r_procs];
        double *val = s.val+s.ptr[i*r_procs];
        double *result = s.result+i*r_procs;
        int *ptr = s.ptr+i*r_procs;
        CSR new_s = {s.row_num-i*r_procs, s.col_num, nnz, ptr, idx, val, s.vec, result};
        START_TIMING
        CSRSpMV(new_s, new_s.ptr[0]);
        GET_TIME
        t_cal = duration;
        MPI_Status status;
        MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request)*(nprocs-1));
        START_TIMING
        for (int i = 0; i < nprocs-1; i++) {    //  summary
            MPI_Irecv(s.result+i*r_procs, r_procs, MPI_DOUBLE, i, tag_result, MPI_COMM_WORLD, &req[i]);
        }
        for (int i = 0; i < nprocs-1;i++) {
            MPI_Wait(&req[i], &status);
        }
        GET_TIME
        t_recv = duration;
        free(req);
        double total_duration = MPI_Wtime()-total_start;
        cerr << total_duration << endl;
        fprintf(stderr, "%f\t%f\t%f\t%f\t%f\t%f\t%f\n", t_idx_ptr/total_duration, t_val_vec/total_duration, t_send[0]/total_duration, t_send[1]/total_duration, t_send[2]/total_duration, t_cal/total_duration, t_recv/total_duration);

        printResult(s);
        freeAll(s);
    } else {
        int nnz = 0;
        int r_procs = 0;
        int col_num = 0;
        MPI_Status status;
        int args[5];

        int int_pos, double_pos;
        MPI_Recv(args, 5, MPI_INT, nprocs-1, tag_args, MPI_COMM_WORLD, &status);
        nnz = args[0];
        r_procs = args[1];
        col_num = args[2];
        int_pos = args[3];
        double_pos = args[4];
        DEBUGPRINT("nnz%d, r_procs%d, col_num%d, int_pos%d, double_pos%d\n", nnz, r_procs, col_num, int_pos, double_pos);

        auto ptr = (int *)malloc(sizeof(int)*(r_procs+1));
        auto idx = (int *)malloc(sizeof(int)*nnz);
        auto val = (double *)malloc(sizeof(double)*nnz);
        auto vec = (double *)malloc(sizeof(double)*col_num);
        auto result = (double *)malloc(sizeof(double)*r_procs);


        int ints_pack_size;
        MPI_Pack_size(nnz+r_procs+1, MPI_INT, MPI_COMM_WORLD, &ints_pack_size);
        auto ints_buffer = (int *)malloc(ints_pack_size);
        MPI_Request int_req, double_req;
        MPI_Recv(ints_buffer, int_pos, MPI_PACKED, nprocs-1, tag_ints_pack, MPI_COMM_WORLD, &status);
        int_pos = 0;
        DEBUGPRINT("ints buffer recv complete\n")
        MPI_Unpack(ints_buffer, ints_pack_size, &int_pos, idx, nnz, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(ints_buffer, ints_pack_size, &int_pos, ptr, r_procs+1, MPI_INT, MPI_COMM_WORLD);
        DEBUGPRINT("unpack ints_buffer complete\n")

        int doubles_pack_size;
        MPI_Pack_size(nnz+col_num, MPI_DOUBLE, MPI_COMM_WORLD, &doubles_pack_size);
        auto doubles_buffer = (double *)malloc(doubles_pack_size);
        MPI_Recv(doubles_buffer, double_pos, MPI_DOUBLE, nprocs-1, tag_doubles_pack, MPI_COMM_WORLD, &status);
        double_pos = 0;
        MPI_Unpack(doubles_buffer, doubles_pack_size, &double_pos, val, nnz, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(doubles_buffer, doubles_pack_size, &double_pos, vec, col_num, MPI_DOUBLE, MPI_COMM_WORLD);
        DEBUGPRINT("unpack doubles_buffer complete\n")

        CSR local_s = { r_procs, col_num, nnz, ptr, idx, val, vec, result };
        DEBUGPRINT("rank:%d\n", myrank)
        CSRSpMV(local_s, ptr[0]);
        MPI_Request req;
        MPI_Isend(result, r_procs, MPI_DOUBLE, nprocs-1, tag_result, MPI_COMM_WORLD, &req);
        free(ptr);
        free(idx);
        free(val);
        free(vec);
        free(result);
    }
}
#endif

int main(int argc, char **argv)
{
    #ifndef SERIAL
    MPI_Init(&argc, &argv);
    #endif
    for (int i = 0; i < 1; i++) {
        #ifndef SERIAL
        MPICSRSpMV(argc, argv, stoi(argv[1]));
        #else
        serialSpMV();
        #endif
    }
    #ifndef SERIAL
    MPI_Finalize();
    #endif
    return 0;
}