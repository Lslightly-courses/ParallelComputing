#include "util.h"
#include <iostream>

using namespace std;

void mallocAll(CSR &s) {
    s.ptr = (int *)malloc(sizeof(int)*(s.row_num+1));
    s.idx = (int *)malloc(sizeof(int)*s.nnz);
    s.val = (double *)malloc(sizeof(double)*s.nnz);
    s.vec = (double *)malloc(sizeof(double)*s.col_num);
}

CSR input() {
    freopen("input.txt", "r", stdin);
    CSR s;
    int row, col;
    double v;
    cin >> s.row_num >> s.col_num >> s.nnz;
    mallocAll(s);
    cin >> row >> col >> v;
    int count = 0;
    bool new_row = true;
    int i;
    for (i = 0; i < s.row_num; i++) {
        if (row > i) {
            s.ptr[i] = count;
            new_row = true;
            continue;
        }
        if (new_row) {
            s.ptr[i] = count;
            s.idx[count] = col;
            s.val[count] = v;
            count++;
            if (count == s.nnz) {
                i++;
                break;
            }
            new_row = false;
        }
        while (!new_row) {
            cin >> row >> col >> v;
            if (row == i) {
                s.idx[count] = col;
                s.val[count] = v;
                count++;
                if (count == s.nnz) break;
            } else {
                new_row = true;
                break;
            }
        }
        if (count == s.nnz) {
            i++;
            break;
        }
    }
    for (; i <= s.row_num; i++) {
        s.ptr[i] = count;
    }
    // for (int i = 0; i < s.row_num+1; i++) {
    //     DEBUGPRINT("row%d ptr[%d]=%d\n", i, i, s.ptr[i])
    // }
    return s;
}

void freeAll(CSR &s) {
    free(s.ptr);
    free(s.idx);
    free(s.val);
    free(s.vec);
}

void printResult(double *result, int row_num) {
    for (int i = 0; i < row_num; i++) {
        cout << result[i] << endl;
    }
}

void initVec(CSR &s) {
    for (int i = 0; i < s.col_num; i++) {
        s.vec[i] = i+1;
    }
}


