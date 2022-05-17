import scanpy as sc
from typing import *
import sys
from scipy import sparse

data = sc.read_mtx(sys.argv[1])
X = sparse.csr_matrix(data.X)
row_num, col_num = X.shape
with open('input.txt', 'w', encoding='utf-8') as f:
    f.write('{} {} {}\n'.format(row_num, col_num, X.nnz))
    for row in range(0, row_num):
        for j in range(X.indptr[row], X.indptr[row+1]):
            f.write("{} {} {}\n".format(row, X.indices[j], X.data[j]))