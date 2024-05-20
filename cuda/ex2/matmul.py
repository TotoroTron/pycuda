# https://numba.pydata.org/numba-doc/latest/cuda/kernels.html
# https://numba.pydata.org/numba-doc/latest/cuda/examples.html 

from numba import jit
from numba import cuda
import numpy as np
import time
# import csv

def matmul(A, B, C):
    """
    Most basic/naive/intuitive matrix multiplication.
    In C++ this works better if first transpose B to allow more
    sequential memory access. Not sure if this works in Python.
    """
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

@jit(nopython=True)
def matmul_jit(A, B, C):
    """
    Naive square matrix multiplication, but compiled with JIT.
    """
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

@cuda.jit
def matmul_cudajit_globalmem(A, B, C):
    """
    Naive square matrix multiplication, but compiled with CUDA JIT.
    Performs poorly bc of redundant data fetching from device (global) memory.

    i, j = cuda.grid(2) basically short-hand for:
        i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        i is the ROW index in matrix C that the current thread is responsible for.
        j is the COL index in matrix C that the current thread is responsible for.

    Each thread is responsible for calculating one element in C
    by iterating over the corresponding ROW of A and COL of B.
    """

    i, j = cuda.grid(2)

    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

