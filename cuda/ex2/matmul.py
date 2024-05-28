# https://numba.pydata.org/numba-doc/latest/cuda/kernels.html
# https://numba.pydata.org/numba-doc/latest/cuda/examples.html 

from numba import jit
from numba import cuda, float32
import math
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

@cuda.jit
def matmul_cudajit_sharedmem(A, B, C):
    """
    Controls threads per block and shared memory usage.
    The computation will be done on blocks of TPBxTPB elements.
    """
    TPB = 16
    # 16x16 = 256 threads
    # 256 x float32 = 8192 bits = 1024 bytes = 1KB

    # Define an array in the shared memory.
    # The size and type of the arrays must be known at compile time.  
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return
    
    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.0
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Compute partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()
    
    C[x, y] = tmp


















