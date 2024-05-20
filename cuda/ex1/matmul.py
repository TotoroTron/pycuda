# https://numba.pydata.org/numba-doc/latest/cuda/kernels.html
# https://numba.pydata.org/numba-doc/latest/cuda/examples.html 

from numba import jit
from numba import cuda
import numpy as np
import time
# import csv

def verify(A, B):
    """
    Compare matrices A and B for equality.
    """
    return np.allclose(A, B)

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


def main():
    # INITIALIZE SQUARE MATRICES
    N = 256
    matA = np.random.random(size=(N,N)).astype(np.float32)
    matB = np.random.random(size=(N,N)).astype(np.float32)
    matC = np.zeros(shape=(N, N), dtype=np.float32)
    outputs = [] # List of outputs
    
    # TEST BASIC MATMUL
    start = time.time()
    matmul(matA, matB, matC) # All np arrays are passed by reference!
    elapsed_time = time.time() - start
    print(f"Basic Elapsed time: {elapsed_time:.3f} seconds.")
    outputs.append(matC)

    # TEST JIT MATMUL
    start = time.time()
    matmul_jit(matA, matB, matC)
    elapsed_time = time.time() - start
    print(f"JIT Elapsed time: {elapsed_time:.3f} seconds.")
    outputs.append(matC)

    # PREPARE INPUTS TO CUDA KERNEL
    cuda.to_device(matA)
    cuda.to_device(matB)
    d_matC = cuda.to_device(matC)

    threads_per_block = (16, 16) # 16x16=256 TPB (2-dimensional grid)
    blocks_per_grid_x = int(np.ceil(N / threads_per_block[0])) # 256/16=16 BPGX
    blocks_per_grid_y = int(np.ceil(N / threads_per_block[1])) # 256/16=16 BPGY
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # (16,16) BPG (2-dimensional grid)

    # TEST CUDAJIT MATMUL
    start = time.time()
    matmul_cudajit_globalmem[blocks_per_grid, threads_per_block](matA, matB, d_matC)
    elapsed_time = time.time() - start
    print(f"CUDAJIT Global Memory Elapsed time: {elapsed_time:.3f} seconds.")
    h_matC = d_matC.copy_to_host()
    outputs.append(h_matC)

    # VERIFY OUTPUTS
    print(f"verify: {verify(outputs[0], outputs[1])}")
    print(f"verify: {verify(outputs[0], outputs[2])}")

    # SAVE INPUTS TO CSV
    filename = f"logs/matA.csv"
    np.savetxt(filename, matA, delimiter=',', fmt='%f')
    filename = f"logs/matB.csv"
    np.savetxt(filename, matB, delimiter=',', fmt='%f')

    # SAVE OUTPUTS TO CSV
    for idx, mat in enumerate(outputs):
        filename = f"logs/matC_{idx}.csv"
        np.savetxt(filename, mat, delimiter=',', fmt='%f')
    
    print("Done.")

    
if __name__ == '__main__':
    main()
