from numba import jit
from numba import cuda
import numpy as np
import time

import matmul as m
import utils as u


def main():
    # INITIALIZE SQUARE MATRICES
    N = 8192
    matA = np.random.random(size=(N,N)).astype(np.float32)
    matB = np.random.random(size=(N,N)).astype(np.float32)
    matC = np.zeros(shape=(N, N), dtype=np.float32)
    outputs = [] # List of outputs
    
    print("Size of matrices A, B, C: ", matA.shape, matB.shape, matC.shape)
    print("Size of matrix A in MB: ", matA.nbytes / 1024 / 1024)

    # PREPARE INPUTS TO CUDA KERNEL
    start = time.time()
    d_matA = cuda.to_device(matA)
    elapsed_time = time.time() - start
    print(f"matA to_device Elapsed time: {elapsed_time:.6f} seconds.")

    start = time.time()
    d_matB = cuda.to_device(matB)
    elapsed_time = time.time() - start
    print(f"matB to_device Elapsed time: {elapsed_time:.6f} seconds.")

    start = time.time()
    d_matC = cuda.to_device(matC)
    elapsed_time = time.time() - start
    print(f"matC to_device Elapsed time: {elapsed_time:.6f} seconds.")

    threads_per_block = (16, 16) # 256 TPB (2-dimensional grid)
    blocks_per_grid_x = int(np.ceil(N / threads_per_block[0])) # BPGX
    blocks_per_grid_y = int(np.ceil(N / threads_per_block[1])) # BPGY
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # BPG (2-dimensional grid)

    # TEST CUDAJIT MATMUL
    for idx in range(2):

        start = time.time()
        m.matmul_cudajit_globalmem[blocks_per_grid, threads_per_block](d_matA, d_matB, d_matC)
        end = time.time() - start
        print(f"CUDAJIT Global Memory Kernel Elapsed time run {idx}: {end:.6f} seconds.")

        start = time.time()
        cuda.synchronize()
        end = time.time() - start
        print(f"CUDAJIT Global Memory Synchronize Elapsed time run {idx}: {end:.6f} seconds.")

        start = time.time()
        h_matC = d_matC.copy_to_host()
        end = time.time() - start
        print(f"CUDAJIT Global Memory to_host Elapsed time run {idx}: {end:.6f} seconds.")

        start = time.time()
        outputs.append(h_matC)
        end = time.time() - start
        print(f"Append Elapsed time run {idx}: {end:.6f} seconds.")


    print("Done.")

    
if __name__ == '__main__':
    main()
