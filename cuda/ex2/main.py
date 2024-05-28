# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model 

from numba import jit
from numba import cuda
import numpy as np
import time

import matmul as m
import utils as u


def main():
    # INITIALIZE SQUARE MATRICES
    N = 8192+2048
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

    print("\n")

    threads_per_block = (16, 16) # 256 TPB (2-dimensional grid)
    blocks_per_grid_x = int(np.ceil(N / threads_per_block[0])) # BPGX
    blocks_per_grid_y = int(np.ceil(N / threads_per_block[1])) # BPGY
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # BPG (2-dimensional grid)


    # TEST CUDAJIT GLOBAL MEMORY

    start = time.time()
    m.matmul_cudajit_globalmem[blocks_per_grid, threads_per_block](d_matA, d_matB, d_matC)
    cuda.synchronize() # Wait for all threads to finish
    end = time.time() - start
    print(f"CUDAJIT GlobalMem kernel: {end:.6f} seconds.")
    h_matC = d_matC.copy_to_host()
    outputs.append(h_matC)

    # TEST CUDAJIT SHARED MEMORY

    start = time.time()
    m.matmul_cudajit_sharedmem[blocks_per_grid, threads_per_block](d_matA, d_matB, d_matC)
    cuda.synchronize() # Wait for all threads to finish
    end = time.time() - start
    print(f"CUDAJIT SharedMem kernel: {end:.6f} seconds.")
    h_matC = d_matC.copy_to_host()
    outputs.append(h_matC)

    # VERIFY RESULTS
    for i in range(len(outputs)):
        print(f"verify: {u.verify(outputs[0], outputs[i])}")


    print("\n")

    
if __name__ == '__main__':
    main()
