from numba import jit
from numba import cuda
import numpy as np
import time

import matmul as m
import utils as u


def main():
    # INITIALIZE SQUARE MATRICES
    N = 1024
    matA = np.random.random(size=(N,N)).astype(np.float32)
    matB = np.random.random(size=(N,N)).astype(np.float32)
    matC = np.zeros(shape=(N, N), dtype=np.float32)
    outputs = [] # List of outputs
    
    print("Size of matrices A, B, C: ", matA.shape, matB.shape, matC.shape)

    # TEST BASIC MATMUL
    start = time.time()
    m.matmul(matA, matB, matC) # All np arrays are passed by reference!
    elapsed_time = time.time() - start
    print(f"Basic Elapsed time: {elapsed_time:.3f} seconds.")
    outputs.append(matC)
    

    # TEST JIT MATMUL
    for idx in range(2): # Test twice to because 1st run includes jit compile time
        start = time.time()
        m.matmul_jit(matA, matB, matC)
        elapsed_time = time.time() - start
        print(f"JIT Elapsed time run {idx}: {elapsed_time:.6f} seconds.")
        outputs.append(matC)

    # PREPARE INPUTS TO CUDA KERNEL
    d_matA = cuda.to_device(matA)
    d_matB = cuda.to_device(matB)
    d_matC = cuda.to_device(matC)

    threads_per_block = (16, 16) # 256 TPB (2-dimensional grid)
    blocks_per_grid_x = int(np.ceil(N / threads_per_block[0])) # BPGX
    blocks_per_grid_y = int(np.ceil(N / threads_per_block[1])) # BPGY
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # BPG (2-dimensional grid)

    # TEST CUDAJIT MATMUL
    for idx in range(2):
        start = time.time()
        m.matmul_cudajit_globalmem[blocks_per_grid, threads_per_block](d_matA, d_matB, d_matC)
        elapsed_time = time.time() - start
        print(f"CUDAJIT Global Memory Elapsed time run {idx}: {elapsed_time:.6f} seconds.")
        h_matC = d_matC.copy_to_host()
        outputs.append(h_matC)

    # VERIFY OUTPUTS
    for idx in range(len(outputs)):
        print(f"verify: {u.verify(outputs[0], outputs[idx])}")


    print("Done.")

    
if __name__ == '__main__':
    main()
