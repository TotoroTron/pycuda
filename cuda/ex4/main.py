# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model 

from numba import jit
from numba import cuda

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import time

import matmul as mat
import testbench as tb

def print_gpu_info():
    device = cuda.get_current_device()
    
    # BASIC INFO
    print("Device name:", device.name)
    print("Compute capability:", device.compute_capability)

    # CURRENT MEMORY INFORMATION
    mem_info = cuda.current_context().get_memory_info()
    total_memory = mem_info[1]
    free_memory = mem_info[0]
    print("Total global memory (GB):", total_memory / (1024 ** 3))
    print("Free global memory (GB):", free_memory / (1024 ** 3))
    print("Used global memory (GB):", (total_memory - free_memory) / (1024 ** 3))

    # GRID AND BLOCK DIMENSIONS
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    max_block_dim_x = device.MAX_BLOCK_DIM_X
    max_block_dim_y = device.MAX_BLOCK_DIM_Y
    max_block_dim_z = device.MAX_BLOCK_DIM_Z
    max_grid_dim_x = device.MAX_GRID_DIM_X
    max_grid_dim_y = device.MAX_GRID_DIM_Y
    max_grid_dim_z = device.MAX_GRID_DIM_Z

    print("Max threads per block:", max_threads_per_block)
    print("Max block dimensions: x={}, y={}, z={}".format(max_block_dim_x, max_block_dim_y, max_block_dim_z))
    print("Max grid dimensions: x={}, y={}, z={}".format(max_grid_dim_x, max_grid_dim_y, max_grid_dim_z))

    # SHARED MEMORY INFORMATION
    shared_memory_per_block = device.MAX_SHARED_MEMORY_PER_BLOCK
    print("Max shared memory per block (KB):", shared_memory_per_block / 1024)


def plot(results):
    methods = [result[0] for result in results]
    dimensions = [result[1] for result in results]
    times = [result[3] for result in results]

    plt.figure(figsize=(10, 6))

    for i, method in enumerate(methods):
        dims = [dim[0] for dim in dimensions[i]]  # First element of tuple as x-axis
        plt.plot(dims, times[i], label=method)

    plt.title('Performance Comparison of Methods')
    plt.xlabel('Dimension M')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.grid(True)
    plt.savefig('plot.png')

# Dims: (M, N, K)
# Dim A: (M, K)
# Dim B: (K, N)
# Dim C: (M, N)

def main():

    print_gpu_info()

    methods = [ mat.Numpy, mat.CudaGlobalMemory, mat.CudaSharedMemory ]
    squares = []
    for i in range(1, 64+1):
        dim = 64 * i # 64 to 4096
        squares.append((dim, dim, dim))

    tb_squares = tb.Testbench(methods, squares)
    tb_squares.test_all()
    results = tb_squares.get_results()

    plot(results)
    for idx, dim in enumerate(squares):
        for result in results:
            if result[1][idx] == dim:
                print(f"Method: {result[0]:<{20}}, dims: {str(result[1][idx]):<{20}}, pass(1)/fail(0): {result[2][idx]},\t time: {result[3][idx]}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

