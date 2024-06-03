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

    methods = [ mat.Numpy, mat.JitNumpy, mat.CudaGlobalMemory, mat.CudaSharedMemory ]
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

