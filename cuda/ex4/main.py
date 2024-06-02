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


def plot():
    pass

def main():

    # Dims: (M, N, K)
    # Dim A: (M, K)
    # Dim B: (K, N)
    # Dim C: (M, N)
    
    methods = [ mat.Numpy, mat.JitNumpy, mat.CudaGlobalMemory, mat.CudaSharedMemory ]

    squares = [ (512, 512, 512) ]
    tb_squares = tb.Testbench(methods, squares)
    tb_squares.test_all()
    results = tb_squares.get_results()

    print(f"Square Results:\n", results)

    # Python garbage collection implicitly calls __del__ on instances that are no longer in use


def print_conda_env():
    try:
        stream = os.popen('conda env export') # pipe open
        output = stream.read()
        print(output)
    except Exception as e:
        print(f"Error occurred while printing conda env: {e}")

if __name__ == '__main__':

    # Check if an output file argument is provided
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')
    
    # print_conda_env()
    # Just print from the bash script

    main()
