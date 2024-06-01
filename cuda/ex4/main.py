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

    # Check if an output file argument is provided
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    # Dims: (dim_m, dim_n, dim_k)
    # Dim A: (dim_m, dim_k)
    # Dim B: (dim_k, dim_n)
    # Dim C: (dim_m, dim_n)
    
    methods = [ mat.Numpy, mat.JitNumpy ]

    squares = [ (3, 3, 3), (4, 4, 4) ]
    tb_squares = tb.Testbench(methods, squares)
    tb_squares.test_all()
    results = tb_squares.get_results()

    print(f"Square Results:\n", results)
    print("\n=========================================\n")

    nonsquares = [ (3, 3, 4), (4, 4, 3), (4, 3, 4), (3, 4, 4) ]
    tb_nonsquares = tb.Testbench(methods, nonsquares)
    tb_nonsquares.test_all()
    results = tb_nonsquares.get_results()

    print(f"Non-square Results:\n", results)
    print("\n=========================================\n")

    # Python garbage collection implicitly calls __del__ on instances that are no longer in use



    
if __name__ == '__main__':
    main()
