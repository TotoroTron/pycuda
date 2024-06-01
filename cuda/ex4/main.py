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

    squares = [ (3, 3, 3) , (4, 4, 4), (5, 5, 5)]
    methods = [mat.Numpy, mat.JitNumpy]

    tb_squares = tb.Testbench(methods, squares)
    tb_squares.test_all()
    results = tb_squares.get_results()

    print(f"Results:\n", results)

    
if __name__ == '__main__':
    main()
