# https://developer.nvidia.com/blog/numba-python-cuda-acceleration/

import numpy as np
from numba import cuda
from numba import vectorize

import time
import os


def add_cpu(a, b):
    return a + b


@vectorize(['float32(float32, float32)'], target='cuda')
def add_gpu(a, b):
    """
    - @vectorize : numba decorator
    - float32(float32, float32) : signature of the function : two float32 inputs and a float32 output.
    """
    return a + b


def main():
    """
    Simple vector addition.
    """
    
    # Initialize
    N = 1_000_000
    A = np.ones(N, dtype=np.float32)
    B = np.ones(A.shape, dtype=A.dtype)
    C = np.empty_like(A, dtype=A.dtype)

    
    for _ in range(10):
        # Add arrays on CPU
        start = time.time()
        C = add_cpu(A, B)
        elapsed_time = time.time() - start
        print(f"add_cpu: {elapsed_time:.3f} seconds.")

        # Add arrays on GPU
        start = time.time()
        C = add_gpu(A, B)
        elapsed_time = time.time() - start
        print(f"add_gpu: {elapsed_time:.3f} seconds.\n")


if __name__ == '__main__':
    main()