# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model 

from numba import jit
from numba import cuda
import numpy as np
import time

import matmul as mat


def main():
    # (dim_m, dim_k, dim_n)
    # A cols = B rows = dim_k
    # dimensions = [ (256, 256, 256) , (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048) ]
    dimensions = [ (16, 16, 16) , (32, 32, 32), (64, 64, 64) ]

    results = []


    for dim in dimensions:

        A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
        B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
        C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
        matrices = (A, B, C)


        # BASIC DOT PRODUCT
        basic_test = mat.Basic(*matrices)
        basic_test.run()
        expected = basic_test.C.copy()
        results.append((basic_test.__class__.__name__, basic_test.elapsed_time))

        # NUMPY DOT PRODUCT
        numpy_test = mat.Numpy(*matrices)
        numpy_test.run()
        numpy_test.verify(expected)
        results.append((numpy_test.__class__.__name__, numpy_test.elapsed_time))


    
if __name__ == '__main__':
    main()
