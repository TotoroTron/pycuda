# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model 

from numba import jit
from numba import cuda
import numpy as np
import time

import matmul as m


def main():
    # (dim_m, dim_n, dim_k)
    # A cols = B rows = dim_n
    dimensions = [ (256, 256, 256) , (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048) ]

    for dim in dimensions:
        
        # VALIDATION JIT

        # CUDAJIT GLOBALMEM 
        # VERIFY CUDAJIT GLOBALMEM

        # CUDAJIT SHAREDMEM
        # VERIFY CUDAJIT SHAREDMEM

        pass
    
if __name__ == '__main__':
    main()
