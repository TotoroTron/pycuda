# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model 

from numba import jit
from numba import cuda

import numpy as np
import time
import matmul as mat


def main():
    # (dim_m, dim_k, dim_n)
    # A cols = B rows = dim_k
    # dimensions = [ (256, 256, 256) , (512, 512, 512), (1024, 1024, 1024) ]
    # dimensions = [ (128, 128, 128) , (256, 256, 256), (512, 512, 512)]
    # dimensions = [ (16, 16, 16) , (32, 32, 32), (64, 64, 64)]
    dimensions = [ (4, 4, 4) , (8, 8, 8), (16, 16, 16)]
    # dimensions = [ (16, 16, 16), (2048, 2048, 2048) ]

    results = []


    for dim in dimensions:

        A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
        B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
        C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
        matrices = (A, B, C)

        # NUMPY DOT
        numpy_test = mat.Numpy(*matrices)
        numpy_test.run()
        expected = numpy_test.C.copy()
        results.append((dim, numpy_test.__class__.__name__, numpy_test.elapsed_time))

        # JIT NUMPY DOT
        jit_np_test = mat.JitNumpy(*matrices)
        jit_np_test.run()
        jit_np_test.verify(expected)
        results.append((dim, jit_np_test.__class__.__name__, jit_np_test.elapsed_time))

    prev_dim = None
    for result in results:
        dim_str = ",".join(map(str, result[0]))
        if prev_dim is not None and prev_dim != result[0]:
            print("\n")
        print("Dimensions: ({:<12}) \tMethod: {:<16} \tElapsed time: {:>12.6f} seconds.".format(dim_str, result[1], result[2]))
        prev_dim = result[0]

    print("Done.")
    
if __name__ == '__main__':
    main()
