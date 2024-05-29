# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model 

from numba import jit
from numba import cuda

import numpy as np
import time
import matmul as mat
import matplotlib.pyplot as plt


def main():
    # (dim_m, dim_k, dim_n)
    # A cols = B rows = dim_k
    # dimensions = [ (256, 256, 256) , (512, 512, 512), (1024, 1024, 1024) ]
    # dimensions = [ (128, 128, 128) , (256, 256, 256), (512, 512, 512)]
    # dimensions = [ (16, 16, 16) , (32, 32, 32), (64, 64, 64)]
    # dimensions = [ (4, 4, 4) , (8, 8, 8), (16, 16, 16)]
    # dimensions = [ (16, 16, 16),
    #                (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256),
    #                (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
    #                (2560, 2560, 2560), (3072, 3072, 3072), (3584, 3548, 3548), (4096, 4096, 4096)]

    dimensions = [ (1, 1, 1)]
    for i in range(1, 17):
        size = 64 * i
        dimensions.append( (size, size, size))

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

    # Remove all (1, 1, 1) tests
    results.pop(0)
    results.pop(0)
    dimensions.pop(0)
    list_of_dims = [ dim[0] for dim in dimensions ]
    list_of_numpy_times = [ result[2] for result in results if result[1] == 'Numpy' ]
    list_of_jit_times = [ result[2] for result in results if result[1] == 'JitNumpy' ]

    # PLOT
    plt.plot(list_of_dims, list_of_numpy_times, 'b', label='Numpy')
    plt.plot(list_of_dims, list_of_jit_times, 'r', label='JitNumpy')
    plt.yscale("log")
    plt.grid(which='both', axis='both')
    plt.xlabel("dim_m")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.savefig('plot.png')

    print("\nRESULTS:\n")
    prev_dim = None
    for result in results:
        dim_str = ",".join(map(str, result[0]))
        if prev_dim is not None and prev_dim != result[0]:
            print()
        print("Dimensions: ({:<16}) \tMethod: {:<16} \tElapsed time: {:>12.6f} seconds.".format(dim_str, result[1], result[2]))
        prev_dim = result[0]

    print("Done.")
    
if __name__ == '__main__':
    main()
