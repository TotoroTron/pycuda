# https://numba.pydata.org/numba-doc/latest/cuda/kernels.html

from numba import jit
import numpy as np
import time

@cuda.jit
def increment_by_one(array):
	# Thread id in a 1D block
	tx = cuda.threadIdx.x
	
	# Block id in a 1D grid
	ty = cuda.blockIdx.x
	
	# Block width, i.e. number of threads per block
	bw = cuda.blockDim.x

	# Compute flattened index inside the array
	pos = tx + ty * bw
	if pos < array.size: # Check array boundaries
		array[pos] += 1

