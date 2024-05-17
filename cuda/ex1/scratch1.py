# https://numba.pydata.org/numba-doc/latest/cuda/kernels.html

from numba import jit
from numba import cuda
import numpy as np
import time

@cuda.jit
def increment_v1(array):
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
	

@cuda.jit
def increment_v2(array):
	pos = cuda.grid(1)
	if pos < array.size:
		array[pos] += 1

def main():
	arr = np.ones((10,), dtype=int)
	print(arr)

	a = cuda.to_device(arr)
	c = cuda.device_array_like(a)
	f.forall(len(a))(a, c)
	c.copy_to_host()
	
	print(c)
	
	"""
	print(a)
	increment_v2(arr)
	print(arr)
	"""
	
if __name__ == '__main__':
	main()
