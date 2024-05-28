import numpy as np
from numba import jit
from numba import cuda, float32
from abc import ABC, abstractmethod
import time

# https://docs.python.org/3/library/abc.html 

# ABSTRACT METHOD DOT PRODUCT
class DotProduct(ABC):
    def __init__(self, A, B, C):
        assert A.shape[1] == B.shape[0], "A and B shapes misaligned!"
        assert A.shape[0] == C.shape[0], "A and C shapes misaligned!"
        assert B.shape[1] == C.shape[1], "B and C shapes misaligned!"
        self.A = A
        self.B = B
        self.C = C
        self.dim_m = A.shape[0]
        self.dim_n = A.shape[1]
        self.dim_k = B.shape[1]
        self.elapsed_time = 0.0
    
    @abstractmethod
    def _dot(self): # abstract, protected method (single underscore prefix)
        pass

    def run(self):
        start_time = time.time()
        self._dot()
        self.elapsed_time = time.time() - start_time

    def verify(self, expected):
        assert np.allclose(self.C, expected, rtol=1e-05, atol=1e-08), "Results do not match!"
        # print("Results match.")

class Basic(DotProduct):
    def _dot(self):
        A, B, C = self.A, self.B, self.C
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                sum = 0.0
                for k in range(A.shape[1]):
                    sum += A[i, k] * B[k, j]
                C[i, j] = sum

class Numpy(DotProduct):
    def _dot(self):
        A, B, C = self.A, self.B, self.C
        C = np.dot(A, B)

class Jit(DotProduct):
    @staticmethod
    @jit(nopython=True)
    def __dot_kernel(A, B, C): # private method (double underscore prefix)
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                sum = 0.0
                for k in range(A.shape[1]):
                    sum += A[i, k] * B[k, j]
                C[i, j] = sum
    
    def _dot(self):
        self.__dot_kernel(self.A, self.B, self.C)


# ABSTRACT METHOD CUDAJIT
class CudaJit(DotProduct):
    def __init__(self, A, B, C):
        super().__init__(A, B, C) # call base class constructor (in DotProduct)
        self.TPB = (16, 16) # threads per block
        self.bpgx = (self.dim_m + self.TPB[0] - 1) // self.TPB[0] # blocks per grid x
        self.bpgy = (self.dim_n + self.TPB[1] - 1) // self.TPB[1] # blocks per grid y
        self.BPG = (self.bpgx, self.bpgy)

    def __configure(self): # private method (double underscore prefix)
        dA = cuda.to_device(self.A)
        dB = cuda.to_device(self.B)
        dC = cuda.to_device(self.C)
        return dA, dB, dC
    
    def run(self): # Override run method from parent class
        start_time = time.time()
        dA, dB, dC = self.__configure()
        self._dot[self.BPG, self.TPB](dA, dB, dC)
        dC.copy_to_host(self.C)
        self.elapsed_time = time.time() - start_time


class CudaGlobalMemory(CudaJit):
    @cuda.jit
    def _dot(self, A, B, C):
        x, y = cuda.grid(2)
        if x < C.shape[0] and y < C.shape[1]:
            sum = 0.0
            for k in range(A.shape[1]):
                sum += A[x, k] * B[k, y]
            C[x, y] = sum


class CudaSharedMemory(CudaJit):
    @cuda.jit
    def _dot(A, B, C):
        """
        Controls threads per block and shared memory usage.
        The computation will be done on blocks of TPBxTPB elements.
        """
        TPB = 16
        # 16x16 = 256 threads
        # 256 x float32 = 8192 bits = 1024 bytes = 1KB

        # Define an array in the shared memory.
        # The size and type of the arrays must be known at compile time.  
        sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
        sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
        # Total 2KB of shared memory per block
        
        x, y = cuda.grid(2)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bpg = cuda.gridDim.x
        
        if x >= C.shape[0] and y >= C.shape[1]:
            # Quit if (x, y) is outside of valid C boundary
            return
        
        # Each thread computes one element in the result matrix.
        # The dot product is chunked into dot products of TPB-long vectors.
        tmp = 0.0
        for i in range(bpg):
            # Preload data into shared memory
            sA[tx, ty] = A[x, ty + i * TPB]
            sB[tx, ty] = B[tx + i * TPB, y]

            # Wait until all threads finish preloading
            cuda.syncthreads()

            # Compute partial product on the shared memory
            for j in range(TPB):
                tmp += sA[tx, j] * sB[j, ty]

            # Wait until all threads finish computing
            cuda.syncthreads()
        
        C[x, y] = tmp
