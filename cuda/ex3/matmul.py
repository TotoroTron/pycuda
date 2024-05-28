import numpy as np
from numba import jit
from numba import cuda, float32
from abc import ABC, abstractmethod

# https://docs.python.org/3/library/abc.html 

# ABSTRACT METHOD DOT PRODUCT
class DotProduct:
    def __init__(self, dim_m, dim_n, dim_k):
        self.dim_m = dim_m
        self.dim_n = dim_n
        self.dim_k = dim_k
        self.A = np.random.random(size=(dim_m, dim_k)).astype(np.float32)
        self.B = np.random.random(size=(dim_k, dim_n)).astype(np.float32)
        self.C = np.zeros(shape=(dim_m, dim_n), dtype=np.float32)
    
    @abstractmethod
    def run(self):
        pass

    def verify(self, expected):
        assert np.allclose(self.C, expected), "Results do not match!"
        print("Results match.")

class Jit(DotProduct):
    @jit(nopython=True)
    def dot(A, B, C):
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                sum = 0.0
                for k in range(A.shape[1]):
                    sum += A[i, k] * B[k, j]
                C[i, j] = sum
    
    def run(self): 
        self.dot(self.A, self.B, self.C)

# ABSTRACT METHOD CUDAJIT
class CudaJit(DotProduct):
    def __init__(self, dim_m, dim_n, dim_k):
        super().__init__(dim_m, dim_n, dim_k) # call base class constructor (in DotProduct)
        self.TPB = (16, 16) # threads per block
        self.bpgx = (self.dim_m + self.tpb[0] - 1) // self.tpb[0] # blocks per grid x
        self.bpgy = (self.dim_n + self.tpb[1] - 1) // self.tpb[1] # blocks per grid y
        self.BPG = (self.bpgx, self.bpgy)
    
    def configure(self):
        dA = cuda.to_device(self.A)
        dB = cuda.to_device(self.B)
        dC = cuda.to_device(self.C)
        return dA, dB, dC
    
    @abstractmethod
    def dot(self, dA, dB, dC):
        pass

    def run(self):
        dA, dB, dC = self.configure()
        self.dot[self.BPG, self.TPB](dA, dB, dC)
        dC.copy_to_host(self.C)


class GlobalMemory(CudaJit):
    @cuda.jit
    def dot(A, B, C):
        x, y = cuda.grid(2)
        if x < C.shape[0] and y < C.shape[1]:
            sum = 0.0
            for k in range(A.shape[1]):
                sum += A[x, k] * B[k, y]
            C[x, y] = sum


class SharedMemory(CudaJit):
    def __init__(self, dim_m, dim_n, dim_k):
        super().__init__(dim_m, dim_n, dim_k) # call base class constructor (in CudaJit)

    @cuda.jit
    def dot(A, B, C):
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

