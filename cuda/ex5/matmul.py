import numpy as np
from numba import jit
from numba import cuda, float32
from abc import ABC, abstractmethod

# ABSTRACT METHOD DOT PRODUCT
class DotProduct(ABC):
    def __init__(self, A, B, C):
        assert A.shape[1] == B.shape[0], "A and B shapes misaligned!"
        assert A.shape[0] == C.shape[0], "A and C shapes misaligned!"
        assert B.shape[1] == C.shape[1], "B and C shapes misaligned!"
        self._A = A
        self._B = B
        self._C = C
        self._dim_m = A.shape[0]
        self._dim_n = A.shape[1]
        self._dim_k = B.shape[1]
    
    @abstractmethod
    def _dot(self): # abstract, protected method (single underscore prefix)
        pass

    def run(self):
        self._dot()
    
    def get_result(self):
        return self._C

class Basic(DotProduct):
    def _dot(self):
        A, B, C = self._A, self._B, self._C
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                sum = 0.0
                for k in range(A.shape[1]):
                    sum += A[i, k] * B[k, j]
                C[i, j] = sum

class Numpy(DotProduct):
    def _dot(self):
        A, B, C = self._A, self._B, self._C
        np.dot(A, B, out=C)
        # self.C[:] = np.dot(A, B) # Modify C in-place

class JitBasic(DotProduct):
    @staticmethod
    @jit(nopython=True)
    def __dot_kernel(A, B, C): # private method (double underscore prefix)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                sum = 0.0
                for k in range(A.shape[1]):
                    sum += A[i, k] * B[k, j]
                C[i, j] = sum
    
    def _dot(self):
        self.__dot_kernel(self._A, self._B, self._C)

class JitNumpy(DotProduct):
    @staticmethod
    @jit(nopython=True)
    def __dot_kernel(A, B, C): # private method (double underscore prefix)
        # np.dot(A, B, out=C) # Modify C in-place
        return A.dot(B)

    def _dot(self):
        # self.__dot_kernel(self._A, self._B, self._C)
        A, B, C = self._A, self._B, self._C
        np.dot(A, B, out=C)


# ABSTRACT METHOD CUDAJIT
class CudaJit(DotProduct):
    def __init__(self, A, B, C):
        super().__init__(A, B, C) # call base class constructor (in DotProduct)
        self.TPB = (16, 16) # threads per block
        self.bpgx = (self._dim_m + self.TPB[0] - 1) // self.TPB[0] # blocks per grid x
        self.bpgy = (self._dim_n + self.TPB[1] - 1) // self.TPB[1] # blocks per grid y
        self.BPG = (self.bpgx, self.bpgy)
    
    def run(self): # Override run method from parent class
        dA = cuda.to_device(self._A)
        dB = cuda.to_device(self._B)
        dC = cuda.to_device(self._C)
        self._dot[self.BPG, self.TPB](dA, dB, dC)
        cuda.synchronize()
        dC.copy_to_host(self._C)


class CudaGlobalMemory(CudaJit):
    @staticmethod
    @cuda.jit
    def _dot(A, B, C):
        x, y = cuda.grid(2)
        if x < C.shape[0] and y < C.shape[1]:
            sum = 0.0
            for k in range(A.shape[1]):
                sum += A[x, k] * B[k, y]
            C[x, y] = sum


class CudaSharedMemory(CudaJit):
    @staticmethod
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

