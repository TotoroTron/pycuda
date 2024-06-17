import numpy as np
from numba import jit
from numba import cuda, float32
from abc import ABC, abstractmethod
import cupy as cp

# ABSTRACT CLASS DOT PRODUCT
class DotProduct(ABC):
    def __init__(self, A, B, C):
        assert A.shape[1] == B.shape[0], "A and B shapes misaligned!"
        assert A.shape[0] == C.shape[0], "A and C shapes misaligned!"
        assert B.shape[1] == C.shape[1], "B and C shapes misaligned!"
        self._A = A
        self._B = B
        self._C = C
    
    @abstractmethod
    def _dot(self):
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

class JitBasic(DotProduct):
    @staticmethod
    @jit(nopython=True)
    def __dot_kernel(A, B, C):
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
    def _dot(self):
        A, B, C = self._A, self._B, self._C
        np.dot(A, B, out=C)


# ABSTRACT CLASS CUDAJIT
class CudaJit(DotProduct):
    def __init__(self, A, B, C):
        super().__init__(A, B, C) # call base class constructor (in DotProduct)
        self.TPB = (16, 16) # threads per block (x, y)
        grid_y_max = max(A.shape[0], B.shape[0])
        grid_x_max = max(A.shape[1], B.shape[1])
        self.bpgy = (grid_y_max + self.TPB[1] - 1) // self.TPB[1] # blocks per grid y
        self.bpgx = (grid_x_max + self.TPB[0] - 1) // self.TPB[0] # blocks per grid x
        self.BPG = (self.bpgx, self.bpgy) # ROUNDUP(dim/tpbx), ROUNDUP(dim/tpby)
        # print("BPG: ", self.BPG, " TPB: ", self.TPB)
    
    def get_grid_dims(self):
        return (self.BPG[0], self.BPG[1])
    
    def get_block_dims(self):
        return (self.TPB[0], self.TPB[1])
    
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
        
        # https://stackoverflow.com/questions/18815489/cuda-tiled-matrix-matrix-multiplication-with-shared-memory-and-matrix-size-whic
        """
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        bw = cuda.blockDim.x # 16
        bh = cuda.blockDim.y # 16

        x = bx * bw + tx
        y = by * bh + ty
        """
        # SHORTHAND FOR ABOVE:
        x, y = cuda.grid(2)

        if y < C.shape[0] and x < C.shape[1]:
            sum = 0.0
            for k in range(A.shape[1]):
                sum += A[y, k] * B[k, x]
            C[y, x] = sum
        else:
            pass

class CudaSharedMemory(CudaJit):
    @staticmethod
    @cuda.jit
    def _dot(A, B, C): 
        TILE_DIM = 16

        sA = cuda.shared.array(shape=(TILE_DIM, TILE_DIM), dtype=float32)
        sB = cuda.shared.array(shape=(TILE_DIM, TILE_DIM), dtype=float32)

        tx = cuda.threadIdx.x # thread idx x within block
        ty = cuda.threadIdx.y # thread idx y within block
        bpgx = cuda.gridDim.x # blocks per grid x
        x, y = cuda.grid(2) # global thread idx x, y

        # LOAD TILES INTO SHARED MEMORY
        acc = float32(0.0)
        for i in range(bpgx):
            sA[ty, tx] = 0.0
            sB[ty, tx] = 0.0

            if y < A.shape[0] and (i*TILE_DIM + tx) < A.shape[1]:
                sA[ty, tx] = A[y, i*TILE_DIM + tx]

            if (i*TILE_DIM + ty) < B.shape[0] and x < B.shape[1]:
                sB[ty, tx] = B[i*TILE_DIM + ty, x]

            cuda.syncthreads()

            for j in range(TILE_DIM):
                acc += sA[ty, j] * sB[j, tx]

            cuda.syncthreads()
        
        if y < C.shape[0] and x < C.shape[1]:
            C[y, x] += acc


class CupyMatmul(DotProduct):
    def _dot(self):
        dA = cp.array(self._A)
        dB = cp.array(self._B)
        dC = cp.matmul(dA, dB)
        self._C[:] = cp.asnumpy(dC)


class CupyDot(DotProduct):
    def _dot(self):
        dA = cp.array(self._A)
        dB = cp.array(self._B)
        dC = cp.dot(dA, dB)
        self._C[:] = cp.asnumpy(dC)