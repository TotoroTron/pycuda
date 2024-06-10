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
        self._dim_n = B.shape[1]
        self._dim_k = B.shape[0] # = A.shape[1]
    
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
        self.BPG = (self.bpgx, self.bpgy) # ROUNDUP(dim/bpgx), ROUNDUP(dim/bpgy)
        # print("BPG: ", self.BPG, " TPB: ", self.TPB)
    
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

        if x < C.shape[0] and y < C.shape[1]:
            sum = 0.0
            for k in range(A.shape[1]):
                sum += A[x, k] * B[k, y]
            C[x, y] = sum
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

        bx = cuda.blockIdx.x # block idx x within grid
        by = cuda.blockIdx.y # block idx y within grid
        
        bw = cuda.blockDim.x # block dim x
        bh = cuda.blockDim.y # block dim y

        bpgx = cuda.gridDim.x # blocks per grid x
        bpgy = cuda.gridDim.y # blocks per grid y

        x = bx * bw + tx # global thread idx x
        y = by * bh + ty # global thread idx y

        # LOAD TILES INTO SHARED MEMORY
        for i in range(bpgy):
            for j in range(bpgx):
                sA[ty, tx] = 0.0
                sB[ty, tx] = 0.0

                if (ty + i * TILE_DIM) < A.shape[0] and (tx + j * TILE_DIM) < A.shape[1]:
                    sA[ty, tx] = A[ty + i * TILE_DIM, tx + j * TILE_DIM]

                if (ty + i * TILE_DIM) < B.shape[0] and (tx + j * TILE_DIM) < B.shape[1]:
                    sB[ty, tx] = B[ty + i * TILE_DIM, tx + j * TILE_DIM]

                cuda.syncthreads()

                sum = float32(0.0)
                for k in range(TILE_DIM):
                    sum += sA[ty, k] * sB[k, tx]

                cuda.syncthreads()

                # if x < C.shape[0] and y < C.shape[1]:
                C[y, x] += sum




"""
class __CudaSharedMemorySquare(CudaJit):
    @staticmethod
    @cuda.jit
    def _dot(A, B, C):
        TILE_DIM = 16
        sA = cuda.shared.array(shape=(TILE_DIM, TILE_DIM), dtype=float32)
        sB = cuda.shared.array(shape=(TILE_DIM, TILE_DIM), dtype=float32)
        
        x, y = cuda.grid(2)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bpg = cuda.gridDim.x
        
        if x >= C.shape[0] and y >= C.shape[1]:
            # Quit if (x, y) is outside of valid C boundary
            return
        
        C[x, y] = 0.0
        tmp = 0.0
        for i in range(bpg):
            # Preload data into shared memory
            sA[tx, ty] = A[x, ty + i * TILE_DIM]
            sB[tx, ty] = B[tx + i * TILE_DIM, y]

            # Wait until all threads finish preloading
            cuda.syncthreads()

            # Compute partial product on the shared memory
            for j in range(TILE_DIM):
                tmp += sA[tx, j] * sB[j, ty]

            # Wait until all threads finish computing
            cuda.syncthreads()
        
        C[x, y] = tmp





class __CudaSharedMemoryGeneral(CudaJit):
    @staticmethod
    @cuda.jit
    def _dot(A, B, C):
        
        Ref:
        # https://stackoverflow.com/questions/64197780/how-to-generalize-fast-matrix-multiplication-on-gpu-using-numba/64198479#64198479
        # https://github.com/numba/numba/blob/556545c5b2b162574c600490a855ba8856255154/numba/cuda/tests/doc_examples/test_matmul.py 
        # https://stackoverflow.com/questions/18815489/cuda-tiled-matrix-matrix-multiplication-with-shared-memory-and-matrix-size-whic 
        
        TILE_DIM = 16
        sA = cuda.shared.array(shape=(TILE_DIM, TILE_DIM), dtype=float32)
        sB = cuda.shared.array(shape=(TILE_DIM, TILE_DIM), dtype=float32)

        # (16, 16)
        x, y = cuda.grid(2)

        tx = cuda.threadIdx.x # thread index in x-dimension
        ty = cuda.threadIdx.y # thread index in y-dimension
        
        bx = cuda.blockIdx.x # block index in x-dimension
        by = cuda.blockIdx.y # block index in y-dimension
        
        bpg = cuda.gridDim.x 

        sum = float32(0.0)
        for i in range(bpg):
            sA[ty, tx] = 0.0
            sB[ty, tx] = 0.0

            if y < A.shape[0] and (tx + i * TILE_DIM) < A.shape[1]:
                sA[ty, tx] = A[y, tx + i * TILE_DIM]

            if x < B.shape[1] and (ty + i * TILE_DIM) < B.shape[0]:
                sB[ty, tx] = B[ty + i * TILE_DIM, x]

            cuda.syncthreads()

            for j in range(TILE_DIM):
                sum += sA[ty, j] * sB[j, tx]

            cuda.syncthreads()
        
        if y < C.shape[0] and x < C.shape[1]:
            C[y, x] = sum

"""