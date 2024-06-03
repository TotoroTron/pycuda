import matmul as mat
import numpy as np

def main():
    dim = (4, 4, 4)
    A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
    B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
    C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)

    shared_mem = mat.CudaSharedMemory(A, B, C)
    shared_mem.run()
    print(shared_mem._C.copy())

    global_mem = mat.CudaGlobalMemory(A, B, C)
    global_mem.run()
    print(global_mem._C.copy())

    pass

if __name__ == '__main__':
    main()
