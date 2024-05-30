import numpy as np
import matmul as mat
from abc import ABC, abstractmethod
import time

class Testbench():
    def __init__(self, methods, dims):
        self._methods = methods
        self._dims = [ (1, 1, 1) ] # (1, 1, 1) dummy test to force jit compile
        self._dims.extend(dims)
        self._expectation = []
        
        self._passfail = []
        self._times = []

    def _verify(self, result):
        return np.allclose(self.result, self._expectation, rtol=1e-5, atol=1e-8)
    
    def set_methods(self, m):
        self._methods.extend(m)

    def set_dims(self, d):
        self._dims.extend(d)

    def get_results(self):
        return self._dims, [ self._methods, self._passfail, self._times ] 

    def test_all(self):
        for dim in self._dims:
            A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
            B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
            C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
            self._expectation = mat.Numpy(A, B, C).run()

            for method in self._methods:
                instance = method(A, B, C) # Basic, Numpy, etc.

                start_time = time.time()
                result = instance.run()
                elapsed_time = start_time - time.time()

                self._passfail.append(self._verify(result))
                self._times.append(elapsed_time)