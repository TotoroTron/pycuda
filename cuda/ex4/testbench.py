import numpy as np
import pandas as pd
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
        return np.allclose(result, self._expectation, rtol=1e-5, atol=1e-8)
    
    def get_results(self):


        pass


    def test_all(self):
        # Methods on outer loop:
        # Easier indexing
        # But need to recreate A B C every loop (could become a problem for large dim?)
        for method in self._methods:

            method_passfail = []
            method_times = []
            
            for dim in self._methods:
                A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
                B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
                C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
                self._expectation = mat.Numpy(A, B, C).run()

                instance = method(A, B, C) # Basic, Numpy, JitNumpy, etc.
                start_time = time.time()
                result = instance.run()
                elapsed_time = start_time - time.time()

                method_passfail.append(self._verify(result))
                method_times.append(elapsed_time)
            
            self._passfail.append(method_passfail)
            self._times.append(method_times)



    def __test_all_alt(self):
        # Dims on outer loop: create A B C once for each dim, harder indexing
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