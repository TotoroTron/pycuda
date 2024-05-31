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

        self._expectations = []
        self._report = []

    def _set_expectations(self):
        for dim in self._dims:
            A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
            B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
            C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
            self._expectations.append(mat.Numpy(A, B, C).run())

    def _verify(self, result, expected):
        print("Result:\n", result)
        print("Expected:\n", expected)
        return np.allclose(result, expected, rtol=1e-05, atol=1e-08)
    
    def get_results(self):
        return self._report.copy()


    # this doesn't work.
    def test_all(self):
        print("Methods: ", self._methods)
        print("Dims: ", self._dims)

        self._set_expectations()
        # Methods on outer loop:
        # Easier indexing
        # Have to store expectation for every dim for entire test. Wasteful memory.
        # Verify isnt even using the same inputs as test.
        # Have to store the inputs for every dim too. Even more waste.
        for method in self._methods:

            method_passfails = []
            method_times = []

            for idx, dim in enumerate(self._dims):
                A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
                B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
                C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)

                instance = method(A, B, C) # Basic, Numpy, JitNumpy, etc.
                print("Instance: ", instance)
                start_time = time.time()
                result = instance.run()
                elapsed_time = start_time - time.time()

                method_passfails.append(self._verify(result, self._expectations[idx]))
                method_times.append(elapsed_time)
            
            # Remove (1, 1, 1) dummy test
            method_dims = self._dims.copy()
            method_dims.pop(0)
            method_passfails.pop(0)
            method_times.pop(0)

            method_str = method.__name__
            report_entry = [ method_str, method_dims, method_passfails, method_times ]
            self._report.append(report_entry)


    def test_all_alt(self):
        print("Methods: ", self._methods)
        print("Dims: ", self._dims)

        self._set_expectations()

        for dim in self._dims:
            A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
            B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
            C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)

            #_set_expectation(A, B, C)

            for idx, method in enumerate(self._methods):
                pass
