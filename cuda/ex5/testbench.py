import numpy as np
import matmul as mat
import time

class Testbench():
    def __init__(self, methods, dims):
        self._methods = methods
        self._dims = [ (1, 1, 1) ]  # (1, 1, 1) dummy test to force jit compile
        self._dims.extend(dims)

        self._inputs = []
        self._expectations = []
        self._report = []

    def _set_expectations(self):
        for dim in self._dims:
            A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
            B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
            C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
            self._inputs.append((A, B, C))
            expectation = mat.Numpy(A, B, C.copy())
            expectation.run()
            self._expectations.append(expectation.get_result())

    def _verify(self, result, expected):
        # print("Result:\n", result)
        # print("Expected:\n", expected)
        return int(np.allclose(result, expected, rtol=1e-05, atol=1e-08))
    
    def get_results(self):
        return self._report.copy()

    def test_all(self):
        # print("Methods: ", self._methods)
        # print("Dims: ", self._dims)
        self._set_expectations()

        for method in self._methods:
            method_passfails = []
            method_times = []

            for idx, dim in enumerate(self._dims):
                A, B, C = self._inputs[idx]
                instance = method(A, B, C.copy())  # Basic, Numpy, JitNumpy, etc.
                # print("Instance: ", instance)
                start_time = time.time()
                instance.run()
                elapsed_time = time.time() - start_time

                result = instance.get_result()
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


class Testbench_alt():
    def __init__(self, methods, dims):
        self._methods = methods
        self._dims = [ (1, 1, 1) ]  # (1, 1, 1) dummy test to force jit compile
        self._dims.extend(dims)

        self._inputs = []
        self._expectations = []
        self._report = []

    def _verify(self, result, expected):
        # print("Result:\n", result)
        # print("Expected:\n", expected)
        # print()
        return np.allclose(result, expected, rtol=1e-05, atol=1e-08)
    
    def get_results(self):
        return self._report.copy()

    def test_all(self):
        # print("Methods: ", self._methods)
        # print("Dims: ", self._dims)
        
        for method in self._methods:
            self._report.append( [] ) # list of empty lists
            
        
        for dim in self._dims:
            A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
            B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
            C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
            expectation = mat.Numpy(A, B, C).run()
                             
            for idx, method in enumerate(self._methods):
                instance = method(A, B, C)
                start_time = time.time()
                instance.run()
                elapsed_time = start_time - time.time()
