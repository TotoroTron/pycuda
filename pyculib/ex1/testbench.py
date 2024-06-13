import numpy as np
import matmul as mat
import time

class Testbench():
    def __init__(self, methods, dims):
        self._methods = methods
        self._dims = [ (1, 1, 1) ]  # (1, 1, 1) dummy test to force jit compile
        self._dims.extend(dims)
        self._report = []

    def _verify(self, result, expected):
        return int(not(np.allclose(result, expected, rtol=1e-05, atol=1e-08)))
    
    def get_report(self):
        return self._report.copy()

    def test_all(self):

        # report = [ entry, entry, ..., entry ]
        # entry = [ METHOD_NAME, LIST_DIMS, LIST_PASSFAILS, LIST_TIMES ]
        for idx, method in enumerate(self._methods):
            self._report.append( [] )
            self._report[idx].append( method.__name__ ) # METHOD_NAME
            self._report[idx].append( [] ) # LIST_DIMS
            self._report[idx].append( [] ) # LIST_PASSFAILS
            self._report[idx].append( [] ) # LIST_TIMES

        for dim in self._dims:

            # A = np.ones(shape=(dim[0], dim[2]), dtype=np.float32)
            # B = np.full(shape=(dim[2], dim[1]), fill_value=5, dtype=np.float32)

            A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
            B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
            C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
            validation = mat.Numpy(A, B, C.copy())
            validation.run()
            expectation = validation.get_result()
                             
            for idx, method in enumerate(self._methods):
                instance = method(A, B, C.copy())
                start_time = time.time()
                instance.run()
                elapsed_time = time.time() - start_time
                # print("Elapsed time: ", elapsed_time)

                result = instance.get_result()
                # [ METHOD_NAME, LIST_DIMS, LIST_PASSFAILS, LIST_TIMES ]
                self._report[idx][1].append( dim )
                self._report[idx][2].append( self._verify(result, expectation) )
                self._report[idx][3].append( elapsed_time )

                # probably easier to just use pandas at this point
                # verif = self._verify(result, expectation)
                # if not verif:
                #     print("FAILED! Dimensions: ", dim, " Method: ", method.__name__)
                #     print("Result:\n", result)
                #     print("Expected:\n", expectation)
                #     print("Difference:\n", result - expectation)
                #     print()
        
        # Remove (1, 1, 1) dummy tests
        for entry in self._report:
            entry[1].pop(0)
            entry[2].pop(0)
            entry[3].pop(0)

