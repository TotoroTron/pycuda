import numpy as np
import matmul as mat
import time
import pandas as pd

class Testbench():
    def __init__(self, dims, methods, validation_method):
        self._methods = [ validation_method ] # Use mat.Numpy as validation run
        self._methods.extend(methods)
        self._dims = [ (1, 1, 1) ]  # (1, 1, 1) dummy test to force jit compile
        self._dims.extend(dims)
        self._report = []
        self._dataframe = pd.DataFrame()

    def _verify(self, result, expected):
        return int(not(np.allclose(result, expected, rtol=1e-05, atol=1e-08)))
    
    def get_report(self):
        return self._report.copy()

    def get_dataframe(self):
        return self._dataframe.copy()

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
            A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
            B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
            C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
            expectation = None
            
            for idx, method in enumerate(self._methods):
                instance = method(A, B, C.copy())
                start_time = time.time()
                instance.run()
                elapsed_time = time.time() - start_time
                result = instance.get_result()

                if idx == 0: # First method as validation run
                    expectation = instance.get_result() # redundant verify but not big deal

                # [ METHOD_NAME, LIST_DIMS, LIST_PASSFAILS, LIST_TIMES ]
                self._report[idx][1].append( dim )
                self._report[idx][2].append( self._verify(result, expectation) )
                self._report[idx][3].append( elapsed_time )
        
        # Remove (1, 1, 1) dummy tests
        for entry in self._report:
            entry[1].pop(0)
            entry[2].pop(0)
            entry[3].pop(0)
        
        # Construct pandas dataframes out of report
        for entry in self._report:
            # entry = [ METHOD_NAME, LIST_DIMS, LIST_PASSFAILS, LIST_TIMES ]
            df = pd.DataFrame({
                # 'dims' : entry[1],
                'dim_M' : [ dim[0] for dim in entry[1] ],
                'dim_N' : [ dim[1] for dim in entry[1] ],
                'dim_K' : [ dim[2] for dim in entry[1] ],
                f'{entry[0]}_failed (1)' : entry[2],
                f'{entry[0]}_time (s)' : entry[3]
            })
            # self._dataframe = self._dataframe.append(df)

            if self._dataframe.empty:
                self._dataframe = df
            else:
                self._dataframe = pd.merge(self._dataframe, df, on=['dim_M', 'dim_N', 'dim_K'], how='outer')

