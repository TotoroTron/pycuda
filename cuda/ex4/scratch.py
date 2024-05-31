
class MatNumpy:
    def __init__(self, A, B, C):
        self._A = A
        self._B = B
        self._C = C

    def run(self):
        A, B, C = self._A, self._B, self._C
        np.dot(A, B, out=C)

class Testbench:
    def __init__(self):
        self._expectation = []

    def _set_expectation(self, A, B, C):
        self._expectation = MatNumpy(A, B, C).run()

    def test_all(self):
        dim = (2, 3, 4)
        A = np.random.random(size=(dim[0], dim[2])).astype(np.float32)
        B = np.random.random(size=(dim[2], dim[1])).astype(np.float32)
        C = np.zeros(shape=(dim[0], dim[1]), dtype=np.float32)
        print("Initial C:\n ", C)
        self._set_expectation(A, B, C.copy())
        print("C inside test_all after _set_expectation:\n", C)
        self._set_expectation(A, B, C)
        print("C inside test_all after _set_expectation:\n", C)

import numpy as np
tb = Testbench()
tb.test_all()

"""
All variables in python are implicitly passed by reference. BUT:

If a mutable var is passed into a function:
    Any modification to that var within the function modifies the original var
    outside the function.

If an immutable var is passed into a function:
    Any modification to that var within the function will allocate more memory to
    create a new var and modify that var instead.

IMMUTABLE TYPES:  integers, floats, strings, tuples
MUTABLE TYPES:    lists, dictionaries, sets, most objects

LISTS:    mylist  = [1, 2, 3]   Ordered,    Mutable,    Allows duplicate elements.
SETS:     myset   = {1, 2, 3}   Unordered,  Mutable,    Unique elements only.
TUPLES:   mytuple = (1, 2, 3)   Ordered,    Immutable,  Allows duplicate elements.

Mutable: You can change elements after creation.
Ordered: You can access elements by numbered index.
"""