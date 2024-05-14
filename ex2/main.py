# https://www.kaggle.com/code/harshwalia/1-introduction-to-cuda-python-with-numba

from numba import jit
import math
import timeit


# This is the function decorator syntax and is equivalent to `hypot = jit(hypot)`.
# The Numba compiler is just a function you can call whenever you want!
@jit
def hypotenuse(x, y):
    x = abs(x)
    y = abs(y)
    t = min(x, y)
    x = max(x, y)
    # t = min(x,y) / max(x,y)
    t = t / x 

    return x * math.sqrt(1+t*t)

def main():

    x = 3.0
    y = 4.0

    printer = lambda x, y, r: f"Hypotenuse of {x} and {y} is {r}"
    # var names in lambda expr won't conflict with outer scope

    r = hypotenuse(x, y) # jit-compiled implementation
    print(printer(x, y, r))

    r = hypotenuse.py_func(x, y) # original Python implementation without jit
    print(printer(x, y, r))

if __name__ == '__main__':
    main()


