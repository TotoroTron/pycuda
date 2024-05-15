# https://docs.python.org/3/library/timeit.html 

import timeit
import math

def hypot_1(x, y):
    return math.hypot(x, y)


def hypot_2(x, y):
    return math.sqrt(x**2 + y**2)


def hypot_3(x, y):
    return (x**2 + y**2) ** 0.5


def hypot_4(x, y):
    x = abs(x)
    y = abs(y)
    t = min(x, y)
    x = max(x, y)
    t = t / x
    return x * math.sqrt(1+t*t)


# Dictionary of hypot functions
hypot_funcs = {
    # 'key': 'value'
    'hypot_1': hypot_1,
    'hypot_2': hypot_2,
    'hypot_3': hypot_3,
    'hypot_4': hypot_4
}


# Higher-order function for hypot functions
def hypotenuse(x, y, method='hypot_1'): # Defaults to the 'hypot_1' method\
    if method in hypot_funcs:
        return hypot_funcs[method](x, y) # Call the corresponding hypot function
    else:
        raise ValueError(f"Unknown method: {method}")


# Main function
def main():

    # Set up the timeit Timer
    t = timeit.Timer(lambda: hypot_1(3.0, 4.0))

    # Run the timer with a specified number of executions
    num_executions = 1000000
    execution_time = t.timeit(number=num_executions)
    print(f"No repeats, hypot_1: {execution_time:.6f} seconds.")
    

    x, y = 3.0, 4.0
    num_repeats = 5
    num_executions = 1000000

    for method in hypot_funcs.keys(): # for each method in dictionary
        # Measure the execution time
        times = timeit.repeat(
            stmt=f"hypotenuse({x}, {y}, method='{method}')", # Code to execute as a string
            setup="from __main__ import hypotenuse", # Code to set up the environment
            repeat=num_repeats,
            number=num_executions
        )
        fastest_time = min(times)
        
        # Calculate the result using the method
        result = hypotenuse(x, y, method)
        
        # Print the result and the fastest execution time
        print(f"Method: {method}, Result: {result}, Fastest Time: {fastest_time:.6f} seconds")



if __name__ == '__main__':
    main()
