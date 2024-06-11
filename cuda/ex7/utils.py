from numba import cuda
import matplotlib.pyplot as plt
import numpy as np

def generate_combinations(M, N, K):
    values = list(set([M, N, K])) # remove duplicates
    combinations = []
    for i in values:
        for j in values:
            for k in values:
                combinations.append((i, j, k))
    return combinations

def generate_randoms(low=1, high=256, size=128):
    dims = np.random.randint(low, high, size=(size, 3))
    dims = [tuple(row) for row in dims]
    return dims

def print_gpu_info():
    device = cuda.get_current_device()
    
    # BASIC INFO
    print("Device name:", device.name)
    print("Compute capability:", device.compute_capability)

    # CURRENT MEMORY INFORMATION
    mem_info = cuda.current_context().get_memory_info()
    total_memory = mem_info[1]
    free_memory = mem_info[0]
    print("Total global memory (GB):", total_memory / (1024 ** 3))
    print("Free global memory (GB):", free_memory / (1024 ** 3))
    print("Used global memory (GB):", (total_memory - free_memory) / (1024 ** 3))

    # GRID AND BLOCK DIMENSIONS
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    max_block_dim_x = device.MAX_BLOCK_DIM_X
    max_block_dim_y = device.MAX_BLOCK_DIM_Y
    max_block_dim_z = device.MAX_BLOCK_DIM_Z
    max_grid_dim_x = device.MAX_GRID_DIM_X
    max_grid_dim_y = device.MAX_GRID_DIM_Y
    max_grid_dim_z = device.MAX_GRID_DIM_Z

    print("Max threads per block:", max_threads_per_block)
    print("Max block dimensions: x={}, y={}, z={}".format(max_block_dim_x, max_block_dim_y, max_block_dim_z))
    print("Max grid dimensions: x={}, y={}, z={}".format(max_grid_dim_x, max_grid_dim_y, max_grid_dim_z))

    # SHARED MEMORY INFORMATION
    shared_memory_per_block = device.MAX_SHARED_MEMORY_PER_BLOCK
    print("Max shared memory per block (KB):", shared_memory_per_block / 1024)


def plot(report, vary_dim, filename=''):

    plt.figure(figsize=(12, 8))
    for entry in report:
        method = entry[0]
        dims = entry[1]
        if vary_dim == 'M': dims = [dim[0] for dim in dims]
        elif vary_dim == 'N': dims = [dim[1] for dim in dims]
        elif vary_dim == 'K': dims = [dim[2] for dim in dims]
        times = entry[3]
        plt.plot(dims, times, label=method)

    device = cuda.get_current_device()

    plt.title(f'Performance Comparison on {str(device.name)}')
    plt.xlabel(f'Dimension {vary_dim} with others fixed at 512')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plot_{str(device.name)}_{filename}.png')

def printout(results):
    fail_count = 0
    dims = results[0][1]
    for idx, dim in enumerate(dims):
        for result in results:
            if result[1][idx] == dim:
                if result[2][idx] == 1:
                    fail_count += 1
                print(f"Method: {result[0]:<{26}}, dims: {str(result[1][idx]):<{20}}, pass(0)/fail(1): {result[2][idx]},\t time (s): {result[3][idx]}")
        print()

    print(f"Total Fail Count: {fail_count}")
                