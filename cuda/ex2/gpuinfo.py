import numpy as np
import numba
from numba import cuda
import time

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

@cuda.jit
def do_something(data):
    idx = cuda.grid(1)
    if idx < data.size:
        data[idx] = 2 * np.sqrt(data[idx]) + 0.5 * np.sine(data[idx]) + 0.25 * np.cos(data[idx])

def main():
    print_gpu_info()
    print("\n")

    data = np.random.rand(10_000_000, dtype=np.float32)
    d_data = cuda.to_device(data)

    # Compile the function first
    do_something[16, 16](d_data) # arbitrary block and grid sizes, just force a compile

    # Experiment with different TPB configurations
    for i in range(1, 16):

        threads_per_block = 64 * i # Test 64, 128, 256, ..., 1024.
        blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block # integer division

        """
        threads_per_block = 256
        blocks_per_grid = (10000 + 256 - 1) // 256 = 10255 // 256 = 40

        threads_per_block = 512
        blocks_per_grid = (10000 + 512 - 1) // 512 = 10255 // 512 = 19

        etc...

        threads_per_block - 1 ensures any partial block gets counted as a full block
        """

        start = time.time()
        do_something[blocks_per_grid, threads_per_block](d_data)
        cuda.synchronize()
        end = time.time() - start
        print(f"TPB: {threads_per_block}, BPG: {blocks_per_grid}: {end:.6f} seconds.")

        result = d_data.copy_to_host()

    # Copy data back to host and check results
    result = d_data.copy_to_host()
    print(result)

if __name__ == "__main__":
    main()
