import sys
import matmul as mat
import testbench as tb
import utils as utils

# Dims: (M, N, K)
# Dim A: (M, K)
# Dim B: (K, N)
# Dim C: (M, N)

def define_squares(min, stride, count):
    dims = []
    for i in range(count):
        dim_m = min + i * stride
        dim_n = min + i * stride
        dim_k = min + i * stride
        dims.append((min + i * stride, min + i * stride, min + i * stride))

    for dim in dims:
        # print size of A in MB
        size_A = dim[0] * dim[2] * 4 / 1024 / 1024
        size_B = dim[2] * dim[1] * 4 / 1024 / 1024
        size_C = dim[0] * dim[1] * 4 / 1024 / 1024
        print(f"A: {size_A} MB, B: {size_B} MB, C: {size_C} MB")

    return dims


def main():
    utils.print_gpu_info()

    methods = [ mat.Numpy, mat.CudaGlobalMemory, mat.CudaSharedMemoryGeneral ]
    dims = define_squares(min=256, stride=256, count=32) # (256, 256, 256) to (8192, 8192, 8192)

    test = tb.Testbench(methods, dims)
    test.test_all()
    results = test.get_results()
    utils.printout(results)
    utils.plot(results, 'main')


if __name__ == '__main__':
    # If executed on local, explicitly route all print() to file
    if len(sys.argv) > 1: 
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

