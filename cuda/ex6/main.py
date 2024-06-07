import sys
import matmul as mat
import testbench as tb
import utils as utils

def define_combinations(M, N, K):
    values = list(set([M, N, K])) # remove duplicates
    combinations = []

    for i in values:
        for j in values:
            for k in values:
                combinations.append((i, j, k))

    return combinations

def test_kernel(M, N, K):
    methods = [ mat.CudaGlobalMemory, mat.CudaSharedMemorySquare, mat.CudaSharedMemoryGeneral ]
    dims = define_combinations(M, N, K)
    test = tb.Testbench(methods, dims)
    test.test_all()
    results = test.get_results()
    utils.printout(results)


def main():
    utils.print_gpu_info()

    test_kernel(2, 3, 3)
    test_kernel(2, 2, 3)
    test_kernel(17, 16, 16)
    test_kernel(17, 17, 16)
    test_kernel(31, 32, 32)
    test_kernel(128, 256, 256)
    test_kernel(128, 128, 256)
    test_kernel(726, 859, 1000)



if __name__ == '__main__':
    # If executed on local, explicitly route all print() to file
    if len(sys.argv) > 1: 
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

