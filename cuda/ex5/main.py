import sys
import matmul as mat
import testbench as tb
import utils as utils

# Dims: (M, N, K)
# Dim A: (M, K)
# Dim B: (K, N)
# Dim C: (M, N)

def main():
    utils.print_gpu_info()

    methods = [ mat.Numpy, mat.CudaGlobalMemory, mat.CudaSharedMemoryGeneral ]
    # dims = [ (256, 512, 512), (512, 256, 512), (512, 512, 256) ]
    # dims = [ (2, 2, 2), (4, 2, 2), (2, 4, 2), (2, 2, 4) ]
    dims = [ (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096) ]

    print("Test of Testbench()")
    test = tb.Testbench(methods, dims)
    test.test_all()
    results = test.get_results()
    utils.printout(results)

    print("Test of Testbench_alt()")
    test_alt = tb.Testbench_alt(methods, dims)
    test_alt.test_all()
    results_alt = test_alt.get_results()
    utils.printout(results_alt)


if __name__ == '__main__':
    # If executed on local, explicitly route all print() to file
    if len(sys.argv) > 1: 
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

