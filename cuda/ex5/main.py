import sys
import matmul as mat
import testbench as tb
import utils as utils

# Dims: (M, N, K)
# Dim A: (M, K)
# Dim B: (K, N)
# Dim C: (M, N)

def main():
    methods = [ mat.Numpy, mat.CudaSharedMemoryGeneral ]
    dims = [ (256, 512, 512), (512, 256, 512), (512, 512, 256) ]

    test = tb.Testbench(methods, dims)
    test.test_all()
    results = test.get_results()

    utils.print_gpu_info()
    # utils.plot(results) # plotting doesn't make sense here
    utils.printout(results)


if __name__ == '__main__':
    # If executed on local, explicitly route all print() to file
    if len(sys.argv) > 1: 
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

