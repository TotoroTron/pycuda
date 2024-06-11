import sys
import matmul as mat
import testbench as tb
import utils as utils

def test_kernel(methods, dims):
    test = tb.Testbench(methods, dims)
    test.test_all()
    results = test.get_results()
    utils.printout(results)


def main():
    utils.print_gpu_info()
    methods = [ mat.Numpy, mat.JitNumpy, mat.CudaGlobalMemory, mat.CudaSharedMemory ]
    dims = utils.define_randoms(low=1, high=512, size=2048)
    test_kernel(methods, dims)


if __name__ == '__main__':
    # If executed on local, explicitly route all print() to file
    if len(sys.argv) > 1: 
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

