import sys
import matmul as mat
import testbench as tb
import utils as utils

def test_kernel(methods, dims):
    test = tb.Testbench(methods, dims)
    test.test_all()
    results = test.get_results()
    utils.printout(results)

def print_divider():
    print("\n========================================================================================\n")

def main():
    utils.print_gpu_info()
    methods = [ mat.Numpy, mat.CudaGlobalMemory, mat.CudaSharedMemory1, mat.CudaSharedMemory2 ]

    print_divider()

    dims = []
    for i in range(1, 256+1):
        dims.append((32*i, 512, 512))
    test = tb.Testbench(methods, dims)
    test.test_all()
    report = test.get_report()
    utils.printout(report)
    utils.plot(report, vary_dim='M', filename='Varying_M')

    print_divider()

    dims = []
    for i in range(1, 256+1):
        dims.append((512, 32*i, 512))
    test = tb.Testbench(methods, dims)
    test.test_all()
    report = test.get_report()
    utils.printout(report)
    utils.plot(report, vary_dim='N', filename='Varying_N')

    print_divider()

    dims = []
    for i in range(1, 256+1):
        dims.append((512, 512, 32*i))
    test = tb.Testbench(methods, dims)
    test.test_all()
    report = test.get_report()
    utils.printout(report)
    utils.plot(report, vary_dim='K', filename='Varying_K')

    print_divider()

    dims = []
    for i in range(1, 256+1):
        dims.append((32*i, 32*i, 32*i))
    test = tb.Testbench(methods, dims)
    test.test_all()
    report = test.get_report()
    utils.printout(report)
    utils.plot(report, vary_dim='all', filename='Squares')


if __name__ == '__main__':
    # If executed on local, explicitly route all print() to file
    if len(sys.argv) > 1: 
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

