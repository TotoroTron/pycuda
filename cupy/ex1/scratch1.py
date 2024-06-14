import sys
import matmul as mat
import testbench as tb
import utils as utils

def main():
    utils.print_gpu_info()
    methods = [ mat.Numpy, mat.CudaSharedMemory, mat.CupyMatmul ]
    validation_method = mat.CudaGlobalMemory
    dims = [ (256, 256, 256) ]

    test = tb.Testbench(dims, methods, validation_method)
    test.test_all()
    report = test.get_report()
    utils.printout(report)



if __name__ == '__main__':
    # If executed on local, explicitly route all print() to file
    if len(sys.argv) > 1: 
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

