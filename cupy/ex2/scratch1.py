import sys
import matmul as mat
import testbench as tb
import utils as utils

def main():
    utils.print_gpu_info()
    methods = [ mat.CudaSharedMemory, mat.CupyMatmul, mat.CupyDot ]
    validation_method = mat.CudaGlobalMemory

    dims =  [   (6144, 6144, 11312), (6144, 6144, 11712), (6144, 6144, 11904),
                (6144, 6144, 11936), (6144, 6144, 12192), (6144, 6144, 12208),
                (10864, 10864, 10864), (10944, 10944, 10944), (11104, 11104, 11104)
            ]

    test = tb.Testbench(dims, methods, validation_method)
    test.test_all()
    report = test.get_report()
    utils.printout(report)

if __name__ == '__main__':
    main()