import sys
import matmul as mat
import testbench as tb
import utils as utils

def test_kernel(methods, dims):
    test = tb.Testbench(methods, dims)
    test.test_all()
    report = test.get_report()
    utils.printout(report)
    utils.plot(report)

def print_divider():
    print("\n========================================================================================\n")

def generate_stride(stride=16, count=128, vary_dim='M'):
    dims = []
    fixed_dim = (stride * count // 2) // 16 * 16 # Multiple of 16
    if vary_dim == 'M':
        for i in range(1, count+1):
            dims.append((stride*i, fixed_dim, fixed_dim))
    elif vary_dim == 'N':
        for i in range(1, count+1):
            dims.append((fixed_dim, stride*i, fixed_dim))
    elif vary_dim == 'K':
        for i in range(1, count+1):
            dims.append((fixed_dim, fixed_dim, stride*i))
    elif vary_dim == 'Squares':
        for i in range(1, count+1):
            dims.append((stride*i, stride*i, stride*i))
    else:
        print("Unrecognized vary_dim. Use 'M', 'N', 'K', or 'Squares'.")
        return

    return dims

def main():
    utils.print_gpu_info()
    methods = [ mat.Numpy, mat.CudaGlobalMemory, mat.CudaSharedMemory1, mat.CudaSharedMemory2 ]

    print_divider()

    dims = generate_stride(stride=16, count=128, vary_dim='M')
    test_kernel(methods, dims)
    print_divider()

    dims = generate_stride(stride=16, count=128, vary_dim='N')
    test_kernel(methods, dims)
    print_divider()

    dims = generate_stride(stride=16, count=128, vary_dim='K')
    test_kernel(methods, dims)
    print_divider()

    dims = generate_stride(stride=16, count=128, vary_dim='Squares')
    test_kernel(methods, dims)
    print_divider()

if __name__ == '__main__':
    # If executed on local, explicitly route all print() to file
    if len(sys.argv) > 1: 
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

