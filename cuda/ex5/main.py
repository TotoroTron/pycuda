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

    methods = [ mat.Numpy, mat.CudaGlobalMemory, mat.CudaSharedMemory ]
    squares = []
    for i in range(1, 64+1):
        dim = 64 * i # 64 to 4096
        squares.append((dim, dim, dim))

    tb_squares = tb.Testbench(methods, squares)
    tb_squares.test_all()
    results = tb_squares.get_results()

    utils.plot(results)
    for idx, dim in enumerate(squares):
        for result in results:
            if result[1][idx] == dim:
                print(f"Method: {result[0]:<{20}}, dims: {str(result[1][idx]):<{20}}, pass(1)/fail(0): {result[2][idx]},\t time: {result[3][idx]}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        sys.stdout = open(output_file, 'w')
        sys.stderr = open(output_file.replace('.out', '.err'), 'w')

    main()

