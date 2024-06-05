

# Dims: (M, N, K)
# Dim A: (M, K)
# Dim B: (K, N)
# Dim C: (M, N)

def define_combinations(M, N, K):
    values = list(set([M, N, K])) # remove duplicates
    combinations = []

    for i in values:
        for j in values:
            for k in values:
                combinations.append((i, j, k))

    return combinations


def main():
    dims = define_combinations(4, 3, 1)
    print(dims)

    dims = define_combinations(4, 4, 1)
    print(dims)

    dims = define_combinations(4, 4, 4)
    print(dims)


if __name__ == '__main__':
    main()