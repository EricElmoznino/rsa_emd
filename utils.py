import numpy as np


def flatten_matrix(mat):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('The input must be a square matrix')
    n = mat.shape[0]
    flat_vector = np.zeros((n * (n - 1)) // 2)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            flat_vector[count] = mat[i, j]
            count += 1
    return flat_vector
