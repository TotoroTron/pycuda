import numpy as np

def verify(A, B):
    """
    Compare matrices A and B for equality.
    """
    return np.allclose(A, B)

def save_csvs(A, B, outputs):
    """
    Save matrices A, B, and outputs to CSV files.
    """
    # SAVE INPUTS TO CSV
    filename = f"logs/matA.csv"
    np.savetxt(filename, A, delimiter=',', fmt='%f')
    filename = f"logs/matB.csv"
    np.savetxt(filename, B, delimiter=',', fmt='%f')

    # SAVE OUTPUTS TO CSV
    for idx, mat in enumerate(outputs):
        filename = f"logs/matC_{idx}.csv"
        np.savetxt(filename, mat, delimiter=',', fmt='%f')
