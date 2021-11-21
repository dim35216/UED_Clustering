import numpy as np


def FOTS(x, y, dim = 4):
    assert len(x) == len(y)
    
    Cx = autoCorrelationMatrixSW(x)
    Cy = autoCorrelationMatrixSW(y)

    _, Vx = np.linalg.eig(Cx)
    _, Vy = np.linalg.eig(Cy)
    
    k = min(dim, Vx.shape[1])
    
    return np.linalg.norm(Vx[:, :k] - Vy[:, :k], 'fro')


# Auto correlation matrix using sliding window
def autoCorrelationMatrixSW(x, dim = 4):
    assert len(x) >= 2
    
    w = np.floor(len(x) / 2)
    if w < dim:
        w = len(x)
    
    step = w - 1
    lb = len(x) - step

    D = 0
    for i in range(lb):
        D = D + autoCorrelationMatrix(x[i : i + step])

    return D

# Auto correlation matrix
def autoCorrelationMatrix(x):
    D = correlationMatrix(x, x)
    return D

# Correlation matrix
def correlationMatrix(x, y):
    D = np.transpose(x) * y
    return D
