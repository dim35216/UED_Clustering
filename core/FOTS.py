import numpy as np


def FOTS(x, y, dim = 4):
    Cx = autoCorrelationMatrixSW(x)
    Cy = autoCorrelationMatrixSW(y)

    _, Vx = np.linalg.eig(Cx)
    _, Vy = np.linalg.eig(Cy)
    
    k = min(dim, Vx.shape[1])
    
    return np.linalg.norm(Vx[:, :k] - Vy[:, :k], 'fro')


# Auto correlation matrix using sliding window
def autoCorrelationMatrixSW(x, dim = 4):
    assert len(x) >= 2
    
    w = int(np.floor(len(x) / 2))
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
    return x.reshape((-1, 1)) * x.reshape((1, -1))
