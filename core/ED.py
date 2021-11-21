import numpy as np

def ED(x, y):
    assert len(x) == len(y)
    
    return np.sqrt(np.sum((x - y)**2))