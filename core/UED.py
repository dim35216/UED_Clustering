import numpy as np


def UED(x, d_x, y, d_y):
    assert len(x) == len(y)
    assert len(x) == len(d_x)
    assert len(y) == len(d_y)

    euclidean_distance = np.sum((x - y)**2)
    uncertainty = 2 * np.sum(abs(x - y) * (d_x + d_y))

    return (euclidean_distance - uncertainty, euclidean_distance + uncertainty)
