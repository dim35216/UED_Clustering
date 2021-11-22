import numpy as np


def UED(x, d_x, y, d_y):
    diff = abs(x - y)
    best_guess = np.sum(diff**2)
    delta = 2 * np.sum(diff * (d_x + d_y))

    return best_guess, delta
