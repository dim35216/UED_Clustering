import numpy as np
from core.ED import *
from core.FOTS import *
from core.UED import *


def findNN(shapelet, deltas_shapelet, ts, deltas_ts, similarity_measure):
    n = len(ts)
    m = len(shapelet)

    distances = np.empty(n - m, dtype=float)
    deltas_distances = np.zeros(n - m, dtype=float)
    if similarity_measure == 'FOTS':
        for i in range(n - m):
            distances[i] = FOTS(shapelet, ts[i : i + m])
    elif similarity_measure == 'UED':
        for i in range(n - m):
            distances[i], deltas_distances[i] = UED(shapelet, deltas_shapelet, ts[i : i + m], deltas_ts[i : i + m])
    else:
        for i in range(n - m):
            distances[i] = ED(shapelet, ts[i : i + m])
    
    NN_index = np.argmin(distances)
    NN_distance = distances[NN_index]
    NN_deltas_distance = deltas_distances[NN_index]

    # Normalize by the length of the shapelet
    NN_distance = NN_distance / m
    NN_deltas_distance = NN_deltas_distance / m

    return NN_distance, NN_deltas_distance, NN_index

    """ ts = np.concatenate(([0], ts))
    n = len(ts)

    shapelet = (shapelet - np.mean(shapelet)) / np.std(shapelet)
    m = len(shapelet)

    ts = np.concatenate((ts, [0 for _ in range(n, 2*n)]))
    shapelet = shapelet[::-1]                             
    shapelet = np.concatenate((shapelet, [0 for _ in range(m, 2*n)]))

    X = np.fft.fft(ts)
    Y = np.fft.fft(shapelet)
    Z = X * Y
    z = np.fft.ifft(Z)

    sumy = np.sum(shapelet)
    sumy2 = np.sum(shapelet**2)
    
    cum_sumx = np.cumsum(ts)
    cum_sumx2 =  np.cumsum(ts**2)
    sumx = cum_sumx[m:n+1]-cum_sumx[0:n-m+1]
    sumx2 = cum_sumx2[m:n+1]-cum_sumx2[0:n-m+1]
    meanx = sumx/m
    sigmax2 = (sumx2/m)-(meanx**2)
    sigmax = np.sqrt(sigmax2)

    dist = (sumx2 - 2*sumx*meanx + m*(meanx**2))/sigmax2 - 2*(z[m:n+1] - sumy*meanx)/sigmax + sumy2
    dist = np.sqrt(dist)
    
    loc = np.argmin(dist)
    bsf = np.min(dist)
    bsf = bsf/np.sqrt(m)

    return np.real(bsf), loc """
