import numpy as np
from GetUshapeletCandidates import *
from ComputeGap import *


def FindUShapelet(timeseries, labels, deltas, lenSubsequence, similarity_measure):
    ushapeletCandidates = GetUshapeletCandidates(timeseries, lenSubsequence)
    # print("ushapeletCandidates.shape:", ushapeletCandidates.shape)

    numCandidates = len(ushapeletCandidates)

    bestUshapeletCandidate = None
    gap = 0
    RI = 0
    IDX = 0
    distances = np.zeros(len(timeseries))
    delta_distances = np.zeros(len(timeseries))
    locations = np.zeros(len(timeseries), dtype=int)
    if numCandidates > 100:
        onePercent = np.round(numCandidates * 0.01)
        for i in range(int(onePercent)):
            # if i % 1 == 0:
            #     print("Compute gap for i:", i)
            curGap, curRI, curIDX, curDistances, curDelta_distances, curLocations = ComputeGap(ushapeletCandidates[i], lenSubsequence, timeseries, deltas, labels, similarity_measure)
            if curGap > gap:
                gap = curGap
                bestUshapeletCandidate = ushapeletCandidates[i]
                RI = curRI
                IDX = curIDX
                distances = curDistances
                delta_distances = curDelta_distances
                locations = curLocations
    
    return bestUshapeletCandidate, gap, RI, IDX, distances, delta_distances, locations
