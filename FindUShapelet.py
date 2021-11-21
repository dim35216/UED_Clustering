import numpy as np
from GetUshapeletCandidates import *
from ComputeGap import *


def FindUShapelet(timeseries, labels, deltas, lenSubsequence, similarity_measure):
    ushapeletCandidates = GetUshapeletCandidates(timeseries, lenSubsequence)
    print("ushapeletCandidates.shape:", ushapeletCandidates.shape)

    numCandidates = len(ushapeletCandidates)

    bestUshapeletCandidate = None
    gap = 0
    RI = 0
    IDX = 0
    if numCandidates > 100:
        onePercent = np.round(numCandidates * 0.01)
        for i in range(int(onePercent)):
            curGap, curRI, curIDX = ComputeGap(ushapeletCandidates[i], lenSubsequence, timeseries, deltas, labels, similarity_measure)
            if curGap > gap:
                gap = curGap
                bestUshapeletCandidate = ushapeletCandidates[i]
                RI = curRI
                IDX = curIDX
    
    return bestUshapeletCandidate, gap, RI, IDX
