import numpy as np
from GetRandomProjectionsMatrix import *


def GetUshapeletCandidates(timeseries, lenSubsequence):
    numTimeseries, lenTimeseries = timeseries.shape
    numSubsequencesPerTS = lenTimeseries - lenSubsequence + 1
    numSubsequences = numTimeseries * numSubsequencesPerTS

    lb = max(0.1 * numTimeseries, 2)
    ub = 0.9 * numTimeseries

    ushapeletCandidates = np.zeros((numSubsequences, lenSubsequence + 2))
    SAX_shapelets_TS = np.zeros((numSubsequences, 3))
    
    curRow = 0
    for i in range(numTimeseries):
        for j in range(numSubsequencesPerTS):
            ushapeletCandidates[curRow, 0] = i
            ushapeletCandidates[curRow, 1] = j
            SAX_shapelets_TS[curRow, 0] = i
            SAX_shapelets_TS[curRow, 1] = j
            subsequence = timeseries[i, j : j + lenSubsequence]
            if subsequence.std(ddof =1 ) > 0.0000001:
                ushapeletCandidates[curRow, 2:] = (subsequence - subsequence.mean()) / subsequence.std(ddof = 1)
            curRow += 1
    
    uShTS = GetRandomProjectionsMatrix(ushapeletCandidates, lenSubsequence, numSubsequences)

    SAX_shapelets_TS = np.concatenate((SAX_shapelets_TS, uShTS), axis=1)
    SAX_shapelets_TS_backup = SAX_shapelets_TS
    hashesTotal = SAX_shapelets_TS.shape[1] - 3

    rowsToDelete = np.ones((numSubsequences))

    for i in range(numSubsequences):
        #print(i, ub, lb, len(np.argwhere((SAX_shapelets_TS[i, 3:] > ub) | (SAX_shapelets_TS[i, 3:] < lb))), hashesTotal*0.5)
        if (len(np.argwhere((SAX_shapelets_TS[i, 3:] > ub) | (SAX_shapelets_TS[i, 3:] < lb))) > hashesTotal*0.5):
            rowsToDelete[i] = 0
    
    SAX_shapelets_TS = SAX_shapelets_TS[np.where(rowsToDelete==1)] 

    stds = np.std(SAX_shapelets_TS[:, 3:], ddof=1, axis=1)
    uShapeletsOrder = np.argsort(stds)
    stds = np.sort(stds)

    if len(stds) > 0:
        medianStd = stds[round(len(stds) / 2)]
        smallStds = stds[stds <= medianStd]
        uShapeletsOrder[stds <= medianStd] = uShapeletsOrder[np.random.permutation(len(smallStds))]
        SAX_shapelets_TS = SAX_shapelets_TS[uShapeletsOrder, :]

    otherInd = np.argwhere(rowsToDelete == 0).reshape(-1)
    otherInd = otherInd[np.random.permutation(len(otherInd))]
    SAX_shapelets_TS = np.concatenate((SAX_shapelets_TS, SAX_shapelets_TS_backup[otherInd]), axis=0)

    return SAX_shapelets_TS

def GetUshapeletCandidates_Slow(timeseries, lenSubsequence):
    numTimeseries, lenTimeseries = timeseries.shape
    numSubsequencesPerTS = lenTimeseries - lenSubsequence + 1
    numSubsequences = numTimeseries * numSubsequencesPerTS

    ushapeletCandidates = np.zeros((numSubsequences, 2))
    
    curRow = 0
    for i in range(numTimeseries):
        for j in range(numSubsequencesPerTS):
            ushapeletCandidates[curRow, 0] = i
            ushapeletCandidates[curRow, 1] = j
            curRow += 1
    
    return ushapeletCandidates
