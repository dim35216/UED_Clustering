from sklearn import metrics
from core.findNN import *


def ComputeGap(ushapeletCandidate, lenSubsequence, timeseries, deltas, labels, similarity_measure):
    numTimeseries = len(timeseries)

    ts_index = int(ushapeletCandidate[0])
    loc = int(ushapeletCandidate[1])
    shapelet = timeseries[ts_index, loc : loc + lenSubsequence]
    deltas_shapelet = deltas[ts_index, loc : loc + lenSubsequence]
    
    # Compute the distances from the shapelet to all timeseries
    distances = np.zeros(numTimeseries, dtype=float)
    deltas_distances = np.zeros(numTimeseries, dtype=float)
    locations = np.zeros(numTimeseries, dtype=int)
    for i in range(numTimeseries):
        distances[i], deltas_distances[i], locations[i] = findNN(shapelet, deltas_shapelet, timeseries[i, :], deltas[i, :], similarity_measure)
    
    # Sorting the distances to get the orderline
    if similarity_measure == 'UED':
        # Simple ordering
        indices = np.lexsort((deltas_distances, distances))
        orderline = distances[indices]
    else:
        orderline = np.sort(distances)

    # Finding the biggest gap in the orderline
    lb = int(np.ceil(numTimeseries * 0.167))
    ub = int(np.floor(numTimeseries * 0.833))

    maxGap = 0
    threshold = 0
    for i in range(lb, ub):
        curThreshold = orderline[i]
        curGap = gap(orderline, curThreshold)
        if curGap > maxGap:
            maxGap = curGap
            threshold = curThreshold

    # Evaluation of the threshold
    label = labels[ts_index]
    evaluation_labels = np.zeros(numTimeseries, dtype=int)
    for i in range(numTimeseries):
        if labels[i] == label:
            evaluation_labels[i] = 1
    
    IDX = np.zeros(numTimeseries, dtype=int)
    IDX[distances <= threshold] = 1

    RI = metrics.rand_score(evaluation_labels, IDX)

    return maxGap, RI, IDX


def gap(orderline, d):
    Da = np.argwhere(orderline <= d)
    Db = np.argwhere(d < orderline)

    # Both groups have to contain at least two objects
    if Da.shape[0] < 2 or Db.shape[0] < 2:
       return 0

    gap = 0
    r = len(np.argwhere(Da)) / len(np.argwhere(Db))
    
    # Ensure a high separation power
    if 0.2 < r and r < 5:
        ma = np.mean(orderline[Da])
        mb = np.mean(orderline[Db])
        sa = np.std(orderline[Da])
        sb = np.std(orderline[Db])
        gap = mb - sb - (ma + sa)

    return gap
