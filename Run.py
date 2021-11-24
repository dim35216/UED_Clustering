import numpy as np
from sklearn import metrics
from FindUShapelet import *


def Run(timeseries, deltas, labels, lenSubsequence = 30, similarity_measure = "ED"):
    labels_copy = labels
    uShapelets = []
    firstGap = 0
    remainingIndices = np.array([index for index in range(len(timeseries))])
    clusters = np.zeros(len(timeseries), dtype=int)
    counter = 0
    # totalTime = 0

    while len(remainingIndices) > 3:
        counter += 1
        print()
        print("=====================================================================================")
        print("counter:", counter)
        print("remainingIndices:")
        print(remainingIndices)
        print("clusters:")
        print(clusters)
        print("firstGap:", firstGap)

        ushapelet, gap, RI, IDX = FindUShapelet(timeseries, labels, deltas, lenSubsequence, similarity_measure)
        print("ushapelet:", ushapelet)
        print("gap:", gap)
        print("RI:", RI)
        print("IDX:")
        print(IDX)

        if firstGap == 0:
            if gap > 0:
                firstGap = gap
            else:
                print("BREAK: the first gap is zero!")
                break
        else:
            if gap == 0:
                print("BREAK: the gap is zero!")
                break
            if gap < firstGap / 2:
                print("BREAK: the gap has shrinkened too fast!")
                break
        
        # sys.exit()

        ts_index = remainingIndices[int(ushapelet[0])]
        loc = int(ushapelet[1])
        
        indicesNewCluster = np.argwhere(IDX)

        uShapelets.append([ts_index, loc, lenSubsequence, gap])
        
        clusters[remainingIndices[indicesNewCluster]] = counter
        
        timeseries = np.delete(timeseries, np.concatenate(indicesNewCluster), axis=0)
        labels = np.delete(labels, np.concatenate(indicesNewCluster), axis=0)
        remainingIndices = np.delete(remainingIndices, np.concatenate(indicesNewCluster), axis=0)

        

    print()
    print()
    print()
    print("Result:")
    print("remainingIndices:")
    print(remainingIndices)
    print("clusters:")
    print(clusters)
    print("firstGap:", firstGap)

    resultRI = metrics.rand_score(clusters, labels_copy)
    print("RI:", resultRI)

    return resultRI, counter, uShapelets
