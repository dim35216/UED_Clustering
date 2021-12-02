import numpy as np


def timeseries2symbol(shapelets, lenShapelet, cardinality, dimensionality):
    if cardinality > 20:
        return
    
    window_size = int(np.floor(lenShapelet / dimensionality))
    numShapelets = shapelets.shape[0]
    symbolic_data = np.zeros((numShapelets + 1, dimensionality))
    pointers = -1 * np.ones(numShapelets + 1)

    lastPointer = 0

    for i in range(numShapelets):
        shapelet = shapelets[i, :]

        if lenShapelet == dimensionality:
            PAA = shapelet
        else:
            if lenShapelet / dimensionality > np.floor(lenShapelet / dimensionality):
                temp = np.zeros((dimensionality, lenShapelet))
                for j in range(dimensionality):
                    temp[j, :] = shapelet

                expanded_shapelet = np.reshape(np.transpose(temp), (1, lenShapelet * dimensionality))
                PAA = np.mean(np.reshape(expanded_shapelet, (lenShapelet, dimensionality), order="F"), axis=0)
            else:
                PAA = np.mean(np.reshape(shapelet, (window_size, dimensionality)))

        current_string = map_to_string(PAA, cardinality)

        if not len(np.nonzero(current_string == symbolic_data[lastPointer, :])[0]) == dimensionality: 
            lastPointer = lastPointer + 1
            symbolic_data[lastPointer, :] = current_string[0, :]
            pointers[lastPointer] = i

    symbolic_data = symbolic_data[1:, :]
    pointers = pointers[1:]
    symbolic_data = symbolic_data[:lastPointer, :]
    pointers = pointers[:lastPointer] 

    return symbolic_data, pointers


def map_to_string(PAA, alphabet_size):
    string = np.zeros((1, len(PAA)))

    if alphabet_size == 2:
        cut_points  = [-np.inf, 0]
    elif alphabet_size == 3:
        cut_points  = [-np.inf, -0.43, 0.43]
    elif alphabet_size == 4:
        cut_points  = [-np.inf, -0.67, 0, 0.67]
    elif alphabet_size == 5:
        cut_points  = [-np.inf, -0.84, -0.25, 0.25, 0.84]
    elif alphabet_size == 6:
        cut_points  = [-np.inf, -0.97, -0.43, 0, 0.43, 0.97]
    elif alphabet_size == 7:
        cut_points  = [-np.inf, -1.07, -0.57, -0.18, 0.18, 0.57, 1.07]
    elif alphabet_size == 8:
        cut_points  = [-np.inf, -1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15]
    elif alphabet_size == 9:
        cut_points  = [-np.inf, -1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22]
    elif alphabet_size == 10:
        cut_points  = [-np.inf, -1.28, -0.84, -0.52, -0.25, 0., 0.25, 0.52, 0.84, 1.28]
    elif alphabet_size == 11:
        cut_points  = [-np.inf, -1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34]
    elif alphabet_size == 12:
        cut_points  = [-np.inf, -1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38]
    elif alphabet_size == 13:
        cut_points  = [-np.inf, -1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43]
    elif alphabet_size == 14:
        cut_points  = [-np.inf, -1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47]
    elif alphabet_size == 15:
        cut_points  = [-np.inf, -1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5]
    elif alphabet_size == 16:
        cut_points  = [-np.inf, -1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53]
    elif alphabet_size == 17:
        cut_points  = [-np.inf -1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56]
    elif alphabet_size == 18:
        cut_points  = [-np.inf, -1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59]
    elif alphabet_size == 19:
        cut_points  = [-np.inf, -1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62]
    elif alphabet_size == 20:
        cut_points  = [-np.inf, -1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]

    for i in range(len(PAA)):
        string[0, i] = np.sum((cut_points <= (PAA[i]))) 
    
    return string


def GetSaxHash(ushapeletCandidates, lenShapelet, cardinality, dimensionality):
    shapeletsHash = {}
    shapeletsStorage = {}
    numTimeseries = int(ushapeletCandidates[-1, 0]) + 1

    for i in range(numTimeseries):
        timeseriesIndex = np.argwhere(ushapeletCandidates[:, 0] == i).flatten()
        firstTimeseriesIndex = timeseriesIndex[0]
        curUshapeletCandidates = ushapeletCandidates[timeseriesIndex, 2:]
        lastPointer = curUshapeletCandidates.shape[0]
        symbolic_data, pointers = timeseries2symbol(curUshapeletCandidates, lenShapelet, cardinality, dimensionality)
        lenSymbolic_data = symbolic_data.shape[0]

        for j in range(lenSymbolic_data):
            keySAX = ''
            for s in symbolic_data[j, :]:
                keySAX = keySAX + str(int(s))
            key = int(keySAX)

            if key in shapeletsHash.keys():
                if i not in shapeletsHash[key]:
                    shapeletsHash[key] = shapeletsHash[key] + [i]
                if j == lenSymbolic_data - 1:
                    shapeletsStorage[key] = shapeletsStorage[key] + list(range(int(firstTimeseriesIndex + pointers[j]), int(firstTimeseriesIndex + lastPointer)))
                else:
                    shapeletsStorage[key] = shapeletsStorage[key] + list(range(int(firstTimeseriesIndex + pointers[j]), int(firstTimeseriesIndex + pointers[j + 1])))
            
            else:
                shapeletsHash[key] = [i]
                if j == lenSymbolic_data - 1:
                    shapeletsStorage[key] = list(range(int(firstTimeseriesIndex + pointers[j]), int(firstTimeseriesIndex + lastPointer)))
                else:
                    shapeletsStorage[key] = list(range(int(firstTimeseriesIndex + pointers[j]), int(firstTimeseriesIndex + pointers[j + 1])))

    return shapeletsHash, shapeletsStorage
