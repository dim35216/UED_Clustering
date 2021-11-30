import pandas as pd
from scipy.io import arff
import os
import time
from joblib import Parallel, delayed
import pickle
from Run import *


datasets =\
    (
        # 'CBF',
        # 'ItalyPowerDemand',
        'ECG200',
        # 'ECGFiveDays',
        # 'Plane',
        # 'ShapeletSim',
        # 'SonyAIBORobotSurface1',
        # 'SonyAIBORobotSurface2',
        # 'Trace',
        # 'TwoLeadECG',
        # # 'Chinatown'
    )

lenSubsequences_dict =\
    {
        'CBF': {
            'FOTS': [6],
            'UED': [6, 39],
            'ED': [39]
        },
        'ItalyPowerDemand': {
            'FOTS': [8],
            'UED': [7, 8],
            'ED': [7]
        },
        'ECG200': {
            'FOTS': [5],
            'UED': [5, 24],
            'ED': [24]
        },
        'ECGFiveDays': {
            'FOTS': [14],
            'UED': [14, 32],
            'ED': [32]
        },
        'Plane': {
            'FOTS': [6],
            'UED': [6, 13],
            'ED': [13]
        },
        'ShapeletSim': {
            'FOTS': [10],
            'UED': [6, 10],
            'ED': [6]
        },
        'SonyAIBORobotSurface1': {
            'FOTS': [9],
            'UED': [9, 27],
            'ED': [27]
        },
        'SonyAIBORobotSurface2': {
            'FOTS': [15],
            'UED': [15],
            'ED': [15]
        },
        'Trace': {
            'FOTS': [10],
            'UED': [10, 15],
            'ED': [15]
        },
        'TwoLeadECG': {
            'FOTS': [15],
            'UED': [10, 15],
            'ED': [10]
        }
    }

def run_on_dataset(dataset, uncertainty_level):
    print('run_on_dataset')
    print('dataset:', dataset)
    print('uncertainty_level:', uncertainty_level)

    dataset_path = os.path.join('uncertain_datasets', uncertainty_level, dataset)
    data_test        = pd.DataFrame(arff.loadarff(os.path.join(dataset_path, dataset + '_TEST.arff'))[0]).astype({'target': float})
    data_noise_test  = pd.DataFrame(arff.loadarff(os.path.join(dataset_path, dataset + '_NOISE_TEST.arff'))[0])
    data_train       = pd.DataFrame(arff.loadarff(os.path.join(dataset_path, dataset + '_TRAIN.arff'))[0]).astype({'target': float})
    data_noise_train = pd.DataFrame(arff.loadarff(os.path.join(dataset_path, dataset + '_NOISE_TRAIN.arff'))[0])

    df = pd.concat([data_test, data_train], ignore_index=True)
    df_noise = pd.concat([data_noise_test, data_noise_train], ignore_index=True)  

    timeseries = df.drop('target', axis=1).to_numpy()
    deltas = df_noise.to_numpy()
    labels = df['target'].to_numpy(dtype=int)

    print("timeseries:")
    print(timeseries.shape)
    print("deltas:")
    print(deltas.shape)
    print("labels:")
    print(labels.shape)
    
    results = {}
    for similarity_measure in ['FOTS', 'UED', 'ED']:
        print('similarity_measure:', similarity_measure)
        results[similarity_measure] = {}

        lenSubsequences = lenSubsequences_dict[dataset][similarity_measure]
        # lenSubsequences = range(4, len(timeseries) // 2 + 1, 2)

        for lenSubsequence in lenSubsequences:
            print('lenSubsequence:', lenSubsequence)
            results[similarity_measure][lenSubsequence] = {}

            start = time.time()
            RI, num_clusters, uShapelets, clusters = Run(timeseries, deltas, labels, lenSubsequence = lenSubsequence, similarity_measure = similarity_measure)
            end = time.time()

            results[similarity_measure][lenSubsequence]['RI'] = RI
            results[similarity_measure][lenSubsequence]['num_clusters'] = num_clusters
            results[similarity_measure][lenSubsequence]['uShapelets'] = uShapelets
            results[similarity_measure][lenSubsequence]['clusters'] = clusters
            results[similarity_measure][lenSubsequence]['time'] = end - start

    print('dataset:', dataset)
    print('uncertainty_level:', uncertainty_level)
    print(results)
    try:
        with open('results_' + dataset + '_' + uncertainty_level + '.txt', 'w') as f:
            f.write(str(results))
    except Exception as e:
        print(e)
    try:
        with open('results_' + dataset + '_' + uncertainty_level + '.dat', 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        print(e)

Parallel(n_jobs = -1)(delayed(run_on_dataset)(dataset, uncertainty_level) for dataset in datasets for uncertainty_level in ['0_1', '0_8', '2_0'])
