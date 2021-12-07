import pandas as pd
from scipy.io import arff
import os
import time
from joblib import Parallel, delayed
import pickle
from Run import *


datasets =\
    (
        'CBF',
        'ItalyPowerDemand',
        'ECG200',
        'ECGFiveDays',
        'Plane',
        'ShapeletSim',
        'SonyAIBORobotSurface1',
        'SonyAIBORobotSurface2',
        'Trace',
        'TwoLeadECG'
    )

lenSubsequences_dict =\
    {
        'CBF': {
            'FOTS': [4, 5, 6, 7, 8],
            'UED': [36, 37, 38, 39, 41, 43, 44],
            'ED': [42, 43, 44, 45, 46]
        },
        'ItalyPowerDemand': {
            'FOTS': [6, 8, 10],
            'UED': [6, 8, 10],
            'ED': [6, 8, 10]
        },
        'ECG200': {
            'FOTS': [7, 8, 9, 10],
            'UED': [31, 32, 33, 34],
            'ED': [20, 22, 24, 29, 31, 33]
        },
        'ECGFiveDays': {
            'FOTS': [10, 14, 18],
            'UED': [30, 32, 34, 36, 38],
            'ED': [30, 32, 34, 36, 38]
        },
        'Plane': {
            'FOTS': [5, 6, 7, 8],
            'UED': [4, 5, 6, 8, 10, 12, 13],
            'ED': [17, 18, 19, 20]
        },
        'ShapeletSim': {
            'FOTS': [13, 14, 15],
            'UED': [11, 14, 15, 16],
            'ED': [10, 12, 13, 14]
        },
        'SonyAIBORobotSurface1': {
            'FOTS': [7, 8, 9, 10, 11],
            'UED': [17, 19, 21, 29, 31, 33],
            'ED': [21, 23, 25, 27, 29]
        },
        'SonyAIBORobotSurface2': {
            'FOTS': [11, 15, 19],
            'UED': [14, 15, 16, 18, 19, 20],
            'ED': [9, 11, 13, 17, 19, 21]
        },
        'Trace': {
            'FOTS': [7, 8, 9, 12, 13, 14],
            'UED': [10, 11, 12, 14, 15, 16, 18, 19, 20],
            'ED': [10, 11, 12, 18, 19, 20]
        },
        'TwoLeadECG': {
            'FOTS': [13, 14, 15, 16, 17, 18],
            'UED': [14, 17, 20],
            'ED': [14, 17, 20]
        }
    }


def load_dataset(dataset, uncertainty_level):
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

    return timeseries, deltas, labels


def run_on_dataset(dataset, uncertainty_level, similarity_measure):
    print('run_on_dataset')
    print('dataset:', dataset)
    print('uncertainty_level:', uncertainty_level)
    print('similarity_measure:', similarity_measure)

    timeseries, deltas, labels = load_dataset(dataset, uncertainty_level)

    results_file_txt = 'results_' + dataset + '_' + uncertainty_level + '_' + similarity_measure + '.txt'
    results_file_dat = 'results_' + dataset + '_' + uncertainty_level + '_' + similarity_measure + '.dat'

    if os.path.exists(results_file_txt):
        os.remove(results_file_txt)

    lenSubsequences = lenSubsequences_dict[dataset][similarity_measure]
    # lenSubsequences = range(4, len(timeseries) // 2 + 1, 2)

    results = {}
    for lenSubsequence in lenSubsequences:
        print('lenSubsequence:', lenSubsequence)
        results[lenSubsequence] = {}

        start = time.time()
        RI, num_clusters, uShapelets, clusters = Run(timeseries, deltas, labels, lenSubsequence = lenSubsequence, similarity_measure = similarity_measure)
        end = time.time()

        results[lenSubsequence]['RI'] = RI
        results[lenSubsequence]['num_clusters'] = num_clusters
        results[lenSubsequence]['uShapelets'] = uShapelets
        results[lenSubsequence]['clusters'] = clusters
        results[lenSubsequence]['time'] = end - start

        try:
            with open(results_file_txt, 'a') as f:
                f.write(f'lenSubsequence: {lenSubsequence}')
                f.write(str(results[lenSubsequence]))
                f.write('\n')
        except Exception as e:
            print(e)

    print('dataset:', dataset)
    print('uncertainty_level:', uncertainty_level)
    print('similarity_measure:', similarity_measure)
    print(results)
    try:
        with open(results_file_dat, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    Parallel(n_jobs = -1)(delayed(run_on_dataset)(dataset, uncertainty_level, similarity_measure) \
        for dataset in datasets \
        for uncertainty_level in ['0_1', '0_8', '2_0'] \
        for similarity_measure in ['FOTS', 'ED', 'UED'] \
    )
