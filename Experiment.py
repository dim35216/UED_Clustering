import pandas as pd
from scipy.io import arff
import os
import time
from Run import *
from joblib import Parallel, delayed


datasets =\
    (
        # 'CBF',
        'Chinatown',
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

def run_on_dataset(dataset, uncertainty_level):
    print('dataset:', dataset)
    results = {}
    results[uncertainty_level] = {}
    results[uncertainty_level][dataset] = {}
    for similarity_measure in ('FOTS', 'UED', 'ED'):
        print('similarity_measure:', similarity_measure)
        results[uncertainty_level][dataset][similarity_measure] = {}
        for run in range(3):
            print('run:', run)
            results[uncertainty_level][dataset][similarity_measure][run] = {}

            dataset_path = os.path.join('uncertain_datasets', uncertainty_level, dataset)
            data_test        = pd.DataFrame(arff.loadarff(os.path.join(dataset_path, dataset + '_TEST.arff'))[0]).astype({'target': float})
            data_noise_test  = pd.DataFrame(arff.loadarff(os.path.join(dataset_path, dataset + '_NOISE_TEST.arff'))[0])
            data_train       = pd.DataFrame(arff.loadarff(os.path.join(dataset_path, dataset + '_TEST.arff'))[0]).astype({'target': float})
            data_noise_train = pd.DataFrame(arff.loadarff(os.path.join(dataset_path, dataset + '_NOISE_TEST.arff'))[0])

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

            start = time.time()
            RI = Run(timeseries, deltas, labels, lenSubsequence = 10, similarity_measure = similarity_measure)
            end = time.time()

            results[uncertainty_level][dataset][similarity_measure][run]['RI'] = RI
            results[uncertainty_level][dataset][similarity_measure][run]['time'] = end - start

            # run
            # break
        # similarity_measure
        # break
    # dataset

    print(results)

for uncertainty_level in ('0_1', '0_8', '2_0'):
    print('uncertainty_level:', uncertainty_level)
    # results[uncertainty_level] = {}
    # for dataset in datasets:
        
    #     break
    # uncertainty_level
    # break

    Parallel(n_jobs=3)(delayed(run_on_dataset)(dataset, uncertainty_level) for dataset in datasets)

# print(results)
