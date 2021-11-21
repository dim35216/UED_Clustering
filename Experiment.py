import pandas as pd
from scipy.io import arff
import os
import time
from Run import *


datasets =\
    (
        # 'CBF',
        'ECG200',
        'ECGFiveDays',
        'ItalyPowerDemand',
        'Plane',
        'ShapeletSim',
        'SonyAIBORobotSurface1',
        'SonyAIBORobotSurface2',
        'Trace',
        'TwoLeadECG'
    )

results = {}
for dataset in datasets:
    results[dataset] = {}
    for uncertainty_level in ('0_1', '0_8', '2_0'):
        results[dataset][uncertainty_level] = {}
        for run in range(3):
            results[dataset][uncertainty_level][run] = {}

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
            print(timeseries)
            print("labels:")
            print(labels)
            print("deltas:")
            print(deltas)

            start = time.time()
            RI = Run(timeseries, deltas, labels, similarity_measure='UED')
            end = time.time()

            results[dataset][uncertainty_level][run]['RI'] = RI
            results[dataset][uncertainty_level][run]['time'] = end - start

            break
        break
    break

print(results)
