import csv
import h5py
import os
import numpy as np 

from data_utils import * 

DATASETS = choices=['bostonHousing', 'concerete', 'energy', 'kin8nm', 
      'naval-propulsion-plant', 'power-plant', 'protein-tertiary-structure', 'wine-quality-red', 'yacht']


with open('predictions/model_output_summary_non_svi_eb.csv', 'a') as csv_file:
  field_names = ['dataset', 'method', 'split', 'traintest', 'coverage', 'width', 'rmse']
  csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
  csv_writer.writeheader()
  with h5py.File('predictions/model_outputs_non_svi.hdf5', 'r') as f: 
    for _DATASET in DATASETS: 
      for _METHOD in ['ensemble', 'dropout', 'll_dropout']: 
        for _SPLIT in range(20 if _DATASET !='protein-tertiary-structure' else 5):
          X_train, y_train, X_validation, y_validation, X_test, y_test = _get_data_splits(_DATASET, _SPLIT)
          y_train_val = np.concatenate((y_train, y_validation))


          trainval_coverage = f[f"{_DATASET}/{_METHOD}/{_SPLIT}/trainval/coverage"]
          trainval_predictions = f[f"{_DATASET}/{_METHOD}/{_SPLIT}/trainval/predictions"]
          trainval_quantiles = f[f"{_DATASET}/{_METHOD}/{_SPLIT}/trainval/quantiles"]
          test_quantiles = f[f"{_DATASET}/{_METHOD}/{_SPLIT}/test/quantiles"]
          test_predictions = f[f"{_DATASET}/{_METHOD}/{_SPLIT}/test/predictions"]
          test_coverage = f[f"{_DATASET}/{_METHOD}/{_SPLIT}/test/coverage"]

          train_RMSE = np.sqrt(np.mean((y_train_val[:,np.newaxis]-trainval_predictions)**2))
          test_RMSE = np.sqrt(np.mean((y_test[:,np.newaxis]-test_predictions)**2))

          train_coverage = np.mean(trainval_coverage)
          test_coverage = np.mean(test_coverage)

          train_width = np.mean(trainval_quantiles[1,:]-trainval_quantiles[0,:])/np.std(y_train_val)
          test_width = np.mean(test_quantiles[1,:]-test_quantiles[0,:])/np.std(y_train_val)

          train_row_dict = {'dataset': _DATASET, 
                      'method': _METHOD, 
                      'split': _SPLIT, 
                      'traintest': 'trainval', 
                      'coverage': train_coverage, 
                      'width': train_width, 
                      'rmse': train_RMSE}
          test_row_dict = {'dataset': _DATASET, 
                      'method': _METHOD, 
                      'split': _SPLIT, 
                      'traintest': 'test', 
                      'coverage': test_coverage, 
                      'width': test_width, 
                      'rmse': test_RMSE}
          csv_writer.writerow(train_row_dict)
          csv_writer.writerow(test_row_dict)
