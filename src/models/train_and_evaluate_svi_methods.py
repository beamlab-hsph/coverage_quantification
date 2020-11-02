import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
keras = tf.keras
tfd = tfp.distributions
gfile = tf.io.gfile
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import kerastuner as kt
import h5py

from src.data.data_utils import * 
from src.models.find_hyperparameters_svi_methods import * 


np.random.seed(0)
tf.random.set_seed(0)

def main(): 
  parser = argparse.ArgumentParser("Find hyper-parameters for vanilla, Dropout, and LL Dropout networks")
  parser.add_argument("--method", type=int, help="Method to train")
  parser.add_argument("--dataset", type=str, help="Dataset to train on")
  parser.add_argument("--outfile" type=str, help="Output path")
  args = parser.parse_args()

  with h5py.File(args.outfile, 'a') as f: 
    for _DATASET in [args.dataset]: 
      for _METHOD in [args.method]: 
        for _SPLIT in range(20 if _DATASET !='protein-tertiary-structure' else 5):
          print(f"{_DATASET}/{_METHOD}/{_SPLIT}")
          if f"{_DATASET}/{_METHOD}/{_SPLIT}" in f:
            continue
          tuner = _get_best_tuner(_DATASET, _METHOD, _SPLIT)
          X_train, y_train, X_validation, y_validation, X_test, y_test = _get_data_splits(_DATASET, _SPLIT)
          X_train, y_train, X_validation, y_validation, X_test, y_test, X_normalizer, y_normalizer = _get_normalized_data(X_train, y_train, X_validation, y_validation, X_test, y_test)
          X_train_val = np.concatenate((X_train, X_validation))
          y_train_val = np.concatenate((y_train, y_validation))
          vanilla_trainval_predictions = np.zeros((len(X_train_val), 40))
          vanilla_test_predictions = np.zeros((len(X_test), 40))
          for _ITER in range(1 if _METHOD!='vanilla' else 40):
            model = _build_and_train_from_tuner(_DATASET, _METHOD, _SPLIT, tuner)
            trainval_predictions = predict_N_times(model, X_train_val, y_normalizer, 200 if _METHOD!='vanilla' else 1)
            test_predictions = predict_N_times(model, X_test, y_normalizer, 200 if _METHOD!='vanilla' else 1)
            if _METHOD == 'vanilla':
              vanilla_trainval_predictions[:, _ITER] = trainval_predictions
              vanilla_test_predictions[:, _ITER] = test_predictions
              f.create_dataset(f"{_DATASET}/{_METHOD}/{_SPLIT}/{_ITER}/trainval/predictions", data=trainval_predictions)
              f.create_dataset(f"{_DATASET}/{_METHOD}/{_SPLIT}/{_ITER}/test/predictions", data=test_predictions)
            if _METHOD!='vanilla':
              trainval_quantiles = compute_quantiles(trainval_predictions)
              trainval_coverage = compute_coverage(trainval_predictions, y_normalizer, y_train_val)
              test_quantiles = compute_quantiles(test_predictions)
              test_coverage = compute_coverage(test_predictions, y_normalizer, y_test)
              f.create_dataset(f"{_DATASET}/{_METHOD}/{_SPLIT}/trainval/predictions", data=trainval_predictions)
              f.create_dataset(f"{_DATASET}/{_METHOD}/{_SPLIT}/test/predictions", data=test_predictions)
              f.create_dataset(f"{_DATASET}/{_METHOD}/{_SPLIT}/trainval/quantiles", data=trainval_quantiles)
              f.create_dataset(f"{_DATASET}/{_METHOD}/{_SPLIT}/trainval/coverage", data=trainval_coverage)
              f.create_dataset(f"{_DATASET}/{_METHOD}/{_SPLIT}/test/quantiles", data=test_quantiles)
              f.create_dataset(f"{_DATASET}/{_METHOD}/{_SPLIT}/test/coverage", data=test_coverage)
          if _METHOD == 'vanilla':
            trainval_quantiles = compute_quantiles(vanilla_trainval_predictions)
            trainval_coverage = compute_coverage(vanilla_trainval_predictions, y_normalizer, y_train_val)
            test_quantiles = compute_quantiles(vanilla_test_predictions)
            test_coverage = compute_coverage(vanilla_test_predictions, y_normalizer, y_test)
            f.create_dataset(f"{_DATASET}/{'ensemble'}/{_SPLIT}/trainval/predictions", data=vanilla_trainval_predictions)
            f.create_dataset(f"{_DATASET}/{'ensemble'}/{_SPLIT}/test/predictions", data=vanilla_test_predictions)
            f.create_dataset(f"{_DATASET}/{'ensemble'}/{_SPLIT}/trainval/quantiles", data=trainval_quantiles)
            f.create_dataset(f"{_DATASET}/{'ensemble'}/{_SPLIT}/trainval/coverage", data=trainval_coverage)
            f.create_dataset(f"{_DATASET}/{'ensemble'}/{_SPLIT}/test/quantiles", data=test_quantiles)
            f.create_dataset(f"{_DATASET}/{'ensemble'}/{_SPLIT}/test/coverage", data=test_coverage)



if __name__ == "__main__": 
    main()






