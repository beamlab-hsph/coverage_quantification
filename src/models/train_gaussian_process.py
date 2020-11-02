# -*- coding: utf-8 -*-

import numpy as np
import os
from datetime import datetime
import argparse
import GPy
from src.data.data_utils import * 


"""# GP Experiments"""

def main(): 
  parser = argparse.ArgumentParser("Find parameters for Gaussian processes")
  parser.add_argument("--split", type=int, help="Split to train on")
  parser.add_argument("--dataset", type=str, help="Dataset to train on")
  args = parser.parse_args()
  for _DATASET in [args.dataset]: 
    for _METHOD in ['gp']: 
      for _SPLIT in [args.split]:
        np.random.seed(0)
        print(f"{_DATASET}/{_METHOD}/{_SPLIT}")
        if os.path.exists(f"models/{_DATASET}/{_METHOD}/{_SPLIT}"):
          continue
        X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_splits(_DATASET, _SPLIT)
        X_train, y_train, X_validation, y_validation, X_test, y_test, X_normalizer, y_normalizer = get_normalized_data(X_train, y_train, X_validation, y_validation, X_test, y_test)
        X_train_val = np.concatenate((X_train, X_validation))
        y_train_val = np.concatenate((y_train, y_validation))
        
        kernel = GPy.kern.RBF(input_dim = X_train.shape[1])
        m = GPy.models.SparseGPRegression(X_train_val, y_train_val[:, np.newaxis], kernel, num_inducing=10)
        m.optimize(messages=True)
        
        os.makedirs(f"models/{_DATASET}/{_METHOD}/{_SPLIT}/")
        np.save(f"models/{_DATASET}/{_METHOD}/{_SPLIT}/{_DATASET}_{_SPLIT}_gp_params.npy", m.param_array)

if __name__ == "__main__": 
    main()



