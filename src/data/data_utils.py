import numpy as np
import os
from sklearn.preprocessing import StandardScaler


"""# Reading Data"""

# Adapted from https://github.com/yaringal/DropoutUncertaintyExps/blob/master/experiment.py
# This git repo should be cloned into /data/ 

def get_data_file_paths(dataset):
  temp_DATA_DIRECTORY_PATH = f"/data/DropoutUncertaintyExps/UCI_Datasets/{dataset}/data/"
  temp_INDEX_FEATURES_FILE = temp_DATA_DIRECTORY_PATH + "index_features.txt"
  temp_INDEX_TARGET_FILE = temp_DATA_DIRECTORY_PATH + "index_target.txt"
  temp_DATA_FILE = temp_DATA_DIRECTORY_PATH + "data.txt"
  return temp_DATA_DIRECTORY_PATH, temp_INDEX_FEATURES_FILE, temp_INDEX_TARGET_FILE, temp_DATA_FILE

def get_index_train_test_path(_DATA_DIRECTORY_PATH, split_num, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt" 

def get_data(dataset):
  _DATA_DIRECTORY_PATH, _INDEX_FEATURES_FILE, _INDEX_TARGET_FILE, _DATA_FILE = get_data_file_paths(dataset)
  data = np.loadtxt(_DATA_FILE) 
  index_features = np.loadtxt(_INDEX_FEATURES_FILE)
  index_target = np.loadtxt(_INDEX_TARGET_FILE)

  X = data[ : , [int(i) for i in index_features.tolist()] ]
  y = data[ : , int(index_target.tolist()) ]

  return X, y

def get_data_splits(dataset, split): 
  X, y = get_data(dataset)

  _DATA_DIRECTORY_PATH, _, _, _ = get_data_file_paths(dataset)

  index_train = np.loadtxt(get_index_train_test_path(_DATA_DIRECTORY_PATH, split, train=True))
  index_test = np.loadtxt(get_index_train_test_path(_DATA_DIRECTORY_PATH, split, train=False))

  X_train = X[ [int(i) for i in index_train.tolist()] ]
  y_train = y[ [int(i) for i in index_train.tolist()] ]

  
  X_test = X[ [int(i) for i in index_test.tolist()] ]
  y_test = y[ [int(i) for i in index_test.tolist()] ]

  X_train_original = X_train
  y_train_original = y_train

  num_training_examples = int(0.8 * X_train.shape[0])
  
  X_validation = X_train[num_training_examples:, :]
  y_validation = y_train[num_training_examples:]

  X_train = X_train[0:num_training_examples, :]
  y_train = y_train[0:num_training_examples]

  return X_train.astype('float32'), y_train.astype('float32'), X_validation.astype('float32'), y_validation.astype('float32'), X_test.astype('float32'), y_test.astype('float32')

def get_normalized_data(xtrain, ytrain, xval, yval, xtest, ytest):

  X_normalizer = StandardScaler()
  X_normalizer.fit(xtrain)
  y_normalizer = StandardScaler()
  y_normalizer.fit(ytrain.reshape(-1,1))

  xtrain_norm = X_normalizer.transform(xtrain)
  xval_norm = X_normalizer.transform(xval)
  xtest_norm = X_normalizer.transform(xtest)

  ytrain_norm = y_normalizer.transform(ytrain.reshape(-1,1)).squeeze()
  yval_norm = y_normalizer.transform(yval.reshape(-1,1)).squeeze()
  ytest_norm = y_normalizer.transform(ytest.reshape(-1,1)).squeeze()

  return xtrain_norm, ytrain_norm, xval_norm, yval_norm, xtest_norm, ytest_norm, X_normalizer, y_normalizer