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
import argparse

from src.data.data_utils import * 

gfile.mkdir('logs')
gfile.mkdir('regression_HP')

np.random.seed(0)
tf.random.set_seed(0)


"""## Model building"""

class EpochRandomTuner(kt.tuners.RandomSearch):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_tria:l`
    kwargs['batch_size'] = 256
    kwargs['epochs'] = 50
    super(EpochRandomTuner, self).run_trial(trial, *args, **kwargs)

def _build_vanilla_model(hp):
  dropout_normal = lambda x: keras.layers.Dropout(hp.Float('dropout', min_value=0, max_value=1, default=0.5))(x, training=None)
  inputs = keras.layers.Input((hp.get('input_dimension'),))
  net = keras.layers.Flatten(input_shape=(hp.get('input_dimension'),))(inputs)
  for i in range(hp.Int('depth', min_value=1, max_value=3)):
    net = dropout_normal(net)
    net = keras.layers.Dense(hp.Int('width', min_value = 16, max_value = 64, step=16), activation='relu')(net)
  net = dropout_normal(net)
  prediction = keras.layers.Dense(1, activation='linear')(net)

  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
        keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=10e-4, max_value=10e-1, sampling='log')),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mse'],
    )
  
  return model

def _build_dropout_model(hp):
  dropout_always = lambda x: keras.layers.Dropout(hp.Float('dropout', min_value=0, max_value=1, default=0.5))(x, training=True)

  inputs = keras.layers.Input((hp.get('input_dimension'),))
  net = keras.layers.Flatten(input_shape=(hp.get('input_dimension'),))(inputs)
  for i in range(hp.Int('depth', min_value=1, max_value=3)):
    net = dropout_always(net)
    net = keras.layers.Dense(hp.Int('width', min_value = 16, max_value = 64, step=16), activation='relu')(net)
  net = dropout_always(net)
  prediction = keras.layers.Dense(1, activation='linear')(net)

  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
        keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=10e-4, max_value=10e-1, sampling='log')),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mse'],
    )
  
  return model

def _build_ll_dropout_model(hp):
  dropout_rate = hp.Float('dropout', min_value=0, max_value=1, default=0.5)
  dropout_normal = lambda x: keras.layers.Dropout(dropout_rate)(x, training=None)
  dropout_always = lambda x: keras.layers.Dropout(dropout_rate)(x, training=True)

  inputs = keras.layers.Input((hp.get('input_dimension'),))
  net = keras.layers.Flatten(input_shape=(hp.get('input_dimension'),))(inputs)
  for i in range(hp.Int('depth', min_value=1, max_value=3)):
    net = dropout_normal(net)
    net = keras.layers.Dense(hp.Int('width', min_value = 16, max_value = 64, step=16), activation='relu')(net)
  net = dropout_always(net)
  prediction = keras.layers.Dense(1, activation='linear')(net)

  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
        keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=10e-4, max_value=10e-1, sampling='log')),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mse'],
    )
  
  return model


def _build_model(architecture):
  return {"vanilla": _build_vanilla_model, "dropout": _build_dropout_model, 
          "ll_dropout": _build_ll_dropout_model}[architecture]

def _build_vanilla_model_tuner(tuner):
  hp = tuner.get_best_hyperparameters()[0]
  dropout_normal = lambda x: keras.layers.Dropout(hp.Float('dropout', min_value=0, max_value=1, default=0.5))(x, training=None)
  inputs = keras.layers.Input((hp.get('input_dimension'),))
  net = keras.layers.Flatten(input_shape=(hp.get('input_dimension'),))(inputs)
  for i in range(hp.get('depth')):
    net = dropout_normal(net)
    net = keras.layers.Dense(hp.get('width'), activation='relu')(net)
  net = dropout_normal(net)
  prediction = keras.layers.Dense(1, activation='linear')(net)

  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
        keras.optimizers.Adam(learning_rate=hp.get('learning_rate', min_value=10e-4, max_value=10e-1, sampling='log')),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mse'],
    )
  
  return model

def _build_dropout_model_tuner(tuner):
  hp = tuner.get_best_hyperparameters()[0]
  dropout_always = lambda x: keras.layers.Dropout(hp.get('dropout'))(x, training=True)

  inputs = keras.layers.Input((hp.get('input_dimension'),))
  net = keras.layers.Flatten(input_shape=(hp.get('input_dimension'),))(inputs)
  for i in range(hp.get('depth')):
    net = dropout_always(net)
    net = keras.layers.Dense(hp.get('width'), activation='relu')(net)
  net = dropout_always(net)
  prediction = keras.layers.Dense(1, activation='linear')(net)

  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
        keras.optimizers.Adam(learning_rate=hp.get('learning_rate')),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mse'],
    )
  
  return model

def _build_ll_dropout_model_tuner(tuner):
  hp = tuner.get_best_hyperparameters()[0]
  dropout_rate = hp.get('dropout')
  dropout_normal = lambda x: keras.layers.Dropout(dropout_rate)(x, training=None)
  dropout_always = lambda x: keras.layers.Dropout(dropout_rate)(x, training=True)

  inputs = keras.layers.Input((hp.get('input_dimension'),))
  net = keras.layers.Flatten(input_shape=(hp.get('input_dimension'),))(inputs)
  for i in range(hp.get('depth')):
    net = dropout_normal(net)
    net = keras.layers.Dense(hp.get('width'), activation='relu')(net)
  net = dropout_always(net)
  prediction = keras.layers.Dense(1, activation='linear')(net)

  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
        keras.optimizers.Adam(learning_rate=hp.get('learning_rate')),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mse'],
    )
  
  return model


def _build_model_from_tuner(architecture):
  return {"vanilla": _build_vanilla_model_tuner, "dropout": _build_dropout_model_tuner, 
          "ll_dropout": _build_ll_dropout_model_tuner}[architecture]

def _get_best_tuner(dataset, method, split):
  X_train, y_train, X_validation, y_validation, X_test, y_test = _get_data_splits(dataset, split)
  X_train, y_train, X_validation, y_validation, X_test, y_test, X_normalizer, y_normalizer = _get_normalized_data(X_train, y_train, X_validation, y_validation, X_test, y_test)
  hp = kt.HyperParameters()
  hp.Fixed('input_dimension', X_train[0].shape[0])
  hp.Fixed('num_train_examples', X_train.shape[0])
  tuner = EpochRandomTuner(_build_model(method), 
                            objective = 'val_mse',
                            hyperparameters = hp, 
                            max_trials = 100, 
                            seed = 42, 
                            directory = 'regression_HP', 
                            project_name = f"{dataset}/{method}/{split}")
  tuner.reload()
  return tuner

def _build_and_train_from_tuner(dataset, method, split):
  X_train, y_train, X_validation, y_validation, X_test, y_test = _get_data_splits(dataset, split)
  X_train, y_train, X_validation, y_validation, X_test, y_test, X_normalizer, y_normalizer = _get_normalized_data(X_train, y_train, X_validation, y_validation, X_test, y_test)

  X_train_val = np.concatenate((X_train, X_validation))
  y_train_val = np.concatenate((y_train, y_validation))

  tuner = _get_best_tuner(dataset, method, split)

  model = _build_model_from_tuner(method)(tuner)

  model.fit(x=X_train_val, y=y_train_val, batch_size=32, epochs=50)

  return model

def predict_N_times(model, test_set, y_norm, num_predictions=200):
  predictions = np.squeeze(np.stack([y_norm.inverse_transform(model.predict(test_set)) for _ in range(num_predictions)], axis=1))
  return predictions

def compute_quantiles(predictions):
  return np.quantile(predictions, [.025, .975], axis=1)

def compute_coverage(predictions, y_norm, y_test):
  y_invnorm = y_norm.inverse_transform(y_test)
  coverage_boundaries = compute_quantiles(predictions)
  covered = np.array([coverage_boundaries[0,i]<y_invnorm[i] and coverage_boundaries[1,i]>y_invnorm[i] for i in range(len(y_invnorm))])
  return covered

"""# Find Hyperparameters"""

def main(): 
  parser = argparse.ArgumentParser("Find hyper-parameters for vanilla, Dropout, and LL Dropout networks")
  parser.add_argument("--method", type=int, help="Method to train")
  parser.add_argument("--dataset", type=str, help="Dataset to train on")
  args = parser.parse_args()
  for _DATASET in [args.dataset]: 
    for _METHOD in [args.method]: 
      for _SPLIT in range(20 if _DATASET!='protein-tertiary-structure' else 5):
        print(f"{_DATASET}/{_METHOD}/{_SPLIT}")
        if os.path.isdir(f"/data/regression_HP/{_DATASET}/{_METHOD}/{_SPLIT}"):
          continue
        X_train, y_train, X_validation, y_validation, X_test, y_test = _get_data_splits(_DATASET, _SPLIT)
        X_train, y_train, X_validation, y_validation, X_test, y_test, X_normalizer, y_normalizer = _get_normalized_data(X_train, y_train, X_validation, y_validation, X_test, y_test)
        hp = kt.HyperParameters()
        hp.Fixed('input_dimension', X_train[0].shape[0])
        hp.Fixed('num_train_examples', X_train.shape[0])
        tuner = EpochRandomTuner(_build_model(_METHOD), 
                              objective = 'val_mse',
                              hyperparameters = hp, 
                              max_trials = 100, 
                              seed = 42, 
                              directory = '/data/regression_HP', 
                              project_name = f"{_DATASET}/{_METHOD}/{_SPLIT}")
        tuner.search(X_train, y_train, validation_data = (X_validation, y_validation), verbose=0)

if __name__ == "__main__": 
    main()




