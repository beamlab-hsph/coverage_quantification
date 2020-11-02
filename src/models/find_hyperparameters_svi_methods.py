
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import h5py
import kerastuner as kt
from datetime import datetime
import argparse

from src.data.data_utils import * 

keras = tf.keras
tfd = tfp.distributions
gfile = tf.io.gfile


"""# Model building"""

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

def _posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  """Posterior function for variational layer."""
  n = kernel_size + bias_size
  c = np.log(np.expm1(1e-5))
  variable_layer = tfp.layers.VariableLayer(
      2 * n, dtype=dtype,
      initializer=tfp.layers.BlockwiseInitializer([
          keras.initializers.TruncatedNormal(mean=0., stddev=0.05, seed=None),
          keras.initializers.Constant(np.log(np.expm1(1e-5)))], sizes=[n, n]))

  def distribution_fn(t):
    scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, n:])
    return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                            reinterpreted_batch_ndims=1)
  distribution_layer = tfp.layers.DistributionLambda(distribution_fn)
  return tf.keras.Sequential([variable_layer, distribution_layer])

def _make_prior_fn(kernel_size, bias_size=0, dtype=None):
  del dtype 
  loc = tf.zeros(kernel_size + bias_size)
  def distribution_fn(_):
    return tfd.Independent(tfd.Normal(loc=loc, scale=1),
                            reinterpreted_batch_ndims=1)
  return distribution_fn

def make_divergence_fn_for_empirical_bayes(std_prior_scale, examples_per_epoch):
  def divergence_fn(q, p, _):
    log_probs = tfd.LogNormal(0., std_prior_scale).log_prob(p.stddev())
    out = tfd.kl_divergence(q, p) - tf.reduce_sum(log_probs)
    return out / examples_per_epoch
  return divergence_fn

def make_prior_fn_for_empirical_bayes(init_scale_mean=-1, init_scale_std=0.1):
  """Returns a prior function with stateful parameters for EB models."""
  def prior_fn(dtype, shape, name, _, add_variable_fn):
    """A prior for the variational layers."""
    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=(1,),
        initializer=tf.compat.v1.initializers.random_normal(
            mean=init_scale_mean, stddev=init_scale_std),
        dtype=dtype,
        trainable=False)
    loc = add_variable_fn(
        name=name + '_loc',
        initializer=keras.initializers.Zeros(),
        shape=shape,
        dtype=dtype,
        trainable=True)
    scale = 1e-6 + tf.nn.softplus(untransformed_scale)
    dist = tfd.Normal(loc=loc, scale=scale)
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
  return prior_fn

def dense_variational(units, activation, initial_kl_weight):
  return AnnealingDenseVariational(
      units,
      make_posterior_fn=_posterior_mean_field,
      make_prior_fn=make_prior_fn_for_empirical_bayes,
      activation=activation,
      kl_weight=initial_kl_weight)
  
def eb_dense_layer(units, activation, eb_prior_fn, divergence_fn):
  return tfp.layers.DenseReparameterization(
        units,
        activation=activation,
        kernel_prior_fn=eb_prior_fn,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            loc_initializer=keras.initializers.he_normal()),
        kernel_divergence_fn=divergence_fn)

def predict_N_times(model, test_set, y_norm, num_predictions=200):
  predictions = np.squeeze(np.stack([y_norm.inverse_transform(model.predict(test_set)) for _ in range(num_predictions)], axis=1))
  return predictions

def compute_quantiles(predictions):
  return np.quantile(predictions, [.05, .95], axis=1)

def compute_coverage(predictions, y_norm, y_test):
  y_invnorm = y_norm.inverse_transform(y_test)
  coverage_boundaries = compute_quantiles(predictions)
  covered = np.array([coverage_boundaries[0,i]<y_invnorm[i] and coverage_boundaries[1,i]>y_invnorm[i] for i in range(len(y_invnorm))])
  return covered

"""# SVI Experiments"""

def _build_svi_eb_model(hp): 
  div_fn = make_divergence_fn_for_empirical_bayes(hp.get('std_prior_scale'), hp.get('num_train_examples')//hp.get('batch_size'))
  eb_fn = make_prior_fn_for_empirical_bayes(hp.get('init_prior_scale_mean'), hp.get('init_prior_scale_std'))
  dropout_rate = hp.Float('dropout', min_value=0, max_value=.5, default=0.1)
  dropout_normal = lambda x: keras.layers.Dropout(dropout_rate)(x, training=None)
  inputs = keras.layers.Input((hp.get('input_dimension'),))
  width = hp.Int('width', min_value = 16, max_value = 64, step=16)
  for i in range(hp.Int('depth', min_value=1, max_value=3)):
    if i==0:
      net = tfp.layers.DenseReparameterization(
            width,
            activation='relu',
            kernel_prior_fn=eb_fn,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                loc_initializer=keras.initializers.he_normal()),
            kernel_divergence_fn=div_fn)(inputs)
    else:
      net = dropout_normal(net)
      net = tfp.layers.DenseReparameterization(
              width,
              activation='relu',
              kernel_prior_fn=eb_fn,
              kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                  loc_initializer=keras.initializers.he_normal()),
              kernel_divergence_fn=div_fn)(net)
  net = dropout_normal(net)
  net = keras.layers.Dense(
          2,
          activation='linear')(net)
  prediction = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])))(net)
  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
          keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=10e-4, max_value=10e-1, sampling='log')),
          loss=negative_log_likelihood,
          metrics=['mse'],
      )
  return model

def _build_ll_svi_eb_model(hp):
  div_fn = make_divergence_fn_for_empirical_bayes(hp.get('std_prior_scale'), hp.get('num_train_examples')//hp.get('batch_size'))
  eb_fn = make_prior_fn_for_empirical_bayes(hp.get('init_prior_scale_mean'), hp.get('init_prior_scale_std'))
  dropout_rate = hp.Float('dropout', min_value=0, max_value=.5, default=0.1)
  dropout_normal = lambda x: keras.layers.Dropout(dropout_rate)(x, training=None)
  inputs = keras.layers.Input((hp.get('input_dimension'),))
  width = hp.Int('width', min_value = 16, max_value = 64, step=16)
  depth = hp.Int('depth', min_value=2, max_value=3)
  for i in range(depth-1):
    if i==0:
      net = keras.layers.Dense(
          width,
          activation='relu')(inputs)
    else:
      net = dropout_normal(net)
      net = keras.layers.Dense(
          width,
          activation='relu')(net)
  net = dropout_normal(net)
  net = tfp.layers.DenseReparameterization(
              width,
              activation='relu',
              kernel_prior_fn=eb_fn,
              kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                  loc_initializer=keras.initializers.he_normal()),
              kernel_divergence_fn=div_fn)(net)
  net = dropout_normal(net)
  net = keras.layers.Dense(
          2,
          activation='linear')(net)
  prediction = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])))(net)
  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
          keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=10e-4, max_value=10e-1, sampling='log')),
          loss=negative_log_likelihood,
          metrics=['mse'],
      )
  return model

def _build_svi_eb_model_tuner(tuner):
  hp = tuner.get_best_hyperparameters()[0]
  div_fn = make_divergence_fn_for_empirical_bayes(hp.get('std_prior_scale'), hp.get('num_train_examples')//hp.get('batch_size'))
  eb_fn = make_prior_fn_for_empirical_bayes(hp.get('init_prior_scale_mean'), hp.get('init_prior_scale_std'))
  dropout_normal = lambda x: keras.layers.Dropout(hp.get('dropout'))(x, training=None)
  inputs = keras.layers.Input((hp.get('input_dimension'),))
  for i in range(hp.get('depth')):
    if i==0:
      net = tfp.layers.DenseReparameterization(
            hp.get('width'),
            activation='relu',
            kernel_prior_fn=eb_fn,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                loc_initializer=keras.initializers.he_normal()),
            kernel_divergence_fn=div_fn)(inputs)
    else:
      net = dropout_normal(net)
      net = tfp.layers.DenseReparameterization(
              hp.get('width'),
              activation='relu',
              kernel_prior_fn=eb_fn,
              kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                  loc_initializer=keras.initializers.he_normal()),
              kernel_divergence_fn=div_fn)(net)
  net = dropout_normal(net)
  net = keras.layers.Dense(
          2,
          activation='linear')(net)
  prediction = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])))(net)
  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
          keras.optimizers.Adam(learning_rate=hp.get('learning_rate')),
          loss=negative_log_likelihood,
          metrics=['mse'],
      )
  return model

def _build_ll_svi_eb_model_tuner(tuner):
  hp = tuner.get_best_hyperparameters()[0]
  div_fn = make_divergence_fn_for_empirical_bayes(hp.get('std_prior_scale'), hp.get('num_train_examples')//hp.get('batch_size'))
  eb_fn = make_prior_fn_for_empirical_bayes(hp.get('init_prior_scale_mean'), hp.get('init_prior_scale_std'))
  dropout_normal = lambda x: keras.layers.Dropout(hp.get('dropout'))(x, training=None)
  inputs = keras.layers.Input((hp.get('input_dimension'),))
  for i in range(hp.get('depth')-1):
    if i==0:
      net = keras.layers.Dense(
          hp.get('width'),
          activation='relu')(inputs)
    else:
      net = dropout_normal(net)
      net = keras.layers.Dense(
          hp.get('width'),
          activation='relu')(net)
  net = dropout_normal(net)
  net = tfp.layers.DenseReparameterization(
              hp.get('width'),
              activation='relu',
              kernel_prior_fn=eb_fn,
              kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                  loc_initializer=keras.initializers.he_normal()),
              kernel_divergence_fn=div_fn)(net)
  net = dropout_normal(net)
  net = keras.layers.Dense(
          2,
          activation='linear')(net)
  prediction = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])))(net)
  model = keras.Model(inputs=inputs, outputs=prediction)
  model.compile(
          keras.optimizers.Adam(learning_rate=hp_dict.get('learning_rate')),
          loss=negative_log_likelihood,
          metrics=['mse'],
      )
  return model

def _build_model(architecture):
  return {"svi": _build_svi_eb_model, 
          "ll_svi": _build_ll_svi_eb_model}[architecture]

def _build_model_from_tuner(architecture):
  return {"svi": _build_svi_eb_model_tuner, 
          "ll_svi": _build_ll_svi_eb_model_tuner}[architecture]

def _get_best_tuner(dataset, method, split):
  X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_splits(dataset, split)
  X_train, y_train, X_validation, y_validation, X_test, y_test, X_normalizer, y_normalizer = get_normalized_data(X_train, y_train, X_validation, y_validation, X_test, y_test)
  hp = kt.HyperParameters()
  hp.Fixed('input_dimension', X_train[0].shape[0])
  hp.Fixed('num_train_examples', X_train.shape[0])
  tuner = EpochRandomTuner(_build_model(method), 
                            objective = 'val_mse',
                            hyperparameters = hp, 
                            max_trials = 100, 
                            seed = 42, 
                            directory = 'regression_HP_SVI', 
                            project_name = f"{dataset}/{method}/{split}")
  tuner.reload()
  return tuner

def _build_and_train_from_tuner(dataset, method, split, tuner):
  X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_splits(dataset, split)
  X_train, y_train, X_validation, y_validation, X_test, y_test, X_normalizer, y_normalizer = get_normalized_data(X_train, y_train, X_validation, y_validation, X_test, y_test)

  X_train_val = np.concatenate((X_train, X_validation))
  y_train_val = np.concatenate((y_train, y_validation))

  model = _build_model_from_tuner(method)(tuner)
  print(model.summary())
  tensorboard_cb = keras.callbacks.TensorBoard(log_dir=f'/logs/tensorboard/{dataset}/{method}/{split}/')
  model.fit(x=X_train_val, y=y_train_val, batch_size=32, callbacks=[tensorboard_cb], epochs=1000)
  return model

class EpochRandomTuner(kt.tuners.RandomSearch):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    kwargs['batch_size'] = 32
    #kwargs['epochs'] = 300
    super(EpochRandomTuner, self).run_trial(trial, *args, **kwargs)

def main():
  parser = argparse.ArgumentParser("Find HPs for SVI and LL-SVI")
  parser.add_argument("--dataset", type=str, help="Dataset to train on")
  parser.add_argument("--method", default="", type=str, help="Method to train with (svi or ll_svi")
  args = parser.parse_args()

  """# Find Hyperparameters"""

  for _DATASET in [args.dataset]: 
    for _METHOD in [args.method]: 
      for _SPLIT in range(20 if _DATASET !='protein-tertiary-structure' else 5):
        np.random.seed(0)
        tf.random.set_seed(0)
        print(f"{_DATASET}/{_METHOD}/{_SPLIT}")
        if os.path.isdir(f"/data/regression_HP_SVI/{_DATASET}/{_METHOD}/{_SPLIT}"):
          continue
        print(f"{_DATASET}/{_METHOD}/{_SPLIT}")
        X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_splits(_DATASET, _SPLIT)
        X_train, y_train, X_validation, y_validation, X_test, y_test, X_normalizer, y_normalizer = get_normalized_data(X_train, y_train, X_validation, y_validation, X_test, y_test)
        hp = kt.HyperParameters()
        hp.Fixed('input_dimension', X_train[0].shape[0])
        hp.Fixed('num_train_examples', X_train.shape[0])
        hp.Fixed('std_prior_scale', 1.5) 
        hp.Fixed('init_prior_scale_mean', -1)
        hp.Fixed('init_prior_scale_std', .1)
        hp.Fixed('batch_size', 32)

        tensorboard_cb = keras.callbacks.TensorBoard(log_dir=f'/logs/tensorboard/{_DATASET}/{_METHOD}/{_SPLIT}')
        tuner = EpochRandomTuner(_build_model(_METHOD), 
                              objective = 'val_mse',
                              hyperparameters = hp, 
                              max_trials = 50, 
                              seed = 42, 
                              directory = '/data/regression_HP_SVI', 
                              project_name = f"{_DATASET}/{_METHOD}/{_SPLIT}")
        tuner.search(X_train, y_train, validation_data = (X_validation, y_validation), epochs=1000 if _METHOD=='svi' else 200, callbacks=[tensorboard_cb], verbose=0)


if __name__ == "__main__": 
    main()

