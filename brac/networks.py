# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from ebm_rl.brac import dbn
tfd = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 0


def get_spec_means_mags(spec):
  means = (spec.maximum + spec.minimum) / 2.0
  mags = (spec.maximum - spec.minimum) / 2.0
  means = tf.constant(means, dtype=tf.float32)
  mags = tf.constant(mags, dtype=tf.float32)
  return means, mags


class ActorNetwork(tf.Module):
  """Actor network."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      ):
    super(ActorNetwork, self).__init__()
    self._action_spec = action_spec
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.elu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        self._action_spec.shape[0] * 2,
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, state):
    h = state
    for l in self._layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    a_tanh_mode = tf.tanh(mean) * self._action_mags + self._action_means
    log_std = tf.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    a_distribution = tfd.TransformedDistribution(
        distribution=tfd.Independent(
            tfd.Normal(loc=tf.zeros(mean.shape), scale=1.0),
            reinterpreted_batch_ndims=1),
        bijector=tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=self._action_means),
            tfp.bijectors.Scale(scale=self._action_mags),
            tfp.bijectors.Tanh(),
            tfp.bijectors.Shift(shift=mean),
            tfp.bijectors.Scale(scale=std),
        ]))
    return a_distribution, a_tanh_mode

  def get_log_density(self, state, action):
    a_dist, _ = self._get_outputs(state)
    log_density = a_dist.log_prob(action)
    return log_density

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state):
    state = tf.cast(state, tf.float32)
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample()
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample_n(self, state, n=1):
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample(n)
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample(self, state):
    return self.sample_n(state, n=1)[1][0]

class ActorNoisyLinearNetwork(tf.Module):
  """Actor Noisy Linear network."""

  def __init__(
      self,
      action_spec,
      ):
    super(ActorNoisyLinearNetwork, self).__init__()
    self._action_spec = action_spec
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization())
    output_layer = tfa.layers.NoisyDense(
        self._action_spec.shape[0] * 2,
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, embedding, remove_noise):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h, remove_noise=remove_noise)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    a_tanh_mode = tf.tanh(mean) * self._action_mags + self._action_means
    log_std = tf.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    a_distribution = tfd.TransformedDistribution(
        distribution=tfd.Independent(
            tfd.Normal(loc=tf.zeros(mean.shape), scale=1.0),
            reinterpreted_batch_ndims=1),
        bijector=tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=self._action_means),
            tfp.bijectors.Scale(scale=self._action_mags),
            tfp.bijectors.Tanh(),
            tfp.bijectors.Shift(shift=mean),
            tfp.bijectors.Scale(scale=std),
        ]))
    return a_distribution, a_tanh_mode

  def get_log_density(self, embedding, action):
    a_dist, _ = self._get_outputs(embedding)
    log_density = a_dist.log_prob(action)
    return log_density

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, embedding, remove_noise=False):
    # tf.print(embedding.shape)
    a_dist, a_tanh_mode = self._get_outputs(embedding, remove_noise)
    a_sample = a_dist.sample()
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample_n(self, embedding, n=1):
    a_dist, a_tanh_mode = self._get_outputs(embedding)
    a_sample = a_dist.sample(n)
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample(self, embedding):
    return self.sample_n(embedding, n=1)[1][0]



class ActorLinearNetwork(tf.Module):
  """Actor Linear network."""

  def __init__(
      self,
      action_spec,
      ):
    super(ActorLinearNetwork, self).__init__()
    self._action_spec = action_spec
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization())
    output_layer = tf.keras.layers.Dense(
        self._action_spec.shape[0] * 2,
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    a_tanh_mode = tf.tanh(mean) * self._action_mags + self._action_means
    log_std = tf.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    a_distribution = tfd.TransformedDistribution(
        distribution=tfd.Independent(
            tfd.Normal(loc=tf.zeros(mean.shape), scale=1.0),
            reinterpreted_batch_ndims=1),
        bijector=tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=self._action_means),
            tfp.bijectors.Scale(scale=self._action_mags),
            tfp.bijectors.Tanh(),
            tfp.bijectors.Shift(shift=mean),
            tfp.bijectors.Scale(scale=std),
        ]))
    return a_distribution, a_tanh_mode

  def get_log_density(self, embedding, action):
    a_dist, _ = self._get_outputs(embedding)
    log_density = a_dist.log_prob(action)
    return log_density

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, embedding):
    # tf.print(embedding.shape)
    a_dist, a_tanh_mode = self._get_outputs(embedding)
    a_sample = a_dist.sample()
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample_n(self, embedding, n=1):
    a_dist, a_tanh_mode = self._get_outputs(embedding)
    a_sample = a_dist.sample(n)
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample(self, embedding):
    return self.sample_n(embedding, n=1)[1][0]


class CriticNetwork(tf.Module):
  """Critic Network."""

  def __init__(
      self,
      fc_layer_params=(),
      ):
    super(CriticNetwork, self).__init__()
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.elu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, state, action):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.float32)
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class CriticRBFNetworkDBN(tf.Module):
  """Critic RBF Network."""

  def __init__(
      self,
      fc_layer_params,
      trainable=True,
      ):
    super(CriticRBFNetworkDBN, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization(momentum=0.9995, epsilon=1e-8))
    self._layers.append(dbn.DecorelationNormalization(momentum=0.9995))
    # self._layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-8))
    # fc_layer_params = 4096
    bias_init = tf.keras.initializers.RandomUniform(minval=0, maxval=2*3.1415926, seed=None)
    output_layer = tf.keras.layers.Dense(
        fc_layer_params,
        activation=tf.math.cos,
        trainable=trainable,
        kernel_initializer='random_normal',
        bias_initializer=bias_init,
        )
    self._layers.append(output_layer)
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  def get_fourier(self, embedding):
    h = tf.cast(embedding, tf.float32)
    h = self._layers[1](h)
    return h

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class CriticRBFNetwork(tf.Module):
  """Critic RBF Network."""

  def __init__(
      self,
      fc_layer_params,
      trainable=True,
      ):
    super(CriticRBFNetwork, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization(momentum=0.9995, epsilon=1e-8))
    # self._layers.append(dbn.DecorelationNormalization(momentum=0.9995))
    # self._layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-8))
    bias_init = tf.keras.initializers.RandomUniform(minval=0, maxval=2*3.1415926, seed=None)
    # fc_layer_params = 4096
    output_layer = tf.keras.layers.Dense(
        fc_layer_params,
        activation=tf.math.cos,
        trainable=trainable,
        kernel_initializer='random_normal',
        bias_initializer=bias_init,
        )
    self._layers.append(output_layer)
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class CriticLinearNetwork(tf.Module):
  """Critic Linear Network."""

  def __init__(
      self,
      ):
    super(CriticLinearNetwork, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization(momentum=0.9995, epsilon=1e-8))
    # self._layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-8))
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class CriticNoisyLinearNetwork(tf.Module):
  """Critic Noisy Linear Network."""

  def __init__(
      self,
      ):
    super(CriticNoisyLinearNetwork, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization())
    output_layer = tfa.layers.NoisyDense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class CriticMLPNetwork(tf.Module):
  """Linear Network."""

  def __init__(
      self,
      ):
    super(CriticMLPNetwork, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization())
    fc_layer_params=(100, 100)
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.elu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class RewardLinearNetwork(tf.Module):
  """Reward Linear Network."""

  def __init__(
      self,
      ):
    super(RewardLinearNetwork, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization(momentum=0.9995, epsilon=1e-8))
    # self._layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-8))
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    # h = tf.square(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class RewardMLPNetwork(tf.Module):
  """Reward MLP Network."""

  def __init__(
      self,
      ):
    super(RewardMLPNetwork, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization(momentum=0.9995, epsilon=1e-8))
    # self._layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-8))
    fc_layer_params=(512, 512)
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.elu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class LinearNetwork(tf.Module):
  """Linear Network."""

  def __init__(
      self,
      fc_layer_params=100,
      ):
    super(LinearNetwork, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization())
    output_layer = tf.keras.layers.Dense(
        fc_layer_params,
        activation=tf.math.cos,
        # kernel_initializer='random_normal',
        # bias_initializer='random_normal',
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    return h

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class MLPNetwork(tf.Module):
  """Linear Network."""

  def __init__(
      self,
      fc_layer_params=(100,),
      ):
    super(MLPNetwork, self).__init__()
    self._layers = []
    # self._layers.append(tf.keras.layers.BatchNormalization())
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.elu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        fc_layer_params,
        activation=tf.math.cos,
        )
    self._layers.append(output_layer)

  def __call__(self, embedding):
    h = tf.cast(embedding, tf.float32)
    for l in self._layers:
      h = l(h)
    return h

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class SAEmbeddingNetwork(tf.Module):
  """State-Action Embedding Network."""

  def __init__(
      self,
      fc_layer_params=(),
      initializer='glorot_uniform',
      ):
    super(SAEmbeddingNetwork, self).__init__()
    # fc_layer_params = [100]
    self._layers = []
    for n_units in fc_layer_params:
      if len(self._layers) == len(fc_layer_params) - 1:
        # activation = None
        activation = tf.nn.elu
        # self._layers.append(dbn.DecorelationNormalization(momentum=0.9995))
      else:
        activation = tf.nn.elu
      l = tf.keras.layers.Dense(
          n_units,
          activation=activation,
          kernel_initializer=initializer,
          # bias_initializer='normal',
          )
      self._layers.append(l)
      # if len(self._layers) < len(fc_layer_params) - 1:
      #    l = dbn.DecorelationNormalization(momentum=0.9995)
      #   self._layers.append(l)
      # if len(self._layers) != len(fc_layer_params) - 1:
      #   l = tf.keras.layers.LayerNormalization()
      #   self._layers.append(l)

  def __call__(self, state, action):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.float32)
    # cost = -tf.gather(state, [0], axis=-1) + 0.1 * tf.reduce_sum(tf.square(action), axis=-1, keepdims=True)
    # h = tf.concat([state, action, cost], axis=-1)
    h = tf.concat([state, action], axis=-1)
    h = self._layers[0](h)
    for l in self._layers[1:]:
      h = l(h)
    # h = h / tf.norm(h, axis=-1, keepdims=True)
    # h = tf.math.tanh(h)
    return h

  def get_rep(self, state, action):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.float32)
    h = tf.concat([state, action], axis=-1)
    # for i, l in enumerate(self._layers):
    #   if i != len(self._layers) - 1:
    #     h = l(h)
    return h
  
  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class RBFSANetwork(tf.Module):
  """State-Action Embedding Network."""

  def __init__(
      self,
      fc_layer_params=(),
      initializer='glorot_uniform',
      ):
    super(RBFSANetwork, self).__init__()
    # fc_layer_params = [100]
    self._layers = []
    for n_units in fc_layer_params:
      if len(self._layers) == len(fc_layer_params) - 1:
        activation = tf.nn.elu
      else:
        activation = tf.nn.elu
      l = tf.keras.layers.Dense(
          n_units,
          activation=activation,
          kernel_initializer=initializer,
          # bias_initializer='normal',
          )
      self._layers.append(l)
    # l = dbn.DecorelationNormalization(momentum=0.9995)
    # self._layers.append(l)
    n_units = fc_layer_params[-1]
    bias_init = tf.keras.initializers.RandomUniform(minval=0, maxval=2*3.1415926, seed=None)
    l = tf.keras.layers.Dense(
        n_units,
        activation=tf.math.cos,
        kernel_initializer='random_normal',
        bias_initializer=bias_init,
        )
    self._layers.append(l)
      # if len(self._layers) != len(fc_layer_params) - 1:
      #   l = tf.keras.layers.LayerNormalization()
      #   self._layers.append(l)

  def __call__(self, state, action):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.float32)
    # cost = -tf.gather(state, [0], axis=-1) + 0.1 * tf.reduce_sum(tf.square(action), axis=-1, keepdims=True)
    # h = tf.concat([state, action, cost], axis=-1)
    h = tf.concat([state, action], axis=-1)
    h = self._layers[0](h)
    for l in self._layers[1:]:
      h = l(h)
    # h = h / tf.norm(h, axis=-1, keepdims=True)
    # h = tf.math.tanh(h)
    return h

  def get_rep(self, state, action):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.float32)
    h = tf.concat([state, action], axis=-1)
    # for i, l in enumerate(self._layers):
    #   if i != len(self._layers) - 1:
    #     h = l(h)
    return h
  
  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class RBFSNetwork(tf.Module):
  """State Embedding Network."""

  def __init__(
      self,
      fc_layer_params=(),
      initializer='glorot_uniform',
      ):
    super(RBFSNetwork, self).__init__()
    self._layers = []
    for n_units in fc_layer_params:
      if len(self._layers) == len(fc_layer_params) - 1:
        activation = tf.nn.elu
      else:
        activation = tf.nn.elu
      l = tf.keras.layers.Dense(
          n_units,
          activation=activation,
          kernel_initializer=initializer,
          # bias_initializer='normal',
          )
      self._layers.append(l)
    # l = dbn.DecorelationNormalization(momentum=0.9995)
    # self._layers.append(l)
    n_units = fc_layer_params[-1]
    bias_init = tf.keras.initializers.RandomUniform(minval=0, maxval=2*3.1415926, seed=None)
    l = tf.keras.layers.Dense(
        n_units,
        activation=tf.math.cos,
        kernel_initializer='random_normal',
        bias_initializer=bias_init,
        )
    self._layers.append(l)

  def __call__(self, state):
    h = tf.cast(state, tf.float32)
    h = self._layers[0](h)
    for l in self._layers[1:]:
      h = l(h) 
    # h = h / tf.norm(h, axis=-1, keepdims=True)
    # h = tf.math.tanh(h)
    return h
  
  def get_rep(self, state):
    h = tf.cast(state, tf.float32)
    for i, l in enumerate(self._layers):
      if i != len(self._layers) - 1:
        h = l(h)
    return h
 
  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

class SEmbeddingNetwork1(tf.Module):
  """State Embedding Network."""

  def __init__(
      self,
      fc_layer_params=(),
      observation_spec=None,
      initializer='glorot_uniform',
      ):
    super(SEmbeddingNetwork1, self).__init__()
    self._layers = []
    for n_units in fc_layer_params:
      # n_units = int(n_units/2)
      if len(self._layers) == len(fc_layer_params) - 1:
        activation = None
        # self._layers.append(dbn.DecorelationNormalization(momentum=0.9995))
      else:
        activation = tf.nn.elu
      l = tf.keras.layers.Dense(
          n_units,
          activation=activation,
          kernel_initializer=initializer,
          # bias_initializer='normal',
          )
      self._layers.append(l)
    l = tf.keras.layers.Dense(
          observation_spec.shape[0],
          activation=activation,
          kernel_initializer=initializer,
          # bias_initializer='normal',
          )
    self._layers.append(l)
      # if len(self._layers) < len(fc_layer_params) - 1:
      #   l = dbn.DecorelationNormalization(momentum=0.9995)
      #   self._layers.append(l)
      # if len(self._layers) != len(fc_layer_params) - 1:
      #   l = tf.keras.layers.LayerNormalization()
      #   self._layers.append(l)

  def __call__(self, state, action):
    state = tf.cast(state, tf.float32)
    action = tf.cast(action, tf.float32)
    h = tf.concat([state, action], axis=-1)
    h = self._layers[0](h)
    for l in self._layers[1:]:
      h = l(h) 
    # h = h / tf.norm(h, axis=-1, keepdims=True)
    # h = tf.math.tanh(h)
    return h
  
  def get_rep(self, state):
    h = tf.cast(state, tf.float32)
    for i, l in enumerate(self._layers):
      if i != len(self._layers) - 1:
        h = l(h)
    return h
 
  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list



class SEmbeddingNetwork(tf.Module):
  """State Embedding Network."""

  def __init__(
      self,
      fc_layer_params=(),
      initializer='glorot_uniform',
      ):
    super(SEmbeddingNetwork, self).__init__()
    self._layers = []
    for n_units in fc_layer_params:
      # n_units = int(n_units/2)
      if len(self._layers) == len(fc_layer_params) - 1:
        activation = None
        # self._layers.append(dbn.DecorelationNormalization(momentum=0.9995))
      else:
        activation = tf.nn.elu
      l = tf.keras.layers.Dense(
          n_units,
          activation=activation,
          kernel_initializer=initializer,
          # bias_initializer='normal',
          )
      self._layers.append(l)
      # if len(self._layers) < len(fc_layer_params) - 1:
      #   l = dbn.DecorelationNormalization(momentum=0.9995)
      #   self._layers.append(l)
      # if len(self._layers) != len(fc_layer_params) - 1:
      #   l = tf.keras.layers.LayerNormalization()
      #   self._layers.append(l)

  def __call__(self, state):
    h = tf.cast(state, tf.float32)
    h = self._layers[0](h)
    for l in self._layers[1:]:
      h = l(h) 
    # h = h / tf.norm(h, axis=-1, keepdims=True)
    # h = tf.math.tanh(h)
    return h
  
  def get_rep(self, state):
    h = tf.cast(state, tf.float32)
    for i, l in enumerate(self._layers):
      if i != len(self._layers) - 1:
        h = l(h)
    return h
 
  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class BCQActorNetwork(tf.Module):
  """Actor network for BCQ."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      max_perturbation=0.05,
      ):
    super(BCQActorNetwork, self).__init__()
    self._action_spec = action_spec
    self._max_perturbation = max_perturbation
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.elu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        self._action_spec.shape[0],
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, state, action):
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    a = tf.tanh(h) * self._action_mags * self._max_perturbation + action
    a = tf.clip_by_value(
        a, self._action_spec.minimum, self._action_spec.maximum)
    return a

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state, action):
    return self._get_outputs(state, action)


class BCQVAENetwork(tf.Module):
  """VAE for learned behavior policy used by BCQ."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      latent_dim=None,
      ):
    super(BCQVAENetwork, self).__init__()
    if latent_dim is None:
      latent_dim = action_spec.shape[0] * 2
    self._action_spec = action_spec
    self._encoder_layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.elu,
          )
      self._encoder_layers.append(l)
    output_layer = tf.keras.layers.Dense(
        latent_dim * 2,
        activation=None)
    self._encoder_layers.append(output_layer)
    self._decoder_layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.elu,
          )
      self._decoder_layers.append(l)
    output_layer = tf.keras.layers.Dense(
        action_spec.shape[0],
        activation=None)
    self._decoder_layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)
    self._latent_dim = latent_dim

  @property
  def action_spec(self):
    return self._action_spec

  def forward(self, state, action):
    h = tf.concat([state, action], axis=-1)
    for l in self._encoder_layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    std = tf.exp(tf.clip_by_value(log_std, -4, 15))
    z = mean + std * tf.random.normal(shape=std.shape)
    a = self.decode(state, z)
    return a, mean, std

  def decode(self, state, z):
    h = tf.concat([state, z], axis=-1)
    for l in self._decoder_layers:
      h = l(h)
    a = tf.tanh(h) * self._action_mags + self._action_means
    # a = tf.clip_by_value(
    #     a, self._action_spec.minimum, self._action_spec.maximum)
    return a

  def sample(self, state):
    z = tf.random.normal(shape=[state.shape[0], self._latent_dim])
    z = tf.clip_by_value(z, -0.5, 0.5)
    return self.decode(state, z)

  def get_log_density(self, state, action):
    # variational lower bound
    a_recon, mean, std = self._p_fn.forward(state, action)
    log_2pi = tf.log(tf.constant(math.pi))
    recon = - 0.5 * tf.reduce_sum(
        tf.square(a_recon - action) + log_2pi, axis=-1)
    kl = 0.5 * tf.reduce_sum(
        - 1.0 - tf.log(tf.square(std)) + tf.square(mean) + tf.square(std),
        axis=-1)
    return recon - kl

  @property
  def weights(self):
    w_list = []
    for l in self._encoder_layers:
      w_list.append(l.weights[0])
    for l in self._decoder_layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state, action):
    return self._get_outputs(state, action)
