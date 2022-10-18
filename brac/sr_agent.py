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

"""Soft Actor Critic Agent.

Based on 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor' by Tuomas Haarnoja, et al.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow.compat.v1 as tf
from ebm_rl.brac import agent
from ebm_rl.brac import networks
from ebm_rl.brac import policies
from ebm_rl.brac import utils
import copy
import math
import tensorflow_probability as tfp
import random
import numpy as np
from ebm_rl.brac import whiten_utils

@gin.configurable
class Agent(agent.Agent):
  """SAC Agent."""

  def __init__(
      self,
      env='HalfCheetah-v2',
      q_embed_ckpt_file=None,
      p_embed_ckpt_file=None,
      ft_p_backbone=True,
      target_entropy=None,
      ensemble_q_lambda=1.0,
      update_model_freq=1, 
      use_rep=False,
      use_lsvi=False,
      random_feature=True,
      ft_q_backbone=False,
      model_update_tau=0.001,
      buffer_size=100000,
      model_learning_rate=0.0001,
      et=True,
      **kwargs):
    self._ensemble_q_lambda = ensemble_q_lambda
    self._target_entropy = target_entropy
    self._q_embed_ckpt_file = q_embed_ckpt_file
    self._p_embed_ckpt_file = p_embed_ckpt_file
    self._ft_p_backbone = ft_p_backbone
    self._update_model_freq = update_model_freq 
    self._use_rep = use_rep
    self._eta = tf.Variable(1., trainable=False)
    self._noise_type = 'gaussian'
    self._model_idx = tf.Variable(-1, trainable=False)
    self._use_lsvi = use_lsvi
    self._random_feature = random_feature
    self._train_model_steps = 1000000
    self._model_learning_rate = model_learning_rate
    # self._train_model_steps = 0
    self._model_update_tau = tf.Variable(model_update_tau, trainable=False)
    self._buffer_size = buffer_size
    self._ft_q_backbone = ft_q_backbone
    self._whitening = False
    self._env = env
    self._et = et
    super(Agent, self).__init__(**kwargs)
    print(self._weight_decays)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)
    if self._target_entropy is None:
      self._target_entropy = - self._action_spec.shape[0]
    self._q_fns = self._agent_module.q_nets
    self._p_fn = self._agent_module.p_fn
    self._qembed_fns = self._agent_module.q_embeds
    self._f_net_fns = self._agent_module.f_nets
    self._pembed_fn = self._agent_module.p_embed
    self._r_linear_fns = self._agent_module.r_linear_nets
    self._g_net_fns = self._agent_module.g_nets
    self._g1_net_fns = self._agent_module.g1_nets
    self._log_alpha = self._agent_module.log_alpha

  def _get_q_vars(self):
    return self._agent_module.q_source_variables

  def _get_q_target_vars(self):
    return self._agent_module.q_target_variables

  def _get_p_vars(self):
    assert self._ft_p_backbone == True
    if self._ft_p_backbone:
      return self._agent_module.p_variables + self._agent_module.pembed_variables
    else:
      return self._agent_module.p_variables

  def _get_g_vars(self):
    return self._agent_module.g_net_variables
 
  def _get_g1_vars(self):
    return self._agent_module.g1_net_variables
 
  def _get_r_linear_vars(self):
    return self._agent_module.r_linear_source_variables 

  def _get_q_weight_norm(self):
    weights = self._agent_module.q_source_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_p_weight_norm(self):
    weights = self._agent_module.p_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)
  
  def _get_g_net_weight_norm(self):
    weights = self._agent_module.g_net_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_g1_net_weight_norm(self):
    weights = self._agent_module.g1_net_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)
  
  def _get_qembed_vars(self):
    return self._agent_module.qembed_source_variables

  def _get_pembed_vars(self):
    return self._agent_module.pembed_variables

  def _get_g_net_vars(self):
    return self._agent_module.g_net_variables

  def _get_g1_net_vars(self):
    return self._agent_module.g1_net_variables
  
  def _get_f_net_vars(self):
    return self._agent_module.f_net_source_variables

  def _get_qembed_weight_norm(self):
    weights = self._agent_module.qembed_source_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_f_net_weight_norm(self):
    weights = self._agent_module.f_net_source_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_pembed_weight_norm(self):
    weights = self._agent_module.pembed_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def ensemble_q(self, qs):
    lambda_ = self._ensemble_q_lambda
    return (lambda_ * tf.reduce_min(qs, axis=-1)
            + (1 - lambda_) * tf.reduce_max(qs, axis=-1))

  def _ensemble_q2_target(self, q2_targets):
    return self.ensemble_q(q2_targets)

  def _ensemble_q1(self, q1s):
    return self.ensemble_q(q1s)

  def _build_model_loss(self, batch):
    s1 = batch['s1']
    s2 = batch['s2']
    a = batch['a1']
    r = batch['r']
    if self._env == 'HalfCheetah-v2':
      g_reward = tf.gather(tf.cast(s1, dtype=tf.float32), [8], axis=-1) - 0.1 * tf.reduce_sum(tf.square(tf.cast(a, dtype=tf.float32)), axis=-1, keepdims=True)
    elif self._env == 'Walker2d-v2':
      g_reward = tf.gather(tf.cast(s1, dtype=tf.float32), [8], axis=-1) - 3 * tf.math.square(tf.gather(tf.cast(s1, dtype=tf.float32), [0], axis=-1) - 1.3) - \
              0.1 * tf.reduce_sum(tf.square(tf.cast(a, dtype=tf.float32)), axis=-1, keepdims=True)
      r = r / 10
    elif self._env == 'Swimmer-v2':
      g_reward = tf.gather(tf.cast(s1, dtype=tf.float32), [3], axis=-1) - 0.0001 * tf.reduce_sum(tf.square(tf.cast(a, dtype=tf.float32)), axis=-1, keepdims=True)
      r = r / 2
    elif self._env == 'Hopper-v2':
      g_reward = tf.gather(tf.cast(s1, dtype=tf.float32), [5], axis=-1) - 3 * tf.math.square(tf.gather(tf.cast(s1, dtype=tf.float32), [0], axis=-1) - 1.3) - \
              0.1 * tf.reduce_sum(tf.square(tf.cast(a, dtype=tf.float32)), axis=-1, keepdims=True)
    elif self._env == 'Ant-v2':
      g_reward = tf.gather(tf.cast(s1, dtype=tf.float32), [13], axis=-1) - 3 * tf.math.square(tf.gather(tf.cast(s1, dtype=tf.float32), [0], axis=-1) - 0.57) - \
              0.1 * tf.reduce_sum(tf.square(tf.cast(a, dtype=tf.float32)), axis=-1, keepdims=True)
    else:
      assert False
    g_reward = tf.squeeze(g_reward, axis=-1)
    dsc = batch['dsc']
    if self._whitening:
      mean = tf.expand_dims(self._whitening_stats['state']['mean'], axis=0)
      std = tf.expand_dims(self._whitening_stats['state']['std'], axis=0)
      s2 = (s2 - tf.cast(mean, dtype=tf.float64))/tf.cast(std, dtype=tf.float64)
    # if self._use_random_linear:
    #   s_embed = self._p_linear_fn(s_embed)
    state_losses = []
    reward_losses = []
    success_losses = []
    for f_bundle, q_linear_bundle, r_linear_bundle, g_fn, g1_fn, f_net_std in zip(self._f_net_fns, self._q_fns, self._r_linear_fns, self._g_net_fns, self._g1_net_fns, self._agent_module._f_nets_std):
      f_fn, f_fn_target = f_bundle
      q_linear_fn, q_linear_fn_target = q_linear_bundle
      r_linear_fn, r_linear_fn_target = r_linear_bundle

      # g net loss
      s1_embed = g_fn(s1)
      pred_s2 = g1_fn(s1_embed, a)
      pred_r = r_linear_fn(s1_embed)
      pred_sloss = tf.square(pred_s2 - tf.cast(s2, dtype=tf.float32))
      pred_rloss = tf.square(r - pred_r)
      state_losses.append(tf.reduce_mean(pred_sloss))
      reward_losses.append(tf.reduce_mean(pred_rloss))

      # td learning for f
      _, a2, log_pi_a2 = self._p_fn(s2)
      f2_target = f_fn_target(s2, a2)
      f1_pred = f_fn(s1, a)
      f1_target = tf.stop_gradient(g_fn(s2) + self._discount * f2_target)
      f_loss = tf.reduce_mean(tf.square(f1_pred - f1_target))
      success_losses.append(f_loss)
    
    # model_loss = tf.gather(tf.stack(model_losses, axis=-1), self._model_idx)
    # reward_loss = tf.gather(tf.stack(reward_losses, axis=-1), self._model_idx)
    state_loss = tf.add_n(state_losses)
    success_loss = tf.add_n(success_losses)
    reward_loss = tf.add_n(reward_losses) 
    f_w_norm = self._get_f_net_weight_norm()
    g_w_norm = self._get_g_net_weight_norm()
    g1_w_norm = self._get_g1_net_weight_norm()
    norm_loss = self._weight_decays[0] * f_w_norm + self._weight_decays[0] * g_w_norm + self._weight_decays[0] * g1_w_norm 
    loss = (state_loss + reward_loss + success_loss + norm_loss) 

    info = collections.OrderedDict()
    info['state_loss'] = state_loss
    info['success_loss'] = success_loss
    info['reward_loss'] = reward_loss
    info['f_net_norm'] = f_w_norm
    info['g_norm'] = g_w_norm

    return loss, info 

  def _build_q_loss(self, batch):
    s1 = batch['s1']
    s2 = batch['s2']
    a = batch['a1']
    r = batch['r']
    dsc = batch['dsc']
    done = batch['done']
    if False and self._whitening:
      mean = tf.expand_dims(self._whitening_stats['state']['mean'], axis=0)
      std = tf.expand_dims(self._whitening_stats['state']['std'], axis=0)
      s1_whiten = (s1 - tf.cast(mean, dtype=tf.float64))/tf.cast(std, dtype=tf.float64)
      s2_whiten = (s2 - tf.cast(mean, dtype=tf.float64))/tf.cast(std, dtype=tf.float64)
    else:
      s1_whiten = s1
      s2_whiten = s2
    _, a2, log_pi_a2 = self._p_fn(s2)
    q2_targets = []
    q1_preds = []
    for qembed_bundle, q_bundle in zip(self._qembed_fns, self._q_fns):
      q_fn, q_fn_target = q_bundle
      qembed_fn, qembed_fn_target = qembed_bundle
      q2_target_ = q_fn_target(qembed_fn_target(s2, a2))
      q1_pred = q_fn(qembed_fn(s1, a))
      q1_preds.append(q1_pred)
      q2_targets.append(q2_target_)
    q1_preds = tf.stack(q1_preds, axis=-1)
    q2_targets = tf.stack(q2_targets, axis=-1)
    # TODO: changes here
    if tf.equal(self._model_idx, -1):
      if self._et:
        q2_target = self._ensemble_q2_target(q2_targets) * (1 - done)
      else:
        q2_target = self._ensemble_q2_target(q2_targets)
      v2_target = q2_target - tf.exp(self._log_alpha) * log_pi_a2
      q1_target = tf.stop_gradient(r + dsc * self._discount * v2_target)
      q_losses = []
      for i in range(q1_preds.shape[-1]):
        q1_pred = tf.gather(q1_preds, i, axis=-1)
        q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
        q_losses.append(q_loss_)
      q_loss = tf.add_n(q_losses)
    else:
      q1_pred = tf.gather(q1_preds, self._model_idx, axis=-1)
      if self._et:
        q2_target = tf.gather(q2_targets, self._model_idx, axis=-1) * (1 - done)
      else:
        q2_target = tf.gather(q2_targets, self._model_idx, axis=-1)
      v2_target = q2_target - tf.exp(self._log_alpha) * log_pi_a2
      q1_target = tf.stop_gradient(r + dsc * self._discount * v2_target)
      q_loss = tf.reduce_mean(tf.square(q1_pred - q1_target))
    q_w_norm = self._get_q_weight_norm()
    norm_loss = self._weight_decays[0] * q_w_norm * 0
    loss = q_loss + norm_loss

    info = collections.OrderedDict()
    info['q_loss'] = q_loss
    info['q_norm'] = q_w_norm
    info['q1_target_mean'] = tf.reduce_mean(q1_target)

    return loss, info

  def _build_p_loss(self, batch):
    s = batch['s1']
    r = batch['r']
    dsc = batch['dsc']
    if False and self._whitening:
      mean = tf.expand_dims(self._whitening_stats['state']['mean'], axis=0)
      std = tf.expand_dims(self._whitening_stats['state']['std'], axis=0)
      s_whiten = (s - tf.cast(mean, dtype=tf.float64))/tf.cast(std, dtype=tf.float64)
    else:
      s_whiten = s
    _, a, log_pi_a = self._p_fn(s)
    q1s = []
    for q_bundle, qembed_bundle in zip(self._q_fns, self._qembed_fns):
      q_fn, _ = q_bundle
      qembed_fn, qembed_fn_target = qembed_bundle
      q1_ = q_fn(qembed_fn(s, a))
      q1s.append(q1_)
    q1s = tf.stack(q1s, axis=-1)
    # TODO: changes here
    if tf.equal(self._model_idx, -1):
      q1 = self._ensemble_q1(q1s)
    else:
      q1 = tf.gather(q1s, self._model_idx, axis=-1)
    p_loss = tf.reduce_mean(tf.exp(self._log_alpha) * log_pi_a - q1)
    p_w_norm = self._get_p_weight_norm() +  self._get_pembed_weight_norm()
    norm_loss = self._weight_decays[1] * p_w_norm * 0
    loss = p_loss + norm_loss

    info = collections.OrderedDict()
    info['p_loss'] = p_loss
    info['p_norm'] = p_w_norm
    info['r_mean'] = tf.reduce_mean(r)
    info['dsc_mean'] = tf.reduce_mean(dsc)

    return loss, info

  def _build_a_loss(self, batch):
    s = batch['s1']
    _, _, log_pi_a = self._p_fn(s)
    alpha = tf.exp(self._log_alpha)
    a_loss = tf.reduce_mean(alpha * (-log_pi_a - self._target_entropy))

    info = collections.OrderedDict()
    info['a_loss'] = a_loss
    info['alpha'] = alpha

    return a_loss, info

  def _get_source_target_vars(self):
    return (self._agent_module.q_source_variables,
            self._agent_module.q_target_variables)

  def _get_source_target_embed_vars(self):
    return (self._agent_module.qembed_source_variables,
            self._agent_module.qembed_target_variables)
  
  def _get_source_target_f_net_vars(self):
    return (self._agent_module.f_net_source_variables,
            self._agent_module.f_net_target_variables)

  def _build_optimizers(self):
    opts = self._optimizers
    if len(opts) == 1:
      opts = tuple([opts[0]] * 4)
    elif len(opts) < 4:
      raise ValueError('Bad optimizers %s.' % opts)
    self._q_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
    # self._q_optimizer = utils.get_optimizer(opts[0][0])(lr=0.0001)
    self._p_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
    self._a_optimizer = utils.get_optimizer(opts[2][0])(lr=opts[2][1])
    # self._model_optimizer = utils.get_optimizer(opts[3][0])(lr=opts[3][1])
    # self._model_optimizer = utils.get_optimizer(opts[3][0])(lr=0.00003)
    self._model_optimizer = utils.get_optimizer(opts[3][0])(lr=self._model_learning_rate)
    if len(self._weight_decays) == 1:
      self._weight_decays = tuple([self._weight_decays[0]] * 4)

  def train_step(self):
    # if self._global_step % self._update_model_freq == 0 and self._global_step < self._train_model_steps:
    #   model_idx = random.randint(0, len(self._qembed_fns)-1)
    #   self._model_idx.assign(model_idx)
    for _ in range(0):
      train_batch = self._get_train_batch()
      self._optimize_model_alone(train_batch)
    if self._global_step < self._train_model_steps:
      train_batch = self._get_train_batch()
    else:
      # train_batch = self._get_train_batch(exclude=True)
      train_batch = self._get_train_batch()
    info = self._optimize_step(train_batch)
    #TODO: changes here 
    # if True and self._global_step < self._train_model_steps:
    if True:
      for _iter in range(2):
        # source_vars, target_vars = self._get_source_target_vars()
        # self._update_target_fns(source_vars, target_vars)
        # source_vars, target_vars = self._get_source_target_embed_vars()
        # self._update_target_fns(source_vars, target_vars)

        # self._update_target()
        train_batch = self._get_train_batch()
        self._optimize_q_alone(train_batch)
        # source_vars, target_vars = self._get_source_target_embed_vars()
        # utils.soft_variables_update(source_vars, target_vars, tau=1.0)
        self._optimize_p_alone(train_batch)
        # self._optimize_a_alone(train_batch)
    for key, val in info.items():
      self._train_info[key] = val.numpy()
    # self._global_step.assign_add(1)
    
  @tf.function
  def _optimize_q_alone(self, batch):
    q_info = self._optimize_q(batch)
    return None

  @tf.function
  def _optimize_a_alone(self, batch):
    a_info = self._optimize_a(batch)
    return None

  @tf.function
  def _optimize_p_alone(self, batch):
    a_info = self._optimize_p(batch)
    return None
  
  @tf.function
  def _optimize_model_alone(self, batch):
    model_info = self._optimize_model(batch)
    return None
  
  @tf.function
  def _update_target(self):
    if tf.equal(self._global_step % self._update_model_freq, 0) and (tf.less(self._global_step, self._train_model_steps) or tf.equal(self._global_step, self._train_model_steps)):
      f_net_source_vars, f_net_target_vars = self._get_source_target_f_net_vars()
      qembed_source_vars, qembed_target_vars = self._get_source_target_embed_vars()
      utils.soft_variables_update(f_net_source_vars, qembed_source_vars, tau=self._model_update_tau)
    # source_vars, target_vars = self._get_source_target_vars()
    # self._update_target_fns(source_vars, target_vars)
    # source_vars, target_vars = self._get_source_target_embed_vars()
    # self._update_target_fns(source_vars, target_vars)

  def _get_train_batch(self, exclude=False):
    """Samples and constructs batch of transitions."""
    if exclude:
      batch_indices = np.random.choice(np.setdiff1d(np.arange(self._train_data.size), np.arange(10000, 10000+self._train_model_steps)), self._batch_size)
    else:
      size = self._buffer_size
      if True and self._train_data.size > size:
        # batch_indices = np.random.choice(np.setdiff1d(np.arange(self._train_data.size), np.arange(0, self._train_data.size-size)), self._batch_size)
        batch_indices = np.random.randint(self._train_data.size-size, self._train_data.size, self._batch_size)
      else:
        batch_indices = np.random.choice(self._train_data.size, self._batch_size)
      # batch_indices = np.random.choice(np.setdiff1d(np.arange(self._train_data.size), np.arange(10000, 10000 + 
      #     self._update_model_freq * int((self._train_data.size-10000)/self._update_model_freq))), self._batch_size)
    batch_ = self._train_data.get_batch(batch_indices)
    transition_batch = batch_
    batch = dict(
        s1=transition_batch.s1,
        s2=transition_batch.s2,
        r=transition_batch.reward,
        dsc=transition_batch.discount,
        a1=transition_batch.a1,
        a2=transition_batch.a2,
        done=transition_batch.done,
        )
    return batch

  def _reset_weights(self, model):
    for layer in model._layers:
      if isinstance(layer, tf.keras.Model):
        reset_weights(layer)
        continue
      for k, initializer in layer.__dict__.items():
        if "initializer" not in k:
          continue
        # find the corresponding variable
        var = getattr(layer, k.replace("_initializer", ""))
        var.assign(initializer(var.shape, var.dtype)) 
  '''
  def _random_weights(self, model):
    for layer in model._layers:
      if isinstance(layer, tf.keras.Model):
        reset_weights(layer)
        continue
      for k, initializer in layer.__dict__.items():
        if "initializer" not in k:
          continue
        # find the corresponding variable
        var = getattr(layer, k.replace("_initializer", ""))
        var.assign(tf.random.normal(shape=var.shape, dtype=var.dtype)) 
  '''
  @tf.function
  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    self._eta.assign(self._eta * tf.exp(-0.00001))
    # if tf.less(self._global_step, 50000):
    #   self._model_update_tau.assign(tf.cast(self._global_step, dtype=tf.float32)*(0.1-0.0001)/tf.cast(self._train_model_steps, dtype=tf.float32))

    if tf.equal(self._global_step % self._update_freq, 0):
      source_vars, target_vars = self._get_source_target_vars()
      self._update_target_fns(source_vars, target_vars)
      source_vars, target_vars = self._get_source_target_embed_vars()
      self._update_target_fns(source_vars, target_vars)
      source_vars, target_vars = self._get_source_target_f_net_vars()
      self._update_target_fns(source_vars, target_vars)
    
    #TODO: changes here
    # if tf.equal(self._global_step % 1000, 0):
    #   source_vars, target_vars = self._get_source_target_vars()
    #   utils.soft_variables_update(source_vars, target_vars, tau=1.0)
    #   source_vars, target_vars = self._get_source_target_embed_vars()
    #   utils.soft_variables_update(source_vars, target_vars, tau=1.0)
    
    if False and tf.equal(self._global_step % 20000, 0) and tf.less(self._global_step, self._train_model_steps):
      tf.print('Reset sigma')
      weights = self._agent_module.p_variables + self._agent_module.q_source_variables
      for weight in weights:
        if 'sigma' in weight.name:
          weight.assign(weight*0 + 0.01)
    
    if False and tf.equal(self._global_step, self._train_model_steps):
      tf.print('Reinitialize Model')
      # f_net_source_vars, f_net_target_vars = self._get_source_target_f_net_vars()
      # qembed_source_vars, qembed_target_vars = self._get_source_target_embed_vars()
      # utils.soft_variables_update(f_net_source_vars, qembed_source_vars, tau=1.0)
      # utils.soft_variables_update(f_net_source_vars, qembed_target_vars, tau=1.0)
      # utils.soft_variables_update(f_net_target_vars, qembed_target_vars, tau=1.0)
      
      # self._reset_weights(self._agent_module.p_embed)
      # self._reset_weights(self._agent_module.p_net)
      # for q_fn, q_fn_target in self._q_fns:
      #   self._reset_weights(q_fn)
      #   self._reset_weights(q_fn_target)        
      for weight, weight_original in zip(self._p_vars, self._original_p_vars):
        weight.assign(weight_original)
      for weight, weight_original in zip(self._q_vars, self._original_q_vars):
        weight.assign(weight_original)
      for weight, weight_original in zip(self._q_target_vars, self._original_q_target_vars):
        weight.assign(weight_original) 
      self._agent_module._log_alpha.assign(0)
    # if tf.equal(self._global_step, self._train_model_steps):
    #   self._model_idx.assign(-1)

    if tf.equal(self._global_step % self._update_model_freq, 0) and (tf.less(self._global_step, self._train_model_steps) or tf.equal(self._global_step, self._train_model_steps)):
    # if tf.equal(self._global_step % self._update_model_freq, 0):
      f_net_source_vars, f_net_target_vars = self._get_source_target_f_net_vars()
      qembed_source_vars, qembed_target_vars = self._get_source_target_embed_vars()
      utils.soft_variables_update(f_net_source_vars, qembed_source_vars, tau=self._model_update_tau)
      # utils.soft_variables_update(f_net_source_vars, qembed_target_vars, tau=self._model_update_tau)
      # utils.soft_variables_update(f_net_source_vars, qembed_target_vars, tau=1.0)
      # utils.soft_variables_update(f_net_target_vars, qembed_target_vars, tau=1.0)

      # qembed_source_vars, qembed_target_vars = self._get_source_target_embed_vars()
      # tf.print(qembed_source_vars, qembed_target_vars)
      # utils.soft_variables_update(f_net_target_vars, qembed_target_vars, tau=1.0)

      if False:
        # Add noise to network
        if True:
          _eta = tf.cast(1/(self._global_step+1) * 1e-4, dtype=tf.float32)
          # weights = self._p_vars
          # weights = self._agent_module.p_variables
          # for weight in weights:
          #   random_weights = tf.random.normal(tf.shape(weight), dtype=tf.float32) * limit
            # random_weights = tf.random.uniform(tf.shape(weight), -limit, limit, dtype=tf.float32)
          #   weight.assign_add(random_weights)
          # for weight, weight_original in zip(self._q_vars, self._original_q_vars):
          #   weight.assign(weight_original)
          # for weight, weight_original in zip(self._q_target_vars, self._original_q_target_vars):
          #   weight.assign(weight_original)
          # weights = self._q_target_vars
          # for weight in weights:
          #   random_weights = tf.random.normal(tf.shape(weight), dtype=tf.float32) * limit
          #   # random_weights = tf.random.uniform(tf.shape(weight), -limit, limit, dtype=tf.float32)
          #   weight.assign_add(random_weights)
          weights = self._q_vars
          weights = [weights[2]] + [weights[3]] + [weights[6]] + [weights[7]]
          for weight in weights:
            random_weights = tf.random.normal(tf.shape(weight), dtype=tf.float32) * _eta
            # random_weights = tf.random.uniform(tf.shape(weight), -limit, limit, dtype=tf.float32)
            weight.assign_add(random_weights)
          # self._agent_module._log_alpha.assign_add(0.2)
        else:
          for weight, weight_original in zip(self._p_vars, self._original_p_vars):
            weight.assign(weight_original)
          for weight, weight_original in zip(self._q_vars, self._original_q_vars):
            weight.assign(weight_original)
          for weight, weight_original in zip(self._q_target_vars, self._original_q_target_vars):
            weight.assign(weight_original)
          self._agent_module._log_alpha.assign(0)
        # self._reset_weights(self._agent_module.p_embed)
        # self._reset_weights(self._agent_module.p_net)
        # for q_fn, q_fn_target in self._q_fns:
        #   self._reset_weights(q_fn)
        #   self._reset_weights(q_fn_target)        
        # self._agent_module._log_alpha.assign(0)
        
        #TODO: whether reset
        # for var in self._p_optimizer.variables():
        #   var.assign(tf.zeros_like(var))
        # for var in self._q_optimizer.variables():
        #   var.assign(tf.zeros_like(var))
        # for var in self._a_optimizer.variables():
        #   var.assign(tf.zeros_like(var))
    if self._ft_q_backbone and tf.greater(self._global_step, self._train_model_steps):
      q_info = self._optimize_q_ft(batch)
    else:
      q_info = self._optimize_q(batch)
    # source_vars, target_vars = self._get_source_target_embed_vars()
    # utils.soft_variables_update(source_vars, target_vars, tau=1.0)
    p_info = self._optimize_p(batch)
    a_info = self._optimize_a(batch)
    if tf.less(self._global_step, self._train_model_steps) and tf.equal(self._global_step % self._update_model_freq, 0):
      model_info = self._optimize_model(batch)
    else:
      model_info = collections.OrderedDict()
      model_info['state_loss'] = 0.0
      model_info['success_loss'] = 0.0
      model_info['reward_loss'] = 0.0
      model_info['f_net_norm'] = 0.0
      model_info['g_norm'] = 0.0
    info.update(q_info)
    info.update(p_info)
    info.update(a_info)
    info.update(model_info)

    if False and tf.less(self._global_step, 50000):
      weights = self._f_net_vars + self._g_net_vars
      for weight in weights :
        if self._noise_type == 'uniform':
          limit = 1e-3 * self._eta
          random_weights = tf.random.uniform(tf.shape(weight), -limit, limit, dtype=tf.float32)
        elif self._noise_type == 'gaussian':
          _threshold = 1e-4
          _eta = tf.cast(1/(self._global_step+1) * _threshold, dtype=tf.float32)
          # _threshold = 1e-5
          # _eta = tf.cast(1/tf.math.sqrt((tf.cast(self._global_step, dtype=tf.float32)+1)) * _threshold, dtype=tf.float32)
          # _eta = _threshold * self._eta
          random_weights = tf.random.truncated_normal(tf.shape(weight), dtype=tf.float32) * _eta
        else:
          random_weights = tf.random.normal(tf.shape(weight), dtype=tf.float32)
        # random_weights = random_weights * tf.cast(tf.math.greater(tf.math.abs(random_weights), 1e-3), dtype=tf.float32)
        #TODO: no random noise 
        weight.assign_add(random_weights)

    return info
  
  def _optimize_q(self, batch):
    vars_ = self._q_vars
    # vars_ = self._q_vars + self._qembed_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_q_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._q_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_q_ft(self, batch):
    vars_ = self._q_vars + self._qembed_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_q_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._q_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_p(self, batch):
    vars_ = self._p_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_p_loss(batch)
    grads = tape.gradient(loss, vars_)
    #TODO: added gradient clipping
    # grads, _grad_norm = tf.clip_by_global_norm(grads, 1.0)
    grads_and_vars = tuple(zip(grads, vars_))
    self._p_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_a(self, batch):
    vars_ = [self._log_alpha]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_a_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._a_optimizer.apply_gradients(grads_and_vars)
    return info
 
  def _optimize_model(self, batch):
    # vars_ = self._f_net_vars + self._r_linear_vars + self._g_net_vars + self._agent_module.f_net_std_variables
    # vars_ = self._f_net_vars + self._r_linear_vars
    vars_ = self._f_net_vars + self._r_linear_vars + self._g_net_vars + self._g1_net_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_model_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._model_optimizer.apply_gradients(grads_and_vars)
    # This part add random noise
    # weights = self._f_net_vars + self._r_linear_vars + self._g_net_vars
    weights = self._f_net_vars + self._g_net_vars 
    for weight in weights :
      if self._noise_type == 'uniform':
        limit = 1e-3 * self._eta
        random_weights = tf.random.uniform(tf.shape(weight), -limit, limit, dtype=tf.float32)
      elif self._noise_type == 'gaussian':
        _threshold = 1e-6
        _eta = tf.cast(1/tf.math.sqrt(tf.math.sqrt((tf.cast(self._global_step, dtype=tf.float32)+1))) * _threshold, dtype=tf.float32)
        # _eta = tf.cast(1/(tf.cast(self._global_step, dtype=tf.float32)+1) * _threshold, dtype=tf.float32)
        # _eta = _threshold * self._eta
        random_weights = tf.random.truncated_normal(tf.shape(weight), dtype=tf.float32) * _eta
        # random_weights = tf.random.normal(tf.shape(weight), dtype=tf.float32) * _eta
      # random_weights = random_weights * tf.cast(tf.math.greater(tf.math.abs(random_weights), 1e-3), dtype=tf.float32)
      #TODO: no random noise 
      # weight.assign_add(random_weights)
    return info

  def _build_test_policies(self):
    policy = policies.DecoupleDeterministicSoftPolicy(
    # policy = policies.DecoupleRandomSoftPolicy(
        a_embed=self._agent_module.p_embed,
        a_network=self._agent_module.p_net)
    self._test_policies['main'] = policy

  def _build_online_policy(self):
    return policies.DecoupleRandomSoftPolicy(
        a_embed=self._agent_module.p_embed,
        a_network=self._agent_module.p_net,
        )

  def _init_vars(self, batch):
    self._build_q_loss(batch)
    self._build_p_loss(batch)
    self._build_model_loss(batch)
    self._q_vars = self._get_q_vars()
    self._q_target_vars = self._get_q_target_vars()
    self._qembed_vars = self._get_qembed_vars()
    self._p_vars = self._get_p_vars()
    self._g_net_vars = self._get_g_net_vars()
    self._g1_net_vars = self._get_g1_net_vars()
    self._r_linear_vars = self._get_r_linear_vars()
    self._f_net_vars = self._get_f_net_vars()
    print(len(self._q_vars), len(self._q_target_vars), len(self._qembed_vars), len(self._p_vars), 
            len(self._g_net_vars), len(self._r_linear_vars), len(self._f_net_vars))
    f_net_source_vars, f_net_target_vars = self._get_source_target_f_net_vars()
    qembed_source_vars, qembed_target_vars = self._get_source_target_embed_vars()
    utils.soft_variables_update(f_net_source_vars, qembed_source_vars, tau=1.0)
    self._original_p_vars = copy.deepcopy(self._p_vars)
    q_vars, q_target_vars = self._get_source_target_vars()
    self._original_q_vars, self._original_q_target_vars = copy.deepcopy(q_vars), copy.deepcopy(q_target_vars)
     
  def _build_checkpointer(self):
    state_ckpt = tf.train.Checkpoint(
        policy=self._agent_module.p_net,
        agent=self._agent_module)
    q_embed_ckpt = tf.train.Checkpoint(
        q_embeds=self._agent_module.q_embeds)
    p_embed_ckpt = tf.train.Checkpoint(
        p_embed=self._agent_module.p_embed)
    g_net_ckpt = tf.train.Checkpoint(
        g_net=self._agent_module.g_nets)
    g1_net_ckpt = tf.train.Checkpoint(
        g1_net=self._agent_module.g1_nets)
    f_net_ckpt = tf.train.Checkpoint(
        f_nets=self._agent_module.f_nets)
    return dict(state=state_ckpt, q_embed=q_embed_ckpt, p_embed=p_embed_ckpt, g_net=g_net_ckpt, g1_net=g1_net_ckpt, f_net=f_net_ckpt)
 
  def save(self, ckpt_name):
    self._checkpointer['state'].write(ckpt_name)
    self._checkpointer['q_embed'].write(ckpt_name + '_q_embed')
    self._checkpointer['p_embed'].write(ckpt_name + '_p_embed') 
    self._checkpointer['g_net'].write(ckpt_name + '_g_net') 
    self._checkpointer['g1_net'].write(ckpt_name + '_g1_net') 
    self._checkpointer['f_net'].write(ckpt_name + '_f_net') 
  
  def _load_embeds(self):
    if self._q_embed_ckpt_file is not None:
      self._checkpointer['q_embed'].restore(
          self._q_embed_ckpt_file)
    if self._p_embed_ckpt_file is not None:
      self._checkpointer['p_embed'].restore(
          self._p_embed_ckpt_file)


class AgentModule(agent.AgentModule):
  """Tensorflow modules for SAC agent."""

  def _build_modules(self):
    self._q_embeds = []
    self._f_nets_std = []
    self._q_nets = []
    self._r_linear_nets = []
    self._f_nets = []
    self._g_nets = []
    self._g1_nets = []
    n_q_fns = self._modules.n_q_fns
    for _ in range(n_q_fns):
      self._q_nets.append(
          [self._modules.q_net_factory(),  # Learned Q-value.
           self._modules.q_net_factory(),]  # Target Q-value.
          )
    for _ in range(n_q_fns):
      self._q_embeds.append(
          [self._modules.q_embed_factory(),  # Learned Q-value.
           self._modules.q_embed_factory(),]  # Target Q-value.
          )
    for _ in range(n_q_fns):
      self._f_nets_std.append(
          self._modules.f_net_factory()
          )
    for _ in range(n_q_fns):
      self._f_nets.append(
          [self._modules.f_net_factory(),  # Learned Q-value.
           self._modules.f_net_factory(),]  # Target Q-value.
          )
    for _ in range(n_q_fns):
      self._r_linear_nets.append(
          [self._modules.r_linear_factory(),  # Learned Q-value.
          self._modules.r_linear_factory(),]  # Target Q-value.
          )
    for _ in range(n_q_fns):
      self._g_nets.append(self._modules.g_net_factory())
    for _ in range(n_q_fns):
      self._g1_nets.append(self._modules.g1_net_factory())
    self._p_embed = self._modules.p_embed_factory()
    self._p_net = self._modules.p_net_factory()
    self._log_alpha = tf.Variable(0.0)

  @property
  def log_alpha(self):
    return self._log_alpha

  @property
  def q_nets(self):
    return self._q_nets
  
  @property
  def r_linear_nets(self):
    return self._r_linear_nets

  @property
  def g_nets(self):
    return self._g_nets
   
  @property
  def g1_nets(self):
    return self._g1_nets

  @property
  def q_source_weights(self):
    q_weights = []
    for q_net, _ in self._q_nets:
      q_weights += q_net.weights
    return q_weights

  @property
  def q_target_weights(self):
    q_weights = []
    for _, q_net in self._q_nets:
      q_weights += q_net.weights
    return q_weights

  @property
  def q_source_variables(self):
    vars_ = []
    for q_net, _ in self._q_nets:
      vars_ += q_net.trainable_variables
    return tuple(vars_)

  @property
  def q_target_variables(self):
    vars_ = []
    for _, q_net in self._q_nets:
      vars_ += q_net.trainable_variables
    return tuple(vars_)

  @property
  def q_embeds(self):
    return self._q_embeds
 
  @property
  def f_nets(self):
    return self._f_nets

  @property
  def qembed_source_weights(self):
    q_weights = []
    for q_embed, _ in self._q_embeds:
      q_weights += q_embed.weights
    return q_weights

  @property
  def f_net_source_weights(self):
    f_weights = []
    for f_net, _ in self._f_nets:
      f_weights += f_net.weights
    return f_weights

  @property
  def qembed_target_weights(self):
    q_weights = []
    for _, q_embed in self._q_embeds:
      q_weights += q_embed.weights
    return q_weights

  @property
  def f_net_target_weights(self):
    f_weights = []
    for _, f_net in self._f_nets:
      f_weights += f_net.weights
    return f_weights

  @property
  def qembed_source_variables(self):
    vars_ = []
    for q_embed, _ in self._q_embeds:
      vars_ += q_embed.trainable_variables 
    return tuple(vars_)

  @property
  def f_net_source_variables(self):
    vars_ = []
    for f_net, _ in self._f_nets:
      vars_ += f_net.trainable_variables
    return tuple(vars_)

  @property 
  def f_net_std_variables(self):
    vars_ = []
    for f_net_std in self._f_nets_std:
      vars_ += f_net_std.trainable_variables
    return tuple(vars_)

  @property
  def qembed_target_variables(self):
    vars_ = []
    for _, q_embed in self._q_embeds:
      vars_ += q_embed.trainable_variables
    return tuple(vars_)
  
  @property
  def f_net_target_variables(self):
    vars_ = []
    for _, f_net in self._f_nets:
      vars_ += f_net.trainable_variables
    return tuple(vars_)

  @property
  def r_linear_source_weights(self):
    r_weights = []
    for r_net, _ in self._r_linear_nets:
      r_weights += r_linear_net.weights
    return r_weights

  @property
  def r_linear_target_weights(self):
    r_weights = []
    for _, r_net in self._r_linear_nets:
      r_weights += r_linear_net.weights
    return r_weights

  @property
  def r_linear_source_variables(self):
    vars_ = []
    for r_net, _ in self._r_linear_nets:
      vars_ += r_net.trainable_variables
    return tuple(vars_)

  @property
  def r_linear_target_variables(self):
    vars_ = []
    for _, r_net in self._r_linear_nets:
      vars_ += r_net.trainable_variables
    return tuple(vars_)

  @property
  def p_net(self):
    return self._p_net

  def p_fn(self, s):
    return self._p_net(self._p_embed(s))

  @property
  def p_weights(self):
    return self._p_net.weights

  @property
  def p_variables(self):
    return self._p_net.trainable_variables

  @property
  def p_embed(self):
    return self._p_embed

  @property
  def pembed_weights(self):
    return self._p_embed.weights

  @property
  def pembed_variables(self):
    return self._p_embed.trainable_variables

  @property
  def g_net_weights(self):
    g_weights = []
    for g_net in self._g_nets:
      g_weights += g_net.weights
    return g_weights

  @property
  def g_net_variables(self):
    vars_ = []
    for g_net in self._g_nets:
      vars_ += g_net.trainable_variables
    return tuple(vars_)

  @property
  def g1_net_weights(self):
    g1_weights = []
    for g1_net in self._g1_nets:
      g1_weights += g1_net.weights
    return g1_weights

  @property
  def g1_net_variables(self):
    vars_ = []
    for g1_net in self._g1_nets:
      vars_ += g1_net.trainable_variables
    return tuple(vars_)

def get_modules(model_params, action_spec, observation_spec):
  """Creates modules for SA embedding and S embedding."""
  model_params, n_q_fns = model_params
  if len(model_params) == 1:
    model_params = tuple([model_params[0]] * 2)
  elif len(model_params) < 2:
    raise ValueError('Bad model parameters %s.' % model_params)
  def q_embed_factory():
    return networks.SAEmbeddingNetwork(fc_layer_params=model_params[0])
    # return networks.RBFSANetwork(fc_layer_params=model_params[0])
  def p_embed_factory():
    return networks.SEmbeddingNetwork(fc_layer_params=model_params[1])
  def f_net_factory():
    return networks.SAEmbeddingNetwork(fc_layer_params=model_params[0])
    # return networks.RBFSANetwork(fc_layer_params=model_params[0])
  def g_net_factory():
    return networks.SEmbeddingNetwork(fc_layer_params=model_params[0])
    # return networks.RBFSNetwork(fc_layer_params=model_params[0])
  def g1_net_factory():
    return networks.SEmbeddingNetwork1(fc_layer_params=model_params[0], observation_spec=observation_spec)
  def q_net_factory():
    # return networks.CriticRBFNetworkDBN(model_params[0][-1], trainable=False)
    return networks.CriticLinearNetwork()
  def p_net_factory():
    return networks.ActorLinearNetwork(action_spec)
  def r_linear_factory():
    # return networks.RewardLinearNetwork() 
    return networks.RewardMLPNetwork()
  modules = utils.Flags(
      q_embed_factory=q_embed_factory,
      p_embed_factory=p_embed_factory,
      q_net_factory=q_net_factory,
      p_net_factory=p_net_factory,
      f_net_factory=f_net_factory,
      r_linear_factory=r_linear_factory,
      g_net_factory=g_net_factory,
      g1_net_factory=g1_net_factory,
      n_q_fns=n_q_fns,
      )
  return modules

class Config(agent.Config):

  def _get_modules(self):
    return get_modules(
        self._agent_flags.model_params,
        self._agent_flags.action_spec,
        self._agent_flags.observation_spec)
