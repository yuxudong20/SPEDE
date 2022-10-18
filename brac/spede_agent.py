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
      target_entropy=None,
      ensemble_q_lambda=1.0,
      update_model_freq=1, 
      random_feature=True,
      model_update_tau=0.001,
      buffer_size=100000,
      model_learning_rate=0.0001,
      et=True,
      **kwargs):
    self._ensemble_q_lambda = ensemble_q_lambda
    self._target_entropy = target_entropy
    self._q_embed_ckpt_file = q_embed_ckpt_file
    self._p_embed_ckpt_file = p_embed_ckpt_file
    self._update_model_freq = update_model_freq 
    self._use_rep = use_rep
    self._eta = tf.Variable(1., trainable=False)
    self._noise_type = 'gaussian'
    self._model_idx = tf.Variable(-1, trainable=False)
    self._random_feature = random_feature
    self._model_learning_rate = model_learning_rate
    self._model_update_tau = tf.Variable(model_update_tau, trainable=False)
    self._buffer_size = buffer_size
    self._env = env
    self._et = et
    super(Agent, self).__init__(**kwargs)

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
    self._log_alpha = self._agent_module.log_alpha

  def _get_q_vars(self):
    return self._agent_module.q_source_variables

  def _get_q_target_vars(self):
    return self._agent_module.q_target_variables

  def _get_p_vars(self):
    return self._agent_module.p_variables + self._agent_module.pembed_variables

  def _get_g_vars(self):
    return self._agent_module.g_net_variables
  
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

  def _get_qembed_vars(self):
    return self._agent_module.qembed_source_variables

  def _get_pembed_vars(self):
    return self._agent_module.pembed_variables

  def _get_g_net_vars(self):
    return self._agent_module.g_net_variables

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
    dsc = batch['dsc']
    model_losses = []
    reward_losses = []
    for f_bundle, q_linear_bundle, r_linear_bundle, g_fn in zip(self._f_net_fns, self._q_fns, self._r_linear_fns, self._g_net_fns):
      f_fn, f_fn_target = f_bundle
      q_linear_fn, q_linear_fn_target = q_linear_bundle
      r_linear_fn, r_linear_fn_target = r_linear_bundle
      s_embed = g_fn(s2)
      sa_embed = f_fn(s1, a)
      sa_embed_reward = sa_embed
      sa_embed_transition = sa_embed

      # This is transition loss
      logits = tf.reduce_mean(-0.5 * tf.square(tf.expand_dims(sa_embed, axis=1) - tf.expand_dims(s_embed, axis=0)), axis=-1)
      pos = tf.reduce_mean(-0.5 * tf.square(sa_embed - s_embed), axis=-1)
      neg = tf.math.reduce_logsumexp(logits, axis=-1)
      gaussian_loss = - pos + neg
      model_losses.append(tf.reduce_mean(gaussian_loss))

      # This is reward loss
      r_pred = r_linear_fn(sa_embed_reward)
      r_loss = tf.square(r - r_pred)
      reward_losses.append(tf.reduce_mean(r_loss))
    
    model_loss = tf.add_n(model_losses)
    reward_loss = tf.add_n(reward_losses) 
    f_w_norm = self._get_f_net_weight_norm()
    g_w_norm = self._get_g_net_weight_norm()
    norm_loss = self._weight_decays[0] * f_w_norm + self._weight_decays[0] * g_w_norm
    loss = (model_loss + reward_loss + norm_loss) 

    info = collections.OrderedDict()
    info['model_loss'] = model_loss
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
    _, a, log_pi_a = self._p_fn(s)
    q1s = []
    for q_bundle, qembed_bundle in zip(self._q_fns, self._qembed_fns):
      q_fn, _ = q_bundle
      qembed_fn, qembed_fn_target = qembed_bundle
      q1_ = q_fn(qembed_fn(s, a))
      q1s.append(q1_)
    q1s = tf.stack(q1s, axis=-1)
    q1 = self._ensemble_q1(q1s)
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
    self._p_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
    self._a_optimizer = utils.get_optimizer(opts[2][0])(lr=opts[2][1])
    self._model_optimizer = utils.get_optimizer(opts[3][0])(lr=self._model_learning_rate)
    if len(self._weight_decays) == 1:
      self._weight_decays = tuple([self._weight_decays[0]] * 4)

  def train_step(self):
    train_batch = self._get_train_batch()
    info = self._optimize_step(train_batch)
    for _iter in range(2):
      train_batch = self._get_train_batch()
      self._optimize_q_alone(train_batch)
      self._optimize_p_alone(train_batch)
    for key, val in info.items():
      self._train_info[key] = val.numpy()
    
  @tf.function
  def _optimize_q_alone(self, batch):
    q_info = self._optimize_q(batch)
    return q_info

  @tf.function
  def _optimize_a_alone(self, batch):
    a_info = self._optimize_a(batch)
    return a_info

  @tf.function
  def _optimize_p_alone(self, batch):
    p_info = self._optimize_p(batch)
    return p_info
  
  @tf.function
  def _optimize_model_alone(self, batch):
    model_info = self._optimize_model(batch)
    return model_info
  
  def _get_train_batch(self):
    """Samples and constructs batch of transitions."""
    size = self._buffer_size
    if self._train_data.size > size:
      batch_indices = np.random.randint(self._train_data.size-size, self._train_data.size, self._batch_size)
    else:
      batch_indices = np.random.choice(self._train_data.size, self._batch_size)
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

  @tf.function
  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    self._eta.assign(self._eta * tf.exp(-0.00001))

    if tf.equal(self._global_step % self._update_freq, 0):
      source_vars, target_vars = self._get_source_target_vars()
      self._update_target_fns(source_vars, target_vars)
      source_vars, target_vars = self._get_source_target_embed_vars()
      self._update_target_fns(source_vars, target_vars)
    

    if tf.equal(self._global_step % self._update_model_freq, 0):
      f_net_source_vars, f_net_target_vars = self._get_source_target_f_net_vars()
      qembed_source_vars, qembed_target_vars = self._get_source_target_embed_vars()
      utils.soft_variables_update(f_net_source_vars, qembed_source_vars, tau=self._model_update_tau)
    
    q_info = self._optimize_q(batch)
    p_info = self._optimize_p(batch)
    a_info = self._optimize_a(batch)
    if tf.less(self._global_step, self._train_model_steps) and tf.equal(self._global_step % self._update_model_freq, 0):
      model_info = self._optimize_model(batch)
    else:
      model_info = collections.OrderedDict()
      model_info['model_loss'] = 0.0
      model_info['reward_loss'] = 0.0
      model_info['f_net_norm'] = 0.0
      model_info['g_norm'] = 0.0
    info.update(q_info)
    info.update(p_info)
    info.update(a_info)
    info.update(model_info)

    return info
  
  def _optimize_q(self, batch):
    vars_ = self._q_vars
    vars_ = tuple([vars_[0]] + [vars_[1]] + [vars_[4]] + [vars_[5]] + \
            [vars_[6]] + [vars_[7]] + [vars_[10]] + [vars_[11]])
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
    # vars_ = self._f_net_vars + self._r_linear_vars
    vars_ = self._f_net_vars + self._r_linear_vars + self._g_net_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_model_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._model_optimizer.apply_gradients(grads_and_vars)
    
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
      weight.assign_add(random_weights)
    return info

  def _build_test_policies(self):
    policy = policies.DecoupleDeterministicSoftPolicy(
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
    self._r_linear_vars = self._get_r_linear_vars()
    self._f_net_vars = self._get_f_net_vars()
    
    f_net_source_vars, f_net_target_vars = self._get_source_target_f_net_vars()
    qembed_source_vars, qembed_target_vars = self._get_source_target_embed_vars()
    utils.soft_variables_update(f_net_source_vars, qembed_source_vars, tau=1.0)
    source_vars, target_vars = self._get_source_target_vars()
    source_vars = tuple([source_vars[2]] + [source_vars[3]] + [source_vars[8]] + [source_vars[9]])
    target_vars = tuple([target_vars[2]] + [target_vars[3]] + [target_vars[8]] + [target_vars[9]])
    utils.soft_variables_update(source_vars, target_vars, tau=1.0)
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
    f_net_ckpt = tf.train.Checkpoint(
        f_nets=self._agent_module.f_nets)
    return dict(state=state_ckpt, q_embed=q_embed_ckpt, p_embed=p_embed_ckpt, g_net=g_net_ckpt, f_net=f_net_ckpt)
 
  def save(self, ckpt_name):
    self._checkpointer['state'].write(ckpt_name)
    self._checkpointer['q_embed'].write(ckpt_name + '_q_embed')
    self._checkpointer['p_embed'].write(ckpt_name + '_p_embed') 
    self._checkpointer['g_net'].write(ckpt_name + '_g_net') 
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
    self._q_nets = []
    self._r_linear_nets = []
    self._f_nets = []
    self._g_nets = []
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

def get_modules(model_params, action_spec):
  """Creates modules for SA embedding and S embedding."""
  model_params, n_q_fns = model_params
  if len(model_params) == 1:
    model_params = tuple([model_params[0]] * 2)
  elif len(model_params) < 2:
    raise ValueError('Bad model parameters %s.' % model_params)
  def q_embed_factory():
    return networks.SAEmbeddingNetwork(fc_layer_params=model_params[0])
  def p_embed_factory():
    return networks.SEmbeddingNetwork(fc_layer_params=model_params[1])
  def f_net_factory():
    return networks.SAEmbeddingNetwork(fc_layer_params=model_params[0])
  def g_net_factory():
    return networks.SEmbeddingNetwork(fc_layer_params=model_params[0])
  def q_net_factory():
    return networks.CriticRBFNetworkDBN(model_params[0][-1], trainable=False)
  def p_net_factory():
    return networks.ActorLinearNetwork(action_spec)
  def r_linear_factory():
    return networks.RewardMLPNetwork()
  modules = utils.Flags(
      q_embed_factory=q_embed_factory,
      p_embed_factory=p_embed_factory,
      q_net_factory=q_net_factory,
      p_net_factory=p_net_factory,
      f_net_factory=f_net_factory,
      r_linear_factory=r_linear_factory,
      g_net_factory=g_net_factory,
      n_q_fns=n_q_fns,
      )
  return modules

class Config(agent.Config):

  def _get_modules(self):
    return get_modules(
        self._agent_flags.model_params,
        self._agent_flags.action_spec)
