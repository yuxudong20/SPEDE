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


@gin.configurable
class Agent(agent.Agent):
  """SAC Agent."""

  def __init__(
      self,
      target_entropy=None,
      ensemble_q_lambda=1.0,
      use_random_linear=False,
      **kwargs):
    self._ensemble_q_lambda = ensemble_q_lambda
    self._target_entropy = target_entropy
    self._use_random_linear = use_random_linear
    super(Agent, self).__init__(**kwargs)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)
    if self._target_entropy is None:
      self._target_entropy = - self._action_spec.shape[0]
    self._q_fns = self._agent_module.q_nets
    self._q_linear_fns = self._agent_module.q_linear_nets
    self._r_linear_fns = self._agent_module.r_linear_nets
    self._p_fn = self._agent_module.p_fn
    self._p_linear_fn = self._agent_module.p_linear_fn
    self._log_alpha = self._agent_module.log_alpha

  def _get_q_vars(self):
    return self._agent_module.q_source_variables

  def _get_r_linear_vars(self):
    return self._agent_module.r_linear_source_variables
 
  def _get_p_vars(self):
    return self._agent_module.p_variables

  def _get_q_weight_norm(self):
    weights = self._agent_module.q_source_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_r_weight_norm(self):
    weights = self._agent_module.r_linear_source_weights
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

  def ensemble_q(self, qs):
    # This is a little wierd
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
    s_embed = self._p_fn(s2)
    if self._use_random_linear:
      s_embed = self._p_linear_fn(s_embed)
    # s_embed = s_embed + tf.random.normal(s_embed.shape, mean=0.0, stddev=1.0)
    model_losses = []
    reward_losses = []
    for q_bundle, q_linear_bundle, r_linear_bundle in zip(self._q_fns, self._q_linear_fns, self._r_linear_fns):
      q_fn, q_fn_target = q_bundle
      q_linear_fn, q_linear_fn_target = q_linear_bundle
      r_linear_fn, r_linear_fn_target = r_linear_bundle
      sa_embed = q_fn(s1, a)
      sa_embed_target = q_fn_target(s1, a)
      if self._use_random_linear:
        sa_embed = q_linear_fn(sa_embed)
        sa_embed_target = q_linear_fn_target(sa_embed_target)
      # TODO: reduces mean here
      # log_density = tf.reduce_mean(tf.reduce_sum(sa_embed * s_embed, axis=-1))
      # model_losses.append(-log_density)

      # This is for transition loss
      logits = tf.reduce_sum(tf.expand_dims(sa_embed, axis=1) * tf.expand_dims(s_embed, axis=0), axis=-1)
      labels = tf.eye(s1.shape[0])
      ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      model_losses.append(tf.reduce_mean(ce_loss))

      # This is reward loss
      r_pred = r_linear_fn(sa_embed)
      r_loss = tf.nn.l2_loss(r - r_pred)
      reward_losses.append(tf.reduce_mean(r_loss))

    model_loss = tf.add_n(model_losses)
    reward_loss = tf.add_n(reward_losses)
    q_w_norm = self._get_q_weight_norm()
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays[0] * q_w_norm + self._weight_decays[0] * p_w_norm
    loss = model_loss + reward_loss + norm_loss
    
    info = collections.OrderedDict()
    info['model_loss'] = model_loss
    info['reward_loss'] = reward_loss
    info['q_norm'] = q_w_norm
    info['p_norm'] = p_w_norm
    info['r_mean'] = tf.reduce_mean(r)
    info['dsc_mean'] = tf.reduce_mean(dsc)

    return loss, info

  def _build_q_loss(self, batch):
    assert False
    s1 = batch['s1']
    s2 = batch['s2']
    a = batch['a1']
    r = batch['r']
    dsc = batch['dsc']
    _, a2, log_pi_a2 = self._p_fn(s2)
    q2_targets = []
    q1_preds = []
    for q_fn, q_fn_target in self._q_fns:
      q2_target_ = q_fn_target(s2, a2)
      q1_pred = q_fn(s1, a)
      q1_preds.append(q1_pred)
      q2_targets.append(q2_target_)
    q2_targets = tf.stack(q2_targets, axis=-1)
    q2_target = self._ensemble_q2_target(q2_targets)
    v2_target = q2_target - tf.exp(self._log_alpha) * log_pi_a2
    q1_target = tf.stop_gradient(r + dsc * self._discount * v2_target)
    q_losses = []
    for q1_pred in q1_preds:
      q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
      q_losses.append(q_loss_)
    q_loss = tf.add_n(q_losses)
    q_w_norm = self._get_q_weight_norm()
    norm_loss = self._weight_decays[0] * q_w_norm
    loss = q_loss + norm_loss

    info = collections.OrderedDict()
    info['q_loss'] = q_loss
    info['q_norm'] = q_w_norm
    info['r_mean'] = tf.reduce_mean(r)
    info['dsc_mean'] = tf.reduce_mean(dsc)
    info['q1_target_mean'] = tf.reduce_mean(q1_target)

    return loss, info

  def _build_p_loss(self, batch):
    assert False
    s = batch['s1']
    _, a, log_pi_a = self._p_fn(s)
    q1s = []
    for q_fn, _ in self._q_fns:
      q1_ = q_fn(s, a)
      q1s.append(q1_)
    q1s = tf.stack(q1s, axis=-1)
    q1 = self._ensemble_q1(q1s)
    p_loss = tf.reduce_mean(tf.exp(self._log_alpha) * log_pi_a - q1)
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays[1] * p_w_norm
    loss = p_loss + norm_loss

    info = collections.OrderedDict()
    info['p_loss'] = p_loss
    info['p_norm'] = p_w_norm

    return loss, info

  def _build_a_loss(self, batch):
    assert False
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

  def _build_optimizers(self):
    opts = self._optimizers
    if len(opts) == 1:
      opts = tuple([opts[0]] * 4)
    elif len(opts) < 4:
      raise ValueError('Bad optimizers %s.' % opts)
    #TODO: changes here
    self._weight_decays = (1e-5,)
    print('weight decay: ', self._weight_decays)
    self._q_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
    self._p_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
    self._a_optimizer = utils.get_optimizer(opts[2][0])(lr=opts[2][1])
    self._model_optimizer = utils.get_optimizer(opts[3][0])(lr=opts[3][1])
    if len(self._weight_decays) == 1:
      self._weight_decays = tuple([self._weight_decays[0]] * 4)

  @tf.function
  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    # if tf.equal(self._global_step % self._update_freq, 0):
    #   source_vars, target_vars = self._get_source_target_vars()
    #   self._update_target_fns(source_vars, target_vars)
    # q_info = self._optimize_q(batch)
    # p_info = self._optimize_p(batch)
    # a_info = self._optimize_a(batch)
    # info.update(p_info)
    # info.update(q_info)
    # info.update(a_info)
    if tf.equal(self._global_step % self._update_freq, 0):
      source_vars, target_vars = self._get_source_target_vars()
      utils.soft_variables_update(source_vars, target_vars)
    model_info = self._optimize_model(batch)
    info.update(model_info)
    return info

  def _optimize_q(self, batch):
    assert False
    vars_ = self._q_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_q_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._q_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_p(self, batch):
    assert False
    vars_ = self._p_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_p_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._p_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_a(self, batch):
    assert False
    vars_ = [self._log_alpha]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_a_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._a_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_model(self, batch):
    vars_ = self._p_vars + self._q_vars + self._r_linear_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_model_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._model_optimizer.apply_gradients(grads_and_vars)
    return info

  def _build_test_policies(self):
    policy = policies.DeterministicSoftPolicy(
        a_network=self._agent_module.p_net)
    self._test_policies['main'] = policy

  def _build_online_policy(self):
    return policies.RandomSoftPolicy(
        a_network=self._agent_module.p_net,
        )

  def _init_vars(self, batch):
    # self._build_q_loss(batch)
    # self._build_p_loss(batch)
    self._build_model_loss(batch)
    self._q_vars = self._get_q_vars()
    self._p_vars = self._get_p_vars()
    self._r_linear_vars = self._get_r_linear_vars()

  def _build_checkpointer(self):
    state_ckpt = tf.train.Checkpoint(
        policy=self._agent_module.p_net,
        agent=self._agent_module)
    q_embed_ckpt = tf.train.Checkpoint(
        q_embeds=self._agent_module.q_nets)
    p_embed_ckpt = tf.train.Checkpoint(
        p_embed=self._agent_module.p_net)
    return dict(state=state_ckpt, q_embed=q_embed_ckpt, p_embed=p_embed_ckpt)

  def save(self, ckpt_name):
    self._checkpointer['state'].write(ckpt_name)
    self._checkpointer['q_embed'].write(ckpt_name + '_q_embed')
    self._checkpointer['p_embed'].write(ckpt_name + '_p_embed')

  def restore(self, ckpt_name):
    self._checkpointer['state'].restore(ckpt_name)


class AgentModule(agent.AgentModule):
  """Tensorflow modules for SAC agent."""

  def _build_modules(self):
    self._q_nets = []
    self._q_linear_nets = []
    self._r_linear_nets = []
    n_q_fns = self._modules.n_q_fns
    for _ in range(n_q_fns):
      self._q_nets.append(
          [self._modules.q_net_factory(),  # Learned Q-value.
           self._modules.q_net_factory(),]  # Target Q-value.
          )
      self._q_linear_nets.append(
          [self._modules.q_linear_factory(),  # Learned Q-value.
          self._modules.q_linear_factory(),]  # Target Q-value.
          )
      self._r_linear_nets.append(
          [self._modules.r_linear_factory(),  # Learned Q-value.
          self._modules.r_linear_factory(),]  # Target Q-value.
          )
    self._p_net = self._modules.p_net_factory()
    self._p_linear_net = self._modules.p_linear_factory()
    self._log_alpha = tf.Variable(0.0)

  @property
  def log_alpha(self):
    return self._log_alpha

  @property
  def q_nets(self):
    return self._q_nets
  
  @property
  def q_linear_nets(self):
    return self._q_linear_nets
  
  @property
  def r_linear_nets(self):
    return self._r_linear_nets

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
  
  @property
  def p_linear_net(self):
    return self._p_linear_net

  def p_fn(self, s):
    return self._p_net(s)
  
  def p_linear_fn(self, s):
    return self._p_linear_net(s)

  @property
  def p_weights(self):
    return self._p_net.weights

  @property
  def p_variables(self):
    return self._p_net.trainable_variables


def get_modules(model_params, action_spec):
  """Creates modules for SA embedding and S embedding."""
  model_params, n_q_fns = model_params
  if len(model_params) == 1:
    model_params = tuple([model_params[0]] * 2)
  elif len(model_params) < 2:
    raise ValueError('Bad model parameters %s.' % model_params)
  def q_net_factory():
    return networks.SAEmbeddingNetwork(
        fc_layer_params=model_params[0])
  def p_net_factory():
    return networks.SEmbeddingNetwork(
        fc_layer_params=model_params[1])
  def q_linear_factory():
    return networks.LinearNetwork(
        fc_layer_params=model_params[0][-1])
  def p_linear_factory():
    return networks.LinearNetwork(
        fc_layer_params=model_params[1][-1])
  def r_linear_factory():
    return networks.CriticLinearNetwork()

  modules = utils.Flags(
      q_net_factory=q_net_factory,
      p_net_factory=p_net_factory,
      q_linear_factory=q_linear_factory,
      p_linear_factory=p_linear_factory,
      r_linear_factory=r_linear_factory,
      n_q_fns=n_q_fns,
      )
  return modules


class Config(agent.Config):

  def _get_modules(self):
    return get_modules(
        self._agent_flags.model_params,
        self._agent_flags.action_spec)
