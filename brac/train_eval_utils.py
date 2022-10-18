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

"""Utilities for training and evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from ebm_rl.brac import dataset
import tensorflow as tf
from mbbl.env import env_register
import gym

class ActionScaleWrapper(gym.Wrapper):
  def __init__(self, env):
    super(ActionScaleWrapper, self).__init__(env)
    self.env = env
        
  def step(self, action):
    # scaled_action = -1. + (action + 1.)
    action = np.clip(action, -1., 1.)
    next_state, reward, done, info = self.env.step(action)
    return next_state, reward, done, info


def eval_policy_episodes(env, policy, n_episodes):
  """Evaluates policy performance."""
  results = []
  for _ in range(n_episodes):
    time_step = env.reset()
    total_rewards = 0.0
    while not time_step.is_last().numpy()[0]:
      action = policy(time_step.observation)[0]
      time_step = env.step(action)
      total_rewards += time_step.reward
    results.append(total_rewards)
  results = np.array(results)
  return float(np.mean(results)), float(np.std(results))


def eval_policies(env, policies, n_episodes):
  results_episode_return = []
  infos = collections.OrderedDict()
  for name, policy in policies.items():
    mean, _ = eval_policy_episodes(env, policy, n_episodes)
    results_episode_return.append(mean)
    infos[name] = collections.OrderedDict()
    infos[name]['episode_mean'] = mean
  results = results_episode_return
  return results, infos


# TODO(wuyifan): external version for loading environments
def env_factory(env_name):
  #TODO: changes here
  from tf_agents.environments.suite_gym import wrap_env
  py_env, _ = env_register.make_env(env_name, 123, {'reset_type': 'gym'})
  # py_env = ActionScaleWrapper(wrap_env)
  py_env = wrap_env(py_env)
  # py_env = suite_mujoco.load(env_name)
  tf_env = tf_py_environment.TFPyEnvironment(py_env)
  return tf_env


def get_transition(time_step, next_time_step, action, next_action):
  return dataset.Transition(
      s1=time_step.observation,
      s2=next_time_step.observation,
      a1=action,
      a2=next_action,
      reward=next_time_step.reward,
      discount=next_time_step.discount,
      done=tf.cast(next_time_step.is_first(), dtype=tf.float32))


class DataCollector(object):
  """Class for collecting sequence of environment experience."""

  def __init__(self, tf_env, policy, data, eps_portion=0.):
    self._tf_env = tf_env
    self._policy = policy
    self._data = data
    self._eps_portion = eps_portion
    self._saved_action = None

  def collect_transition(self):
    """Collect single transition from environment."""
    time_step = self._tf_env.current_time_step()
    if self._saved_action is None:
      self._saved_action = self._policy(time_step.observation)[0]
    # Using a eps greedy version of sampling
    random_action = tf.convert_to_tensor(self._tf_env._env.envs[0].action_space.sample(), dtype=tf.float32)
    random_prob = tf.convert_to_tensor(np.random.rand(self._saved_action.shape[0]), dtype=tf.float32)
    action = self._saved_action * tf.cast(random_prob > tf.constant([self._eps_portion]), dtype=tf.float32) + \
            random_action * tf.cast(random_prob < tf.constant([self._eps_portion]), dtype=tf.float32)
    # tf.print(action, random_action, self._saved_action, random_prob, self._eps_portion)
    next_time_step = self._tf_env.step(action)
    next_action = self._policy(next_time_step.observation)[0]
    self._saved_action = next_action
    if not time_step.is_last()[0].numpy():
      transition = get_transition(time_step, next_time_step,
                                  action, next_action)
      self._data.add_transitions(transition)
      return 1
    else:
      return 0
