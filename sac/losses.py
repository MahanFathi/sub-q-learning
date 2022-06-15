# Copyright 2022 The Brax Authors.
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

"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""
from typing import Any

from brax.training import types
from brax.training.agents.sac import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(sac_network: sac_networks.SACNetworks, reward_scaling: float,
                discounting: float, action_size: int):
  """Creates the SAC losses."""

  target_entropy = -0.5 * action_size
  policy_network = sac_network.policy_network
  q_network = sac_network.q_network
  parametric_action_distribution = sac_network.parametric_action_distribution

  def _critic_loss(q_params: Params, policy_params: Params,
                  normalizer_params: Any, target_q_params: Params,
                  alpha: jnp.ndarray, transitions: Transition,
                  reward: Transition.reward,
                  key: PRNGKey) -> jnp.ndarray:
    q_old_action = q_network.apply(normalizer_params, q_params,
                                   transitions.observation, transitions.action)
    next_dist_params = policy_network.apply(normalizer_params, policy_params,
                                            transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key)
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(normalizer_params, target_q_params,
                             transitions.next_observation, next_action)
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
    target_q = jax.lax.stop_gradient(reward * reward_scaling +
                                     transitions.discount * discounting *
                                     next_v)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  def _actor_loss(policy_params: Params, normalizer_params: Any,
                 q_params: Params, alpha: jnp.ndarray, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)
    q_action = q_network.apply(normalizer_params, q_params,
                               transitions.observation, action)
    min_q = jnp.min(q_action, axis=-1)
    actor_loss = alpha * log_prob - min_q
    return jnp.mean(actor_loss)

  def alpha_loss(log_alpha: jnp.ndarray, policy_params: Params,
                 normalizer_params: Any, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(alpha_loss)

  def sub_q_loss(sub_q_params: Params, sub_policy_params: Params,
                  normalizer_params: Any, sub_target_q_params: Params,
                  alpha: jnp.ndarray, transitions: Transition,
                  key: PRNGKey) -> jnp.ndarray:

    keydict = {k: jax.random.fold_in(key, hash(k)) for k in sub_q_params.keys()}
    losses = jax.tree_multimap(
      lambda key, q_params, policy_params, target_q_params, reward: _critic_loss(
        q_params, policy_params, normalizer_params, target_q_params, alpha, transitions, reward, key),
      keydict, sub_q_params, sub_policy_params, sub_target_q_params, transitions.sub_rewards,
    )
    sub_q_loss = jnp.sum(jax.flatten_util.ravel_pytree(losses)[0])
    return sub_q_loss

  def sub_policy_loss(sub_policy_params: Params, normalizer_params: Any,
                 sub_q_params: Params, alpha: jnp.ndarray, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    keydict = {k: jax.random.fold_in(key, hash(k)) for k in sub_q_params.keys()}
    losses = jax.tree_multimap(
      lambda key, policy_params, q_params: _actor_loss(
        policy_params, normalizer_params, q_params, alpha, transitions, key),
      keydict, sub_policy_params, sub_q_params,
    )
    sub_policy_loss = jnp.sum(jax.flatten_util.ravel_pytree(losses)[0])
    return sub_policy_loss

  def policy_loss(policy_params: Params, normalizer_params: Any,
                 sub_q_params: Params, alpha: jnp.ndarray, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)

    dummy_dict = {k: 0. for k in sub_q_params.keys()}
    sub_q_action = jax.tree_multimap(
      lambda unused_1st_arg, q_params: q_network.apply(
      normalizer_params, q_params, transitions.observation, action),
      dummy_dict, sub_q_params,
    )
    min_sub_q = jax.tree_map(lambda q: jnp.min(q, axis=-1), sub_q_action)
    actor_loss = alpha * log_prob - jnp.sum(jax.flatten_util.ravel_pytree(min_sub_q)[0]) # TODO: handle linear weights from reward_dict
    return jnp.mean(actor_loss)

  return alpha_loss, sub_q_loss, sub_policy_loss, policy_loss
