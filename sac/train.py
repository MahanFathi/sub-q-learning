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

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import time
from typing import Any, Callable, Optional, Tuple, Mapping

from absl import logging
from brax import envs
from brax.envs import wrappers
from brax.io import model
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from sac import losses as sac_losses
from sac import networks as sac_networks
from sac import acting
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
from ml_collections import FrozenConfigDict
from util import logger
from util.types import Transition, TrainingState

Metrics = types.Metrics
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


def _unpmap(v):
  return jax.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey, obs_size: int, local_devices_to_use: int,
    sac_network: sac_networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    sub_q_optimizer: optax.GradientTransformation,
    sub_policy_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    reward_dict: Mapping[str, float],
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_policy, key_sub_policy, key_sub_q = jax.random.split(key, 3)
  keydict_sub_policy = {k: jax.random.fold_in(key_sub_policy, hash(k))
                        for k in reward_dict.keys()}
  keydict_sub_q = {k: jax.random.fold_in(key_sub_q, hash(k))
                        for k in reward_dict.keys()}

  log_alpha = jnp.asarray(0., dtype=jnp.float32)
  alpha_optimizer_state = alpha_optimizer.init(log_alpha)

  policy_params = sac_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)

  sub_policy_params = jax.tree_map(sac_network.policy_network.init, keydict_sub_policy)
  sub_policy_optimizer_state = sub_policy_optimizer.init(sub_policy_params)

  sub_q_params = jax.tree_map(sac_network.q_network.init, keydict_sub_q)
  sub_q_optimizer_state = sub_q_optimizer.init(sub_q_params)

  normalizer_params = running_statistics.init_state(
      specs.Array((obs_size,), jnp.float32))

  training_state = TrainingState(
      policy_params=policy_params,
      policy_optimizer_state=policy_optimizer_state,
      sub_policy_params=sub_policy_params,
      sub_policy_optimizer_state=sub_policy_optimizer_state,
      sub_q_params=sub_q_params,
      sub_q_optimizer_state=sub_q_optimizer_state,
      sub_target_q_params=sub_q_params,
      gradient_steps=jnp.zeros(()),
      env_steps=jnp.zeros(()),
      alpha_optimizer_state=alpha_optimizer_state,
      alpha_params=log_alpha,
      normalizer_params=normalizer_params)
  return jax.device_put_replicated(training_state,
                                   jax.local_devices()[:local_devices_to_use])


def train(
    cfg: FrozenConfigDict,
    environment: envs.Env,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
):

  """SAC training."""

  # export config
  seed = cfg.SEED
  num_timesteps = cfg.TRAIN.NUM_TIMESTEPS
  episode_length = cfg.TRAIN.EPISODE_LENGTH
  action_repeat = cfg.TRAIN.ACTION_REPEAT
  num_envs = cfg.TRAIN.NUM_ENVS
  num_eval_envs = cfg.TRAIN.NUM_EVAL_ENVS
  learning_rate = cfg.TRAIN.LEARNING_RATE
  discounting = cfg.TRAIN.DISCOUNTING
  batch_size = cfg.TRAIN.BATCH_SIZE
  num_evals = cfg.TRAIN.NUM_EVALS
  normalize_observations = cfg.TRAIN.NORMALIZE_OBSERVATIONS
  max_devices_per_host = cfg.TRAIN.MAX_DEVICES_PER_HOST
  reward_scaling = cfg.TRAIN.REWARD_SCALING
  tau = cfg.TRAIN.TAU
  min_replay_size = cfg.TRAIN.MIN_REPLAY_SIZE
  max_replay_size = cfg.TRAIN.MAX_REPLAY_SIZE
  grad_updates_per_step = cfg.TRAIN.GRAD_UPDATES_PER_STEP
  network_factory: types.NetworkFactory[sac_networks.SACNetworks] = sac_networks.make_sac_networks
  reward_dict = dict(cfg.ENV.REWARD_DICT[cfg.ENV.ENV_NAME])

  # process bookkeeping
  process_id = jax.process_index()
  local_devices_to_use = jax.local_device_count()
  if max_devices_per_host is not None:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  device_count = local_devices_to_use * jax.process_count()
  logging.info('local_device_count: %s; total_device_count: %s',
               local_devices_to_use, device_count)

  if min_replay_size >= num_timesteps:
    raise ValueError(
        'No train will happen because min_replay_size >= num_timesteps')

  if max_replay_size is None:
    max_replay_size = num_timesteps

  # The number of environment steps executed for every `actor_step()` call.
  env_steps_per_actor_step = action_repeat * num_envs
  # equals to ceil(min_replay_size / env_steps_per_actor_step)
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of run_one_sac_epoch calls per run_sac_training.
  # equals to
  # ceil(num_timesteps - num_prefill_env_steps /
  #      (num_evals_after_init * env_steps_per_actor_step))
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps) //
      (num_evals_after_init * env_steps_per_actor_step))

  assert num_envs % device_count == 0
  env = environment
  env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  env = wrappers.VmapWrapper(env)
  env = wrappers.AutoResetWrapper(env)

  obs_size = env.observation_size
  action_size = env.action_size

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  sac_network = network_factory(
      observation_size=obs_size,
      action_size=action_size,
      preprocess_observations_fn=normalize_fn)
  make_policy = sac_networks.make_inference_fn(sac_network)

  alpha_optimizer = optax.adam(learning_rate=3e-4)
  sub_q_optimizer = optax.adam(learning_rate=learning_rate)
  sub_policy_optimizer = optax.adam(learning_rate=learning_rate)
  policy_optimizer = optax.adam(learning_rate=learning_rate)

  dummy_obs = jnp.zeros((obs_size,))
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = Transition(
      observation=dummy_obs,
      action=dummy_action,
      reward=0.,
      sub_rewards=reward_dict,
      discount=0.,
      next_observation=dummy_obs,
      extras={
          'state_extras': {
              'truncation': 0.
          },
          'policy_extras': {}
      })
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count)

  alpha_loss, sub_q_loss, sub_policy_loss, policy_loss = sac_losses.make_losses(
      sac_network=sac_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size)
  alpha_update = gradients.gradient_update_fn(
      alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
  sub_q_update = gradients.gradient_update_fn(
      sub_q_loss, sub_q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
  sub_policy_update = gradients.gradient_update_fn(
      sub_policy_loss, sub_policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
  policy_update = gradients.gradient_update_fn(
      policy_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey],
      transitions: Transition) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry

    key, key_alpha, key_sub_q, key_sub_policy, key_policy = jax.random.split(key, 5)

    alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
        training_state.alpha_params,
        training_state.policy_params,
        training_state.normalizer_params,
        transitions,
        key_alpha,
        optimizer_state=training_state.alpha_optimizer_state)
    alpha = jnp.exp(training_state.alpha_params)
    sub_q_loss, sub_q_params, sub_q_optimizer_state = sub_q_update(
        training_state.sub_q_params,
        training_state.sub_policy_params,
        training_state.normalizer_params,
        training_state.sub_target_q_params,
        alpha,
        transitions,
        key_sub_q,
        optimizer_state=training_state.sub_q_optimizer_state)
    sub_policy_loss, sub_policy_params, sub_policy_optimizer_state = sub_policy_update(
        training_state.sub_policy_params,
        training_state.normalizer_params,
        training_state.sub_q_params,
        alpha,
        transitions,
        key_sub_policy,
        optimizer_state=training_state.sub_policy_optimizer_state)
    policy_loss, policy_params, policy_optimizer_state = policy_update(
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.sub_q_params,
        alpha,
        transitions,
        key_policy,
        optimizer_state=training_state.policy_optimizer_state)

    new_sub_target_q_params = jax.tree_map(lambda x, y: x * (1 - tau) + y * tau,
                                       training_state.sub_target_q_params, sub_q_params)

    metrics = {
        'alpha_loss': alpha_loss,
        'sub_q_loss': sub_q_loss,
        'sub_policy_loss': sub_policy_loss,
        'policy_loss': policy_loss,
        'alpha': jnp.exp(alpha_params),
    }

    new_training_state = TrainingState(
        sub_q_params=sub_q_params,
        sub_q_optimizer_state=sub_q_optimizer_state,
        sub_policy_params=sub_policy_params,
        sub_policy_optimizer_state=sub_policy_optimizer_state,
        policy_params=policy_params,
        policy_optimizer_state=policy_optimizer_state,
        sub_target_q_params=new_sub_target_q_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=alpha_params,
        normalizer_params=training_state.normalizer_params)
    return (new_training_state, key), metrics

  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params, env_state: envs.State,
      buffer_state: ReplayBufferState, key: PRNGKey
  ) -> Tuple[running_statistics.RunningStatisticsState, envs.State,
             ReplayBufferState]:
    policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = acting.actor_step(
        env, env_state, policy, key, reward_dict, extra_fields=('truncation',))

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME)

    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, env_state, buffer_state

  def training_step(
      training_state: TrainingState, env_state: envs.State,
      buffer_state: ReplayBufferState, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    experience_key, training_key = jax.random.split(key)
    normalizer_params, env_state, buffer_state = get_experience(
        training_state.normalizer_params, training_state.policy_params,
        env_state, buffer_state, experience_key)
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_steps_per_actor_step)

    buffer_state, transitions = replay_buffer.sample(buffer_state)
    # Change the front dimension of transitions so 'update_step' is called
    # grad_updates_per_step times by the scan.
    transitions = jax.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions)
    (training_state, _), metrics = jax.lax.scan(sgd_step,
                                                (training_state, training_key),
                                                transitions)

    metrics['buffer_current_size'] = buffer_state.current_size
    metrics['buffer_current_position'] = buffer_state.current_position
    return training_state, env_state, buffer_state, metrics

  def prefill_replay_buffer(
      training_state: TrainingState, env_state: envs.State,
      buffer_state: ReplayBufferState, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, env_state, buffer_state = get_experience(
          training_state.normalizer_params, training_state.policy_params,
          env_state, buffer_state, key)
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          env_steps=training_state.env_steps + env_steps_per_actor_step)
      return (new_training_state, env_state, buffer_state, new_key), ()

    return jax.lax.scan(
        f, (training_state, env_state, buffer_state, key), (),
        length=num_prefill_actor_steps)[0]

  prefill_replay_buffer = jax.pmap(
      prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

  def training_epoch(
      training_state: TrainingState, env_state: envs.State,
      buffer_state: ReplayBufferState, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

    def f(carry, unused_t):
      ts, es, bs, k = carry
      k, new_key = jax.random.split(k)
      ts, es, bs, metrics = training_step(ts, es, bs, k)
      return (ts, es, bs, new_key), metrics

    (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
        f, (training_state, env_state, buffer_state, key), (),
        length=num_training_steps_per_epoch)
    metrics = jax.tree_map(jnp.mean, metrics)
    return training_state, env_state, buffer_state, metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State,
      buffer_state: ReplayBufferState, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    (training_state, env_state, buffer_state,
     metrics) = training_epoch(training_state, env_state, buffer_state, key)
    # This line also guarantees the values are ready.
    metrics = jax.tree_map(jnp.mean, metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (env_steps_per_actor_step *
           num_training_steps_per_epoch) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, env_state, buffer_state, metrics

  global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
  local_key = jax.random.fold_in(local_key, process_id)

  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_size,
      local_devices_to_use=local_devices_to_use,
      sac_network=sac_network,
      alpha_optimizer=alpha_optimizer,
      sub_q_optimizer=sub_q_optimizer,
      sub_policy_optimizer=sub_policy_optimizer,
      policy_optimizer=policy_optimizer,
      reward_dict=reward_dict,
  )
  del global_key

  local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

  # Env init
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(env_keys,
                         (local_devices_to_use, -1) + env_keys.shape[1:])
  env_state = jax.pmap(env.reset)(env_keys)

  # Replay buffer init
  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use))

  evaluator = acting.Evaluator(
      env,
      make_policy,
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
      reward_dict=reward_dict,
  )

  # Run initial eval
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.policy_params)),
        training_metrics={})
    logging.info(metrics)
    progress_fn(0, metrics)

  # Create and initialize the replay buffer.
  t = time.time()
  prefill_key, local_key = jax.random.split(local_key)
  prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  training_state, env_state, buffer_state, _ = prefill_replay_buffer(
      training_state, env_state, buffer_state, prefill_keys)

  replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state))
  logging.info('replay size after prefill %s', replay_size)
  assert replay_size >= min_replay_size
  training_walltime = time.time() - t

  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)

    # Optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state, env_state, buffer_state,
     training_metrics) = training_epoch_with_timing(training_state, env_state,
                                                    buffer_state, epoch_keys)
    current_step = int(_unpmap(training_state.env_steps))

    # Eval and logging
    if process_id == 0:
      # Save current policy.
      params = _unpmap(
          (training_state.normalizer_params, training_state.policy_params))
      path = f'{logger.get_logdir_path(cfg)}/{current_step}.pkl'
      model.save_params(path, params)

      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params)),
          training_metrics)
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  assert total_steps >= num_timesteps

  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params))

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
