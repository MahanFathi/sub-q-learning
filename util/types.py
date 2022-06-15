from jax import numpy as jnp
import flax
import optax
from brax.training.types import *
from brax.training.acme import running_statistics
from brax.training.acme.types import NestedArray

class Transition(NamedTuple):
  """Container for a transition."""
  observation: NestedArray
  action: NestedArray
  reward: NestedArray
  sub_rewards: NestedArray
  discount: NestedArray
  next_observation: NestedArray
  extras: NestedArray = ()


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  policy_params: Params
  policy_optimizer_state: optax.OptState
  sub_policy_params: Params
  sub_policy_optimizer_state: optax.OptState
  sub_q_params: Params
  sub_q_optimizer_state: optax.OptState
  sub_target_q_params: Params
  gradient_steps: jnp.ndarray
  env_steps: jnp.ndarray
  alpha_params: Params
  alpha_optimizer_state: optax.OptState
  sub_alpha_params: Params
  sub_alpha_optimizer_state: optax.OptState
  normalizer_params: running_statistics.RunningStatisticsState
