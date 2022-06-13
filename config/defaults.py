import ml_collections

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = ml_collections.ConfigDict()
_C.EXP_NAME = ""
_C.SEED = 0
_C.WANDB = True
_C.DEBUG = False
_C.MOCK_TPU = False

# ---------------------------------------------------------------------------- #
# Environment
# ---------------------------------------------------------------------------- #
_C.ENV = ml_collections.ConfigDict()
_C.ENV.ENV_NAME = "walker2d"

# ---------------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------------- #
_C.TRAIN = ml_collections.ConfigDict()
_C.TRAIN.NUM_TIMESTEPS = 50_000_000
_C.TRAIN.EPISODE_LENGTH = 1000
_C.TRAIN.ACTION_REPEAT = 1
_C.TRAIN.NUM_ENVS = 32
_C.TRAIN.NUM_EVAL_ENVS = 32
_C.TRAIN.LEARNING_RATE = 1e-4
_C.TRAIN.DISCOUNTING = 0.98
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.NUM_EVALS = 10
_C.TRAIN.NORMALIZE_OBSERVATIONS = True
_C.TRAIN.MAX_DEVICES_PER_HOST = None # TODO: interferes w/ type int?
_C.TRAIN.REWARD_SCALING = 0.1
_C.TRAIN.TAU = 0.005
_C.TRAIN.MIN_REPLAY_SIZE = 8192
_C.TRAIN.MAX_REPLAY_SIZE = 1048576
_C.TRAIN.GRAD_UPDATES_PER_STEP = 1

# ---------------------------------------------------------------------------- #
# DEFAULT CONFIG
# ---------------------------------------------------------------------------- #
def get_config():
    return _C
