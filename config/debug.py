import ml_collections
from config.defaults import get_config as get_default_config

def get_config():
    _C = get_default_config()

    _C.DEBUG = True

    # mocks 8 tpu devices on cpu
    _C.MOCK_TPU = False

    _C.WANDB = False

    _C.EBM.ARCH = "arch1"

    _C.ENV.ENV_NAME = "binary"
    _C.SAMPLER.SAMPLER_NAME = "binary"

    _C.TRAIN.EBM.BATCH_SIZE = 64
    _C.TRAIN.EBM.EVAL_BATCH_SIZE = 32

    _C.TRAIN.EBM.NUM_EPOCHS = 20
    _C.TRAIN.EBM.LOG_FREQUENCY = 20

    _C.TRAIN.EBM.LEARNING_RATE = 1e-6

    return _C
