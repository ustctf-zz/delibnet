# coding=utf-8

from .zhen_wmt17 import transformer_delib_big_v2
from tensor2tensor.utils import registry

@registry.register_hparams
def transformer_delib_big_rl_basic():
    """HParams for transfomer big delibnet (RL Setting) model on WMT."""
    hparams = transformer_delib_big_v2()
    hparams.add_hparam("rl", True)
    hparams.add_hparam("delta_reward", False)
    hparams.add_hparam("rl_beam_size", 1)
    hparams.add_hparam("rl_beam_alpha", 0.6)
    hparams.sampling_method = "random"
    hparams.label_smoothing = 0
    return hparams

