from configs.base import register_config

from util import HParams, teacher_noise_4, rotate_4

@register_config('sketchy')
def sketchy_default():
    return HParams(
        # ----- Dataset Parameters ----- #
        batch_size=256,
        split="",

        # ----- Loading Parameters ----- #
        cycle_length=None,
        num_parallel_calls=None,
        block_length=1,
        buff_size=2,
        shuffle=True,
    )