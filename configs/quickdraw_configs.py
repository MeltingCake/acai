from configs.base import register_config

from util import HParams, teacher_noise_4, rotate_4


@register_config('quickdraw')
def quickdraw_default():
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

@register_config("quickdraw/noisy-rotate")
def quickdraw_noisy_rotate():
    hparams = quickdraw_default()
    hparams.add_hparam("augmentations", [[teacher_noise_4, rotate_4]])

    return hparams

@register_config("quickdraw/noisy")
def quickdraw_noisy():
    hparams = quickdraw_default()
    hparams.add_hparam("augmentations", [[teacher_noise_4]])

    return hparams

@register_config("quickdraw/rotate")
def quickdraw_rotate():
    hparams = quickdraw_default()
    hparams.add_hparam("augmentations", [[rotate_4]])

    return hparams

