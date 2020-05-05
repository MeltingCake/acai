from configs.base import register_config
from util import HParams, rotate_4, teacher_noise_4


@register_config('fs_omniglot')
def fs_omniglot_default():
    return HParams(
        # ----- Dataset Parameters ----- #
        split="",
        mode="",  # episodic or batch

        # ----- Batch Parameters ----- #
        batch_size=256,

        # ----- Episodic Parameters ----- #
        way=0,
        shot=0,

        # ----- Loading Parameters ----- #
        cycle_length=None,
        num_parallel_calls=None,
        block_length=1,
        buff_size=2,
        shuffle=False,
    )

@register_config("fs_omniglot/vinyals/test")
def fs_omniglot_vinyals_test():
    hparams = fs_omniglot_default()
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Gurmukhi/character42.npz,Gurmukhi/character43.npz,Gurmukhi/character44.npz,Gurmukhi/character45.npz,"
                                + "Kannada,Keble,Malayalam,Manipuri,Mongolian,Old_Church_Slavonic_(Cyrillic),Oriya,Syriac_(Serto),Sylheti,"
                                + "Tengwar,Tibetan,ULOG")

    return hparams

@register_config("fs_omniglot/vinyals/train")
def fs_omniglot_vinyals_train():
    hparams = fs_omniglot_default()
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Angelic,Grantha,N_Ko,Aurek-Besh,Japanese_(hiragana),Malay_(Jawi_-_Arabic),Asomtavruli_(Georgian),Sanskrit,"
                                + "Ojibwe_(Canadian_Aboriginal_Syllabics),Korean,Arcadian,Greek,Alphabet_of_the_Magi,"
                                + "Blackfoot_(Canadian_Aboriginal_Syllabics),Futurama,Gurmukhi/character01.npz,Gurmukhi/character02.npz,"
                                + "Gurmukhi/character03.npz,Gurmukhi/character04.npz,Gurmukhi/character05.npz,Gurmukhi/character06.npz,"
                                + "Gurmukhi/character07.npz,Gurmukhi/character08.npz,Gurmukhi/character09.npz,Gurmukhi/character10.npz,"
                                + "Gurmukhi/character11.npz,Gurmukhi/character12.npz,Gurmukhi/character13.npz,Gurmukhi/character14.npz,"
                                + "Gurmukhi/character15.npz,Gurmukhi/character16.npz,Gurmukhi/character17.npz,Gurmukhi/character18.npz,"
                                + "Gurmukhi/character19.npz,Gurmukhi/character20.npz,Gurmukhi/character21.npz,Gurmukhi/character22.npz,"
                                + "Gurmukhi/character23.npz,Gurmukhi/character24.npz,Gurmukhi/character25.npz,Gurmukhi/character26.npz,"
                                + "Gurmukhi/character27.npz,Gurmukhi/character28.npz,Gurmukhi/character29.npz,Gurmukhi/character30.npz,"
                                + "Gurmukhi/character31.npz,Gurmukhi/character32.npz,Gurmukhi/character33.npz,Gurmukhi/character34.npz,"
                                + "Gurmukhi/character35.npz,Gurmukhi/character36.npz,Gurmukhi/character37.npz,Gurmukhi/character38.npz,"
                                + "Gurmukhi/character39.npz,Gurmukhi/character40.npz,Gurmukhi/character41.npz,Tagalog,Anglo-Saxon_Futhorc,"
                                + "Braille,Cyrillic,Burmese_(Myanmar),Avesta,Gujarati,Ge_ez,Syriac_(Estrangelo),Atlantean,"
                                + "Japanese_(katakana),Balinese,Atemayar_Qelisayer,Glagolitic,Tifinagh,Latin,"
                                + "Inuktitut_(Canadian_Aboriginal_Syllabics)")
    hparams.set_hparam("shuffle", True)
    return hparams

@register_config("fs_omniglot/vinyals/train/noisy")
def fs_omniglot_vinyals_train_noisy():
    hparams = fs_omniglot_vinyals_train()
    hparams.add_hparam("augmentations", [[teacher_noise_4]])

    return hparams

@register_config("fs_omniglot/vinyals/train/noisy_rotate")
def fs_omniglot_vinyals_train_noisy():
    hparams = fs_omniglot_vinyals_train()
    hparams.add_hparam("augmentations", [[teacher_noise_4, rotate_4]])

    return hparams

@register_config("fs_omniglot/vinyals/train/rotate")
def fs_omniglot_vinyals_train_noisy():
    hparams = fs_omniglot_vinyals_train()
    hparams.add_hparam("augmentations", [[rotate_4]])

    return hparams

@register_config("fs_omniglot/vinyals/test/20way1shot")
def fs_omniglot_vinyals_test_20way1shot():
    hparams = fs_omniglot_vinyals_test()
    hparams.set_hparam("way", 20)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/vinyals/train/20way1shot")
def fs_omniglot_vinyals_train_20way1shot():
    hparams = fs_omniglot_vinyals_train()
    hparams.del_hparam("augmentations")  # TODO Augmentations currently do not support episodic few-shot, but we don't really care...
    """
    The reason for this is because we are performing augmentation after class selection has occurred. That is to say, if I select
    class c1, c2, ..., c20, post selection they are augmented k times per example and this would become a k * 20 way classification.
    Solutions to this would entail either cutting a random portion of the classes in the augmentation step, OR simply preparing the dataset
    externally and saving it as such. Cutting the portion of classes would likely entail randomely sampling 1 augmentation from each 
    generator OR sampling after augmentation batch is done. If this is the case, we may remove the line above.
    """
    hparams.set_hparam("way", 20)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/vinyals/test/5way1shot")
def fs_omniglot_vinyals_test_5way1shot():
    hparams = fs_omniglot_vinyals_test()
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/vinyals/train/5way1shot")
def fs_omniglot_vinyals_train_5way1shot():
    hparams = fs_omniglot_vinyals_train()
    hparams.del_hparam("augmentations")  # TODO Augmentations currently do not support episodic few-shot, but we don't really care...
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/lake/test")
def fs_omniglot_lake_test():
    hparams = fs_omniglot_default()

    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Angelic,Atemayar_Qelisayer,Atlantean,Aurek-Besh,Avesta,Ge_ez,Glagolitic,Gurmukhi,Kannada,Keble,"
                                + "Malayalam,Manipuri,Mongolian,Old_Church_Slavonic_(Cyrillic),Oriya,Sylheti,Syriac_(Serto),Tengwar,"
                                + "Tibetan,ULOG")

    return hparams

@register_config("fs_omniglot/lake/train")
def fs_omniglot_lake_train():
    hparams = fs_omniglot_default()

    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Alphabet_of_the_Magi,Anglo-Saxon_Futhorc,Arcadian,Armenian,Asomtavruli_(Georgian),Balinese,Bengali,"
                                "Blackfoot_(Canadian_Aboriginal_Syllabics),Braille,Burmese_(Myanmar),Cyrillic,Early_Aramaic,Futurama,"
                                "Grantha,Greek,Gujarati,Hebrew,Inuktitut_(Canadian_Aboriginal_Syllabics),Japanese_(hiragana),"
                                "Japanese_(katakana),Korean,Latin,Malay_(Jawi_-_Arabic),Mkhedruli_(Georgian),N_Ko,"
                                "Ojibwe_(Canadian_Aboriginal_Syllabics),Sanskrit,Syriac_(Estrangelo),Tagalog,Tifinagh")

    hparams.set_hparam("shuffle", True)
    hparams.set_hparam("cycle_length", 50)

    return hparams


@register_config("fs_omniglot/lake/train/noisy")
def fs_omniglot_lake_train_noisy():
    hparams = fs_omniglot_lake_train()
    hparams.add_hparam("augmentations", [[teacher_noise_4]])

    return hparams


@register_config("fs_omniglot/lake/train/noisy_rotate")
def fs_omniglot_lake_train_noisy():
    hparams = fs_omniglot_lake_train()
    hparams.add_hparam("augmentations", [[teacher_noise_4, rotate_4]])

    return hparams\

@register_config("fs_omniglot/lake/train/rotate")
def fs_omniglot_lake_train_rotate():
    hparams = fs_omniglot_lake_train()
    hparams.add_hparam("augmentations", [[rotate_4]])

    return hparams

@register_config("fs_omniglot/lake/test/20way1shot")
def fs_omniglot_lake_test_20way1shot():
    hparams = fs_omniglot_lake_test()

    hparams.del_hparam("augmentations")  # TODO Augmentations currently do not support episodic few-shot, but we don't really care...
    hparams.set_hparam("split", hparams.split.replace(",", ";"))
    hparams.set_hparam("way", 20)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/lake/train/20way1shot")
def fs_omniglot_lake_train_20way1shot():
    hparams = fs_omniglot_lake_train()

    hparams.del_hparam("augmentations")  # TODO Augmentations currently do not support episodic few-shot, but we don't really care...
    """4 of the alphabets have <20 characters. We will combine the two of them together."""
    split = hparams.split.split(",")
    split[split.index("Blackfoot_(Canadian_Aboriginal_Syllabics)")] = "Blackfoot_(Canadian_Aboriginal_Syllabics)," \
                                                                      + "Inuktitut_(Canadian_Aboriginal_Syllabics)"
    split.remove("Inuktitut_(Canadian_Aboriginal_Syllabics)")
    split[split.index("Ojibwe_(Canadian_Aboriginal_Syllabics)")] = "Ojibwe_(Canadian_Aboriginal_Syllabics),Tagalog"
    split.remove("Tagalog")

    hparams.set_hparam("split", ";".join(split))
    hparams.set_hparam("way", 20)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/lake/test/5way1shot")
def fs_omniglot_lake_test_5way1shot():
    hparams = fs_omniglot_lake_test()
    hparams.set_hparam("split", hparams.split.replace(",", ";"))
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/lake/train/5way1shot")
def fs_omniglot_lake_train_5way1shot():
    hparams = fs_omniglot_lake_train()

    hparams.set_hparam("split", hparams.split.replace(",", ";"))
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

