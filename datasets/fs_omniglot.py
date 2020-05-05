import os

import numpy as np
from PIL import Image
from absl import logging

from datasets.base import register_dataset, DatasetBase
from util import string_to_strokes, apply_rdp, strokes_to_stroke_three, scale_and_rasterize, stroke_five_format, stroke_three_format, \
    get_normalizing_scale_factor, scale_and_center_stroke_three, stroke_five_format_scaled_and_centered, rasterize, \
    stroke_three_format_scaled_and_centered


@register_dataset("fs_omniglot")
class FSOmniglotDataset(DatasetBase):
    def __init__(self, data_dir, params):
        super(FSOmniglotDataset, self).__init__(data_dir, params)

        self._mode = params.mode
        self.way = params.way
        self.shot = params.shot

        if "augmentations" in params:
            self._augmentations = params.augmentations

        self._dataset_path = os.path.join(self._data_dir, "fs_omniglot")

    def load(self, repeat=True):
        data_path = os.path.join(self._dataset_path, 'caches')

        if not self._split:
            self._split = self._split = ','.join(sorted(os.listdir(data_path)))

        if self._mode == "batch":
            # If not episdic, i.e. conventional loading
            files = []
            class_list = []
            for alphabet in self._split.split(","):
                path = os.path.join(data_path, alphabet)
                if os.path.isdir(path):
                    file_list = os.listdir(os.path.join(data_path, alphabet))
                    files.extend([os.path.join(data_path, alphabet, c) for c in file_list])
                    class_list.extend([alphabet + str(c).split(".")[0][-2:] for c in file_list])
                else:
                    files.append(path)
                    class_list.append(alphabet + str(path).split(".")[0][-2:])
            return self._create_dataset_from_filepaths(files, repeat), class_list
        else:
            logging.fatal("Dataset mode not \"episodic\" or \"batch\", value supplied: %s", self._mode)

    def _filter_collections(self, files):
        """
        :param files:
        :return: y_strokes(ground_truth), y_strokes(teacher) x_image, class_names
        """
        files = sorted(files)
        return files[4], files[2]
