import os
from concurrent.futures.process import ProcessPoolExecutor
from itertools import repeat
from math import ceil
from time import time

import numpy as np
import psutil as psutil
from absl import logging

from datasets.base import register_dataset, DatasetBase
from util import get_normalizing_scale_factor, quickdraw_process


@register_dataset("quickdraw")
class Quickdraw(DatasetBase):
    def __init__(self, data_dir, params):
        super(Quickdraw, self).__init__(data_dir, params)

        self._dataset_path = os.path.join(self._data_dir, 'quickdraw')

    def load(self, repeat=True):
        data_path = os.path.join(self._dataset_path, 'caches', self._split)
        files = [os.path.join(data_path, shard_name) for shard_name in os.listdir(data_path)]

        return self._create_dataset_from_filepaths(files, repeat)

    def _filter_collections(self, files):
        """
        Selects files from archive.
        :param files:
        :return: x_image, class_name
        """
        files = sorted(files)
        return files[1], files[0]
