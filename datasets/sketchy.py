import os
from concurrent.futures.process import ProcessPoolExecutor
from time import time
from itertools import repeat
from math import ceil

import numpy as np
import psutil
import csv
from PIL import Image

from absl import logging
from svgpathtools import svg2paths

from datasets.base import register_dataset, DatasetBase
from util import sketch_process


@register_dataset("sketchy")
class SketchyDataset(DatasetBase):
    def __init__(self, data_dir, params):
        super(SketchyDataset, self).__init__(data_dir, params)

        self._dataset_path = os.path.join(self._data_dir, "sketchy")

    def load(self, repeat=True):
        data_path = os.path.join(self._dataset_path, 'caches', self._split)
        files = [os.path.join(data_path, shard_name) for shard_name in os.listdir(data_path)]

        return self._create_dataset_from_filepaths(files, repeat)

    def _filter_collections(self, files):
        """
        files = ['natural_image', 'strokes', 'rasterized_strokes', 'imagenet_id', 'sketch_id']
        :param files:
        :return: strokes_gt, strokes_teacher, natural_image, class_name
        """
        return files[1], files[1], files[0], files[3]

