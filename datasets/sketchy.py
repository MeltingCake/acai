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

    def prepare(self, FLAGS, epsilon=5.0, max_seq_len=100, png_dims=(48, 48), padding=None, shard_size=1000):
        padding = padding if padding else round(min(png_dims) / 10.0) * 2

        save_dir = os.path.join(self._dataset_path, "caches")
        sample_dir = os.path.join(self._dataset_path, "processing-samples")
        raw_dir = os.path.join(self._dataset_path, "raw")

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        raw_info_dir = os.path.join(raw_dir, "info-06-04", "info")
        raw_photo_dir = os.path.join(raw_dir, "rendered_256x256", "256x256", "photo", "tx_000000000000")
        raw_sketch_dir = os.path.join(raw_dir, "rendered_256x256", "256x256", "sketch", "tx_000100000000")
        raw_svg_dir = os.path.join(raw_dir, "sketches-06-04", "sketches")

        logging.info("Processing Sketchy | png_dimensions: %s | padding: %s", png_dims, padding)

        imagenet_to_bbox = {}
        with open(os.path.join(raw_info_dir, "stats.csv"), 'r', newline='\n') as statscsv:
            reader = csv.reader(statscsv, delimiter=',')
            next(reader)
            for row in reader:
                imagenet_id, bbox, width_height = row[2], row[14:18], row[12:14]
                if imagenet_id not in imagenet_to_bbox:
                    # Note: BBOX is in BBox_x, BBox_y, BBox_width, BBox_height
                    imagenet_to_bbox[imagenet_id] = [int(x) for x in bbox], [int(x) for x in width_height]

        all_examples = np.empty((0, 6))
        class_list = os.listdir(raw_photo_dir)
        for class_name in class_list:
            logging.info("Loading Class: %s", class_name)
            photo_folder = os.path.join(raw_photo_dir, class_name)
            sketch_folder = os.path.join(raw_sketch_dir, class_name)
            svg_folder = os.path.join(raw_svg_dir, class_name)

            valid_sketches_for_class = open(os.path.join(svg_folder, "checked.txt")).read().splitlines()
            invalid_sketches_for_class = set(open(os.path.join(svg_folder, "invalid.txt")).read().splitlines())
            for photo_file in os.listdir(photo_folder):
                imagenet_id = photo_file.split(".")[0]
                sketches_for_photo = list(filter(lambda x: photo_file[:-4] in x, valid_sketches_for_class))
                valid_sketches_for_photo = list(filter(lambda sketch: sketch not in invalid_sketches_for_class, sketches_for_photo))

                natural_image = Image.open(os.path.join(photo_folder, photo_file))
                for valid_sketch in valid_sketches_for_photo:
                    valid_sketch_path = os.path.join(svg_folder, valid_sketch+".svg")
                    try:
                        svg = svg2paths(valid_sketch_path)[0]
                    except:
                        with open(valid_sketch_path, "r") as errorsvg:
                            val = errorsvg.read()
                        if "</svg>" not in val[-10:]:
                            with open(valid_sketch_path, "a") as errorsvg:
                                errorsvg.write("</svg>\n")
                            logging.info("fixed %s", valid_sketch_path)
                        else:
                            logging.info("still_broken %s", valid_sketch_path)
                        continue

                    sketch_path = os.path.join(sketch_folder, valid_sketch + ".png")

                    x = np.array([[natural_image, sketch_path, svg, valid_sketch] + list(imagenet_to_bbox[imagenet_id])])
                    all_examples = np.concatenate((all_examples, x))

        np.random.shuffle(all_examples)

        logging.info("Beginning Processing | %s sketches | %s classes | %s max_sequence_length",
                     all_examples.shape[0], len(class_list), max_seq_len)

        cpu_count = psutil.cpu_count(logical=False)
        workers_per_cpu = 1
        with ProcessPoolExecutor(max_workers=int(cpu_count * workers_per_cpu)) as executor:
            out = executor.map(sketch_process,
                               (all_examples[i: i + shard_size] for i in range(0, all_examples.shape[0] + shard_size, shard_size)),
                               repeat(padding),
                               repeat(max_seq_len),
                               repeat(epsilon),
                               repeat(png_dims),
                               (os.path.join(save_dir, "{}.npz".format(i)) for i in range(ceil(all_examples.shape[0] // shard_size) + 1)),
                               (os.path.join(sample_dir, "{}".format(i)) for i in range(ceil(all_examples.shape[0] // shard_size) + 1)),
                               chunksize=1)

            total_count = 0
            last_time = time()
            try:
                for write_signal in out:
                    total_count += write_signal
                    curr_time = time()
                    logging.info("Processed Total: {:8}/{:8} | Time/Batch: {:8.2f} | Time/Image: {:8.8f}"
                                .format(total_count,
                                        all_examples.shape[0],
                                        (curr_time - last_time) / cpu_count,
                                        (curr_time - last_time) / (cpu_count * shard_size)))
                    last_time = curr_time
            except Exception as e:
                logging.info("Processing Done")
                raise e

