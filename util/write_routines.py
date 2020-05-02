import os
from time import time

import numpy as np
from PIL import Image
from absl import logging

from util import stroke_three_format, scale_and_rasterize, stroke_three_format_scaled_and_centered, rasterize


def parallel_writer_sketches(path, queue, shard_size=100):
    logging.info("Archiving latent outputs to: %s", path)
    shard_path, sample_path = os.path.join(path, "archives"), os.path.join(path, "samples")
    os.makedirs(shard_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    start_time = last_time = time()

    accumulate, count = None, 0
    while True:
        entry = queue.get()
        if not entry:
            # if accumulate:
            #     np.savez(os.path.join(shard_path, "{}.npz".format(count//shard_size)), **accumulate)
            break

        if not accumulate:
            accumulate = {key: [] for key in entry}
            accumulate['rasterized_predictions'] = []

        stroke_three = stroke_three_format(entry["stroke_predictions"])
        entry["rasterized_predictions"] = scale_and_rasterize(stroke_three, entry['rasterized_images'].shape[:-1]).astype("float32")

        count += 1
        [accumulate[key].append(entry[key]) for key in entry]

        if count and count % shard_size == 0:
            np_image = np.concatenate((entry["rasterized_images"], entry["rasterized_predictions"]))
            img = Image.fromarray(np_image.astype("uint8"))
            try:
                img.save(os.path.join(sample_path, "{}-{}.jpg".format(entry["class_names"].decode("utf-8"), count)))
            except:
                img.save(os.path.join(sample_path, "{}-{}.jpg".format(entry["class_names"], count)))

            # np.savez(os.path.join(shard_path, "{}.npz".format(count//shard_size)), **accumulate)
            accumulate = None

            curr_time = time()
            logging.info("Samples complete: %6d | Time/Sample: %5.4f | Total Elapsed Time: %7d",
                         count, (curr_time-last_time)/shard_size, curr_time-start_time)
            last_time = curr_time


def parallel_writer_latent(path, queue, shard_size=1000):
    logging.info("Archiving latent outputs to: %s", path)
    shard_path = os.path.join(path)
    os.makedirs(shard_path, exist_ok=True)

    start_time = last_time = time()
    accumulate, count = None, 0
    while True:
        entry = queue.get()
        if not entry:
            logging.info("None observed, terminating writing to path: %s", shard_path)
            np.savez(os.path.join(shard_path, "{}.npz".format(count//shard_size)), **accumulate)
            break

        if not accumulate:
            accumulate = {key: [] for key in entry}

        count += 1
        [accumulate[key].append(entry[key]) for key in entry]

        if count and count % shard_size == 0:
            np.savez(os.path.join(shard_path, "{}.npz".format(count//shard_size)), **accumulate)
            accumulate = None

            curr_time = time()
            logging.info("Samples complete: %6d | Time/Sample: %5.4f | Total Elapsed Time: %7d",
                         count, (curr_time-last_time)/shard_size, curr_time-start_time)
            last_time = curr_time


def parallel_writer_vae_latent(path, queue, shard_size=1):
    logging.info("Archiving vae latent outputs to %s", path)

    shard_path, sample_path = os.path.join(path, "archives"), os.path.join(path, "samples")
    os.makedirs(shard_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    start_time = last_time = time()

    accumulate, count = None, 0
    while True:
        entry = queue.get()
        if not entry:
            np.savez(os.path.join(shard_path, "{}.npz".format(count // shard_size)), **accumulate)
            break

        if not accumulate:
            accumulate = {key: [] for key in entry}

        count += 1
        [accumulate[key].append(entry[key]) for key in entry]

        if count and count % shard_size == 0:
            np_image = np.concatenate((entry["rasterized_images"], entry["reconstructed_images"]))
            np_image = np_image.squeeze() * 255.0
            img = Image.fromarray(np_image.astype("uint8"))
            try:
                img.save(os.path.join(sample_path, "{}-{}.jpg".format(entry["class_names"].decode("utf-8"), count)))
            except:
                img.save(os.path.join(sample_path, "{}-{}.jpg".format(entry["class_names"], count)))

            np.savez(os.path.join(shard_path, "{}.npz".format(count // shard_size)), **accumulate)
            accumulate = None

            curr_time = time()
            logging.info("Samples complete: %6d | Time/Sample: %5.4f | Total Elapsed Time: %7d",
                         count, (curr_time - last_time) / shard_size, curr_time - start_time)
            last_time = curr_time
