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
            for alphabet in self._split.split(","):
                path = os.path.join(data_path, alphabet)
                if os.path.isdir(path):
                    file_list = os.listdir(os.path.join(data_path, alphabet))
                    files.extend([os.path.join(data_path, alphabet, c) for c in file_list])
                else:
                    files.append(path)

            return self._create_dataset_from_filepaths(files, repeat)
        elif self._mode == "episodic":
            episodes = []

            for episode_string in self._split.split(";"):
                files = []
                for collection in episode_string.split(","):
                    path = os.path.join(data_path, collection)
                    if os.path.isdir(path):
                        file_list = os.listdir(os.path.join(data_path, collection))
                        files.extend([os.path.join(data_path, collection, file) for file in file_list])
                    else:
                        files.append(path)
                episodes.append(",".join(files))

            return self._create_episodic_dataset_from_nested_filespaths(episodes, self.shot, self.way, repeat)
        else:
            logging.fatal("Dataset mode not \"episodic\" or \"batch\", value supplied: %s", self._mode)

    def _filter_collections(self, files):
        """
        :param files:
        :return: y_strokes(ground_truth), y_strokes(teacher) x_image, class_names
        """
        files = sorted(files)
        return files[5], files[5], files[4], files[2]

    def prepare(self, FLAGS, epsilon=2., png_dims=(48, 48), padding=None):
        padding = padding if padding else round(min(png_dims) / 10.0) * 2

        save_dir = os.path.join(self._dataset_path, "caches")
        sample_dir = os.path.join(self._dataset_path, "processing-samples")
        raw_dir = os.path.join(self._dataset_path, "raw")

        images_background = os.path.join(raw_dir, "images_background")
        strokes_evaluation = os.path.join(raw_dir, "strokes_evaluation")

        images_evaluation = os.path.join(raw_dir, "images_evaluation")
        strokes_background = os.path.join(raw_dir, "strokes_background")

        logging.info("Processing FSO | png_dimensions: %s | padding: %s", png_dims, padding)

        for image_dir, stroke_dir in [(images_background, strokes_background), (images_evaluation, strokes_evaluation)]:
            alphabet_list = os.listdir(image_dir)

            for alphabet in alphabet_list:
                character_list = os.listdir(os.path.join(image_dir, alphabet))

                for character in character_list:
                    logging.info("Processing | Alphabet: %s | Character: %s", alphabet, character)
                    image_files = sorted(os.listdir(os.path.join(image_dir, alphabet, character)))
                    stroke_files = sorted(os.listdir(os.path.join(stroke_dir, alphabet, character)))

                    accumulate = {"image": [], "strokes": [], "rasterized_strokes": [], "character": [], "alphabet": [], "class": []}

                    for image_file, stroke_file in zip(image_files, stroke_files):
                        image: Image.Image = Image.open(os.path.join(image_dir, alphabet, character, image_file)).convert("RGB")
                        strokes_str = open(os.path.join(stroke_dir, alphabet, character, stroke_file)).read()

                        strokes = string_to_strokes(strokes_str)
                        strokes = apply_rdp(strokes, epsilon=epsilon)

                        stroke_three = strokes_to_stroke_three(strokes)
                        stroke_three_scaled_and_centered = scale_and_center_stroke_three(stroke_three, png_dimensions=png_dims, padding=padding)

                        try:
                            stroke_five = stroke_five_format_scaled_and_centered(stroke_three_scaled_and_centered, 65)
                        except:
                            logging.info("Stroke limit exceeds 65 for example: %s",
                                         os.path.join(stroke_dir, alphabet, character, stroke_file))

                        rasterized_strokes = rasterize(stroke_three, png_dims)

                        accumulate["alphabet"].append(alphabet)
                        accumulate["character"].append(int(character[-2:]))
                        accumulate["class"].append(alphabet + character[-2:])
                        accumulate["image"].append(np.array(image, dtype=np.float32))
                        accumulate["rasterized_strokes"].append(np.array(rasterized_strokes, dtype=np.float32))
                        accumulate["strokes"].append(stroke_five.astype(np.float32))

                    # Sample random example
                    rand_idx = np.random.randint(0, len(accumulate["image"])-1)
                    im, im_raster = (Image.fromarray(accumulate['image'][rand_idx].astype('uint8')),
                                     Image.fromarray(accumulate['rasterized_strokes'][rand_idx].astype('uint8')))
                    stroke_three_string = "\n".join([str(x) for x in stroke_three_format_scaled_and_centered(accumulate['strokes'][rand_idx])])

                    save_path = os.path.join(sample_dir, alphabet, character)
                    os.makedirs(save_path, exist_ok=True)

                    im.save(os.path.join(save_path, str(rand_idx) + "_gt.png"))
                    im_raster.save(os.path.join(save_path, str(rand_idx) + "_raster.png"))
                    with open(os.path.join(save_path, str(rand_idx) + "_strokes.txt"), 'w') as f:
                        f.write(stroke_three_string)

                    # Save Archive dir
                    char_save_dir, char_save_path = os.path.join(save_dir, alphabet), os.path.join(save_dir, alphabet, character + ".npz")
                    os.makedirs(char_save_dir, exist_ok=True)

                    np.savez(char_save_path, **accumulate)
