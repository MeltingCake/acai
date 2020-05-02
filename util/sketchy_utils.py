import os

import numpy as np
from PIL import Image
from absl import logging

from util import string_to_strokes, apply_rdp, strokes_to_stroke_three, scale_and_center_stroke_three, \
    stroke_five_format_scaled_and_centered, rasterize, stroke_three_format_scaled_and_centered


def sketch_process(batch_data, padding, max_seq_len, epsilon, png_dims, save_path, sample_path):
    accumulate = {"natural_image": [], "sketch_path": [], "strokes": [], "rasterized_strokes": [], "imagenet_id": [], "sketch_id": []}
    for natural_image, sketch_path, svg, sketch_id, bbx, width_height in batch_data:
        imagenet_id = sketch_id.split("-")[0]
        image: Image = natural_image
        crop_box = [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]
        crop = image.resize(width_height).crop(crop_box)
        crop.save(os.path.join(sample_path + "_test.png"))
        scale = min(png_dims[0] / bbx[2], png_dims[1] / bbx[3])

        resize = crop.resize([int(x * scale) for x in bbx[2:]])
        img_w, img_h = resize.size
        pasted_image = Image.new("RGB", png_dims, (0, 0, 0))
        pasted_image.paste(resize, ((png_dims[0] - img_w) // 2, (png_dims[1] - img_h) // 2))

        processed_natural_image = pasted_image

        stroke_str = "START\n"
        total_length = sum([path.length() for path in svg])
        max_length = max([path.length() for path in svg])
        for path in svg:
            if path.length() < 0.1 * max_length:
                continue
            for curve in path:
                curve_length = curve.length()
                if curve_length == 0.0:
                    continue
                for d in np.linspace(0, curve_length, max(int(curve_length // 20), 3)) / curve_length:
                    point = curve.point(d)
                    x, y = np.real(point), np.imag(point)
                    stroke_str += "{},{}\n".format(x, y)
            stroke_str += "BREAK\n"
        stroke_str = stroke_str[:-1]
        strokes = string_to_strokes(stroke_str)
        strokes = apply_rdp(strokes, epsilon=epsilon)

        stroke_three = strokes_to_stroke_three(strokes)
        stroke_three_scaled_and_centered = scale_and_center_stroke_three(stroke_three, png_dimensions=png_dims, padding=padding)

        try:
            stroke_five = stroke_five_format_scaled_and_centered(stroke_three_scaled_and_centered, max_seq_len)
        except:
            logging.info("Stroke limit exceeds 65 for example: %s | length: %s", sketch_id, stroke_three_scaled_and_centered.shape[0])
            continue

        rasterized_strokes = rasterize(stroke_three, png_dims)

        accumulate["natural_image"].append(np.array(processed_natural_image, dtype=np.float32))
        accumulate["sketch_path"].append(sketch_path)
        accumulate["rasterized_strokes"].append(np.array(rasterized_strokes, dtype=np.float32))
        accumulate["strokes"].append(stroke_five.astype(np.float32))
        accumulate["imagenet_id"].append(imagenet_id)
        accumulate["sketch_id"].append(sketch_id)

    rand_idx = np.random.randint(0, len(accumulate["natural_image"]) - 1)
    im, im_raster = (Image.fromarray(accumulate['natural_image'][rand_idx].astype('uint8')),
                             Image.fromarray(accumulate['rasterized_strokes'][rand_idx].astype('uint8')))
    try:
        sketch = Image.open(accumulate["sketch_path"][rand_idx])
        sketch.save(os.path.join(sample_path + "_{}_sketch.png".format(accumulate['sketch_id'][rand_idx])))
    except:
        logging.info("Sketch not found: %s", accumulate["sketch_path"][rand_idx])
    stroke_three_string = "\n".join([str(x) for x in stroke_three_format_scaled_and_centered(accumulate['strokes'][rand_idx])])

    im.save(os.path.join(sample_path + "_{}_gt.png".format(accumulate['sketch_id'][rand_idx])))
    im_raster.save(os.path.join(sample_path + "_{}_raster.png".format(accumulate['sketch_id'][rand_idx])))
    with open(os.path.join(sample_path + "_{}_strokes.txt".format(accumulate['sketch_id'][rand_idx])), 'w') as f:
        f.write(stroke_three_string)

    del accumulate['sketch_path']
    np.savez(save_path, **accumulate)

    return len(accumulate["sketch_id"])
