import os
from io import BytesIO

import numpy as np
import svgwrite
from PIL import Image

try:
    import cairosvg
except:
    cairosvg = None


def stroke_three_format(big_stroke):
    """
    Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3.
    Note this is only for SCALE INVARIANT and UNCENTERED stroke-5 format.
    """
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def stroke_five_format(sketch, max_len):
    """
    Pad the batch to be stroke-5 bigger format as described in paper.
    This is only for SCALE INVARIANT and UNCENTERED stroke-3 format.
    """
    result = np.zeros((max_len + 1, 5), dtype=float)
    sketch_length = len(sketch)

    result[0:sketch_length, 0:2] = sketch[:, 0:2]
    result[0:sketch_length, 3] = sketch[:, 2]
    result[0:sketch_length, 2] = 1 - result[0:sketch_length, 3]
    result[sketch_length:, 4] = 1
    # put in the first token, as described in sketch-rnn methodology
    result[1:, :] = result[:-1, :]
    result[0, :] = np.array([0, 0, 1, 0, 0])

    return result


def stroke_three_format_scaled_and_centered(big_stroke):
    """
    Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3.
    Note: This is only for SCALED AND CENTERED stroke-5 format.
    """
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l-1, 3))
    result[:, 0:2] = big_stroke[1:l, 0:2]
    result[:, 2] = big_stroke[1:l, 3]
    return result


def stroke_five_format_scaled_and_centered(sketch, max_len):
    """
    Pad the batch to be stroke-5 bigger format as described in paper.
    This is only for SCALED AND CENTERED stroke-3 format
    """
    result = np.zeros((max_len + 2, 5), dtype=float)
    sketch_length = len(sketch)

    result[0:sketch_length, 0:2] = sketch[:, 0:2]
    result[0:sketch_length, 3] = sketch[:, 2]
    result[0:sketch_length, 2] = 1 - result[0:sketch_length, 3]
    result[sketch_length:, 4] = 1
    # put in the first token, as described in sketch-rnn methodology
    result[1:, :] = result[:-1, :]
    result[0, :] = np.array([0, 0, 0, 1, 0])

    return result


def scale_and_center_stroke_three(sketch, png_dimensions, padding):
    min_x, max_x, min_y, max_y = _get_bounds(sketch)
    try:
        x_scale = (png_dimensions[0] - padding) / (max_x - min_x)
    except:
        x_scale = float('inf')
    try:
        y_scale = (png_dimensions[1] - padding) / (max_y - min_y)
    except:
        y_scale = float('inf')
    scale = min(x_scale, y_scale)

    sketch[0, 0:2] = [(png_dimensions[0] / 2) - ((max_x + min_x) / 2)*scale, (png_dimensions[1] / 2) - ((max_y + min_y) / 2)*scale]
    sketch[1:, 0:2] *= scale
    return sketch

def rasterize(sketch, png_dimensions):
    drawing_bytestring = _get_svg_string(sketch, png_dimensions)

    png_image = Image.open(BytesIO(cairosvg.svg2png(bytestring=drawing_bytestring, scale=1.0)))

    padded_image = pad_image(png_image, png_dimensions)

    return padded_image

def _get_svg_string(sketch, png_dimensions):
    dims = png_dimensions
    lift_pen = 0
    stroke_width = 1
    color = "black"
    command = "m"

    dwg = svgwrite.Drawing(size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    start_x, start_y = sketch[0, 0:2]
    p = "M%s, %s " % (start_x, start_y)
    for i in range(1, len(sketch)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(sketch[i, 0])
        y = float(sketch[i, 1])
        lift_pen = sketch[i, 2]
        p += command + str(x) + ", " + str(y) + " "

    dwg.add(dwg.path(p).stroke(color, stroke_width).fill("none"))

    return dwg.tostring()


def scale_and_rasterize(sketch, png_dimensions, padding=10):
    """Converts Stroke-3 SVG image to PNG."""
    svg_dimensions, drawing_bytestring = _scale_and_get_svg_string(sketch, png_dimensions, padding=padding)

    svg_width, svg_height = svg_dimensions
    png_width, png_height = png_dimensions
    x_scale = (png_width) / svg_width
    y_scale = (png_height) / svg_height

    png_image = Image.open(BytesIO(cairosvg.svg2png(bytestring=drawing_bytestring, scale=min(x_scale, y_scale))))

    padded_image = pad_image(png_image, png_dimensions)

    return padded_image


def _scale_and_get_svg_string(svg, png_dimensions, padding):
    """Retrieves SVG native dimension and bytestring."""

    min_x, max_x, min_y, max_y = _get_bounds(svg)
    try:
        x_scale = (png_dimensions[0] - padding) / (max_x - min_x)
    except:
        x_scale = float('inf')
    try:
        y_scale = (png_dimensions[1] - padding) / (max_y - min_y)
    except:
        y_scale = float('inf')
    scale = min(x_scale, y_scale)
    dims = png_dimensions
    lift_pen = 1
    stroke_width = 1
    color = "black"
    command = "m"

    dwg = svgwrite.Drawing(size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    start_x = (png_dimensions[0] / 2) - ((max_x + min_x) / 2) * scale
    start_y = (png_dimensions[1] / 2) - ((max_y + min_y) / 2) * scale
    p = "M%s, %s " % (start_x, start_y)
    for i in range(len(svg)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(svg[i, 0]) * scale
        y = float(svg[i, 1]) * scale
        lift_pen = svg[i, 2]
        p += command + str(x) + ", " + str(y) + " "

    dwg.add(dwg.path(p).stroke(color, stroke_width).fill("none"))

    return dims, dwg.tostring()


def _get_bounds(svg):
    """Return bounds of data."""

    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    abs_x, abs_y = 0, 0

    for i in range(len(svg)):
        x, y = float(svg[i, 0]), float(svg[i, 1])
        abs_x += x
        abs_y += y
        min_x, min_y, max_x, max_y = min(min_x, abs_x), min(min_y, abs_y), max(max_x, abs_x), max(max_y, abs_y)

    return min_x, max_x, min_y, max_y


def pad_image(png, png_dimensions):
    png_curr_w = png.width
    png_curr_h = png.height

    padded_png = np.zeros(shape=[png_dimensions[1], png_dimensions[0], 3], dtype=np.uint8)
    padded_png.fill(255)

    if png_curr_w > png_curr_h:
        pad = int(round((png_curr_w - png_curr_h) / 2))
        padded_png[pad: pad + png_curr_h, :png_curr_w] = np.array(png, dtype=np.uint8)
    else:
        pad = int(round((png_curr_h - png_curr_w) / 2))
        padded_png[:png_curr_h, pad: pad + png_curr_w] = np.array(png, dtype=np.uint8)

    return padded_png


def get_normalizing_scale_factor(sketches):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""

    data = []

    for i in range(len(sketches)):
        for j in range(len(sketches[i])):
            data.append(sketches[i][j, 0])
            data.append(sketches[i][j, 1])

    data = np.array(data)
    return np.std(data)

def quickdraw_process(batch_data, max_seq_len, png_dims, save_path,
                gap_limit=1000):
    """Preprocess sketches to drop large gaps, produce sketch-5 format and generate rasterized images."""
    stroke_five_sketches = []
    rasterized_images = []
    class_names = []

    padding = round(min(png_dims)/10.) * 2

    for sketch, class_name in batch_data:
        # cast and scale
        sketch = np.array(sketch, dtype=np.float32)

        # removes large gaps from the data
        sketch = np.maximum(np.minimum(sketch, gap_limit), -gap_limit)
        sketch = np.concatenate((np.array([[0, 0, 0]]), sketch), axis=0)
        sketch = scale_and_center_stroke_three(sketch, png_dims, padding)
        stroke_five_sketch = stroke_five_format_scaled_and_centered(sketch, max_seq_len)

        # Append initial point so that it doesn't lose the first stroke
        #
        raster_image = rasterize(sketch, png_dims)

        stroke_five_sketches.append(stroke_five_sketch)
        rasterized_images.append(raster_image)
        class_names.append(class_name)

    Image.fromarray(rasterized_images[0].astype(np.uint8)).save("rastertest.png")
    np.savez(save_path,
             stroke_five_sketches=np.array(stroke_five_sketches, dtype=np.float32),
             rasterized_images=np.array(rasterized_images, dtype=np.float32),
             class_name=np.array(class_names))

    return 1
