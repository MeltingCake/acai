import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from PIL import Image

from .utils import gaussian_blur
from .quickdraw_utils import scale_and_rasterize, stroke_three_format


def rasterize_prediction(image, sketch, class_name, prediction):
    png_dimensions = image.shape[:-1]

    stroke_three = stroke_three_format(prediction)
    rasterized_prediction = scale_and_rasterize(stroke_three, png_dimensions)

    return {"rasterized_images": image, "stroke_five_sketches": sketch, "class_names": class_name,
            "stroke_predictions": prediction, "rasterized_predictions": rasterized_prediction}


def compute_pixel_loss(pi, mu1, mu2, sigma1, sigma2, rho, pen, pixel_gt):
    batch_size = pixel_gt.shape[0]

    predicted_strokes = params_to_strokes(pi, mu1, mu2, sigma1, sigma2, rho, pen, batch_size)
    predicted_pixels = strokes_to_image(predicted_strokes, pixel_gt.shape[1:3])

    pix_blur, pixel_gt_blur = tf.split(gaussian_blur(tf.concat((predicted_pixels, pixel_gt), axis=0), (4, 4), 2.), 2, axis=0)

    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(pixel_gt_blur, pix_blur), axis=(1, 2))

    # Image.fromarray(np.concatenate(((pixel_gt[0, :, :, 0] * 255.0).numpy().astype(np.uint8),
    #                                 (pixel_gt_blur[0, :, :, 0] * 255.0).numpy().astype(np.uint8),
    #                                 (predicted_pixels[0, :, :, 0] * 255.0).numpy().astype(np.uint8),
    #                                 (pix_blur[0, :, :, 0] * 255.0).numpy().astype(np.uint8)))
    #                 ).save("toy/dump/300000_pixel/{}.png".format(int(loss[0].numpy() * 100000)))
    return loss

def params_to_strokes(pi, mu1, mu2, sigma1, sigma2, rho, pen, batch_size):
    pen = tf.reshape(pen, shape=(batch_size, -1, 3))

    # This takes the highest weighted Gaussian each step to be plotted.
    max_idx = tf.stack((tf.cast(tf.range(0, pi.shape[0]), tf.int64),
                        tf.argmax(pi, axis=1)), axis=-1)
    step_mu1, step_mu2, step_sigma1, step_sigma2, step_rho = [tf.reshape(tf.gather_nd(param, max_idx), shape=(batch_size, -1))
                                                              for param in [mu1, mu2, sigma1, sigma2, rho]]

    # Compute all my point offsets using parameters per step
    step_mu_2d = tf.stack((step_mu1, step_mu2), axis=-1)
    step_lower_triangular_decomp = tf.stack((tf.stack((step_sigma1, tf.zeros(step_sigma1.shape)), axis=-1),
                                             tf.stack((step_rho * step_sigma2, step_sigma2 * tf.sqrt(1 - step_rho ** 2 + 1e-6)), axis=-1)),
                                            axis=-2)

    mu = tf.reshape(step_mu_2d, (-1, 2))
    eps = tf.random.normal(mu.shape)
    lower_triangular_decomp = tf.reshape(step_lower_triangular_decomp, (-1, 2, 2))
    relative_xy = tf.reshape(mu + tf.einsum("ijk,ik->ij", lower_triangular_decomp, eps), (batch_size, -1, 2))
    # relative_xy = tfp.distributions.MultivariateNormalTriL(mu, lower_triangular_decomp).sample(1)

    # Re-add intial point
    relative_xy = tf.concat((tf.zeros((relative_xy.shape[0], 1, relative_xy.shape[-1])), relative_xy), axis=1)
    pen = tf.concat((tf.tile(tf.constant([[[0., 1., 0.]]]), (batch_size, 1, 1)), pen), axis=1)

    return tf.concat((relative_xy, pen), axis=-1)

def strokes_to_image(strokes, image_dim):
    batch_size = strokes.shape[0]
    relative_xy, pen = strokes[:, :, 0:2], strokes[:, :, 2:]
    abs_xy = tf.cumsum(relative_xy, axis=-2)

    p_1, p_2 = (tf.reshape(x, (batch_size, 1, -1, 2)) for x in (abs_xy[:, :-1, :], abs_xy[:, 1:, :]))
    p_3 = tf.reshape(tf.stack(tf.meshgrid(tf.range(0, image_dim[0], dtype=tf.float32),
                                          tf.range(0, image_dim[1], dtype=tf.float32)), axis=-1), (1, -1, 1, 2))
    ab, ac, bc = p_2 - p_1, p_3 - p_1, p_3 - p_2

    # Computes AB . AC
    ab_dot_ac = tf.einsum("ikl,ijkl->ijk", ab[:, 0], ac)
    ab_cross_ac = (ab[:, :, :, 0] * ac[:, :, :, 1]) - (ab[:, :, :, 1] * ac[:, :, :, 0])
    ab_norm_sq = tf.reduce_sum(ab ** 2, axis=-1)

    pix_dist = tf.where(ab_dot_ac < 0,
                        tf.reduce_sum(ac ** 2, axis=-1),
                        tf.where(ab_dot_ac > ab_norm_sq,
                                 tf.reduce_sum(bc ** 2, axis=-1),
                                 ab_cross_ac ** 2 / (ab_norm_sq + 1e-4)))

    pen_mask = tf.reshape(pen[:, :-1, 0], (batch_size, 1, -1))
    pix_dist += tf.where(pen_mask > 0.5,
                         tf.zeros(pix_dist.shape),
                         tf.ones(pix_dist.shape) * 1e6)
    min_dist = tf.reduce_min(pix_dist, axis=-1)

    pix = tf.sigmoid(2 - 5. * min_dist)
    return tf.reshape(pix, (batch_size + image_dim + [1]))


def compute_pen_state_loss(z_pen_logits, pen_data):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).
    result = tf.nn.softmax_cross_entropy_with_logits(
        labels=pen_data, logits=z_pen_logits)
    result = tf.reshape(result, [-1, 1])

    return result


def compute_mdn_loss(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_gt, x2_gt, pen_gt):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    norm1, norm2, s1s2 = tf.subtract(x1_gt, z_mu1), tf.subtract(x2_gt, z_mu2), tf.multiply(z_sigma1, z_sigma2)
    epsilon = 1e-6

    # Eq 25
    z = (tf.square(tf.divide(norm1, z_sigma1)) + tf.square(tf.divide(norm2, z_sigma2)) -
         2 * tf.divide(tf.multiply(z_corr, tf.multiply(norm1, norm2)), s1s2 + epsilon))

    # Eq 24
    neg_rho = 1 - tf.square(z_corr)
    exp = tf.exp(tf.divide(-z, 2 * neg_rho + epsilon))
    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho + epsilon))

    gmm_pdf = tf.divide(exp, denom + epsilon)

    # Weight GMM PDF
    weighted_gmm_pdf = z_pi * gmm_pdf

    unnorm_log_likelihood = tf.reduce_sum(weighted_gmm_pdf, 1, keepdims=True)
    result = -tf.math.log(unnorm_log_likelihood + epsilon)

    # Zero out loss terms beyond N_s, the last actual stroke
    fs = 1.0 - pen_gt[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    result = tf.multiply(result, fs)

    return result


def compute_grid_loss(grid, x1_gt, x2_gt, pen_gt):
    dim = grid.shape[-1]
    center = tf.cast(dim/2, tf.int32)

    x_ind, y_ind = tf.cast(dim * x1_gt/2, tf.int32) + center, tf.cast(dim * x2_gt/2, tf.int32) + center
    x_ind = tf.math.minimum(dim-1, tf.math.maximum(0, x_ind))
    y_ind = tf.math.minimum(dim-1, tf.math.maximum(0, y_ind))
    batch_idx = tf.constant([[i] for i in range(x_ind.shape[0])])

    scatter_idx = tf.concat([batch_idx, x_ind + center, y_ind + center], axis=1)
    labels = tf.scatter_nd(scatter_idx, updates=tf.ones(scatter_idx.shape[0]), shape=tf.shape(grid))

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(labels, (-1, dim**2)), logits=tf.reshape(grid, (-1, dim**2)))

    fs = 1.0 - pen_gt[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    result = tf.multiply(loss, fs)

    return result


def get_mixture_coef(output):
    """ Returns the tf slices containing mdn dist params. """
    # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
    z = output
    # z = output

    z_pen_logits = z[:, 0:3]  # pen states
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

    # softmax all the pi's and pen states:
    z_pi = tf.nn.softmax(z_pi)
    z_pen = tf.nn.softmax(z_pen_logits)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)  # \rho

    r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
    return r


def sample_2d_normal(inp_mu1, inp_mu2, inp_sigma1, inp_sigma2, inp_rho):
    """
    DEPRECATED, OPERATION DONE IN TENSORFLOW

    Samples from a 2D Normal Distribution
    :param inp_mu1:
    :param inp_mu2:
    :param inp_sigma1:
    :param inp_sigma2:
    :param inp_rho:
    :return:
    """
    res = np.zeros((inp_mu1.shape[0], 2))

    for idx in range(inp_mu1.shape[0]):
        mu1, mu2, sigma1, sigma2, rho = inp_mu1[idx], inp_mu2[idx], inp_sigma1[idx], inp_sigma2[idx], inp_rho[idx]

        mixture_covariance = [[sigma1 ** 2, rho * sigma1 * sigma2], [rho * sigma2 * sigma1, sigma2 ** 2]]
        sample = np.random.multivariate_normal([mu1, mu2], mixture_covariance, 1)
        res[idx, :] = sample[0, :]
    return res
