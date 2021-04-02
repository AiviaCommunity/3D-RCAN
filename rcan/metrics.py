# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import keras.backend as K
import tensorflow as tf


def _get_gaussian_kernel(dim, size, sigma):
    k = size // 2
    normal = tf.distributions.Normal(0.0, sigma)
    p = normal.prob(tf.range(-k, size - k, dtype=tf.float32))

    indices = [chr(i) for i in range(105, 105 + dim)]
    eq = ','.join(indices) + '->' + ''.join(indices)
    kernel = tf.einsum(eq, *([p] * dim))
    kernel /= tf.reduce_sum(kernel)
    kernel = kernel[..., tf.newaxis, tf.newaxis]

    return kernel


def psnr(y_true, y_pred):
    '''
    Computs the peak signal-to-noise ratio between two images. Note that the
    maximum signal value is assumed to be 1.
    '''
    p, q = [K.batch_flatten(y) for y in [y_true, y_pred]]
    return -4.342944819 * K.log(K.mean(K.square(p - q), axis=-1))


def ssim(y_true, y_pred):
    '''
    Computes the structural similarity index between two images. Note that the
    maximum signal value is assumed to be 1.

    References
    ----------
    Image Quality Assessment: From Error Visibility to Structural Similarity
    https://doi.org/10.1109/TIP.2003.819861
    '''

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    dim = K.ndim(y_pred) - 2
    if dim not in (2, 3):
        raise NotImplementedError(f'{dim}D SSIM is not suported')

    num_channels = K.int_shape(y_pred)[-1]

    kernel = _get_gaussian_kernel(dim, 11, 1.5)
    conv = K.conv2d if dim == 2 else K.conv3d

    def average(x):
        # channel-wise weighted average using the Gaussian kernel
        return tf.concat(
            [conv(y, kernel) for y in tf.split(x, num_channels, axis=-1)],
            axis=-1)

    ux = average(y_true)
    uy = average(y_pred)

    a = ux * uy
    b = K.square(ux) + K.square(uy)
    c = average(y_true * y_pred)
    d = average(K.square(y_true) + K.square(y_pred))

    lum = (2 * a + c1) / (b + c1)
    cs = (2 * (c - a) + c2) / (d - b + c2)

    return K.mean(K.batch_flatten(lum * cs), axis=-1)
