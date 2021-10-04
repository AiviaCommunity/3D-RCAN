# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import itertools
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf


def _register_keras_custom_object(cls):
    keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls


@_register_keras_custom_object
class _PixelShuffle(keras.layers.Layer):
    def __init__(self, rank, scale_factor, downshuffle=False, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.scale_factor = keras.utils.conv_utils.normalize_tuple(
            scale_factor, rank, 'scale_factor')
        self.downshuffle = downshuffle

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        output_shape = self.compute_output_shape(input_shape)

        if self.rank == 2 and len(set(self.scale_factor)) == 1:
            return (tf.space_to_depth if self.downshuffle
                    else tf.depth_to_space)(inputs, self.scale_factor[0])

        if self.downshuffle:
            outputs = K.reshape(
                inputs,
                (-1,
                 *list(itertools.chain.from_iterable(
                     zip(output_shape[1:-1], self.scale_factor))),
                 input_shape[-1]))

            outputs = K.permute_dimensions(
                outputs,
                (0,
                 *list(range(1, 2 * self.rank, 2)),
                 *list(range(2, 2 * self.rank + 1, 2)),
                 2 * self.rank + 1))
        else:
            outputs = K.reshape(
                inputs,
                (-1, *input_shape[1:-1], *self.scale_factor, output_shape[-1]))

            outputs = K.permute_dimensions(
                outputs,
                (0,
                 *[x + 1 for x in itertools.chain.from_iterable(
                   zip(range(self.rank), range(self.rank, 2 * self.rank)))],
                 2 * self.rank + 1))

        return K.reshape(outputs, (-1, *output_shape[1:]))

    def compute_output_shape(self, input_shape):
        if len(input_shape) != self.rank + 2:
            raise ValueError(
                f'Inputs should have rank {self.rank + 2}; '
                f'Received input shape: {input_shape}')

        if self.downshuffle:
            output_shape = [input_shape[0]]
            for s, r in zip(input_shape[1:-1], self.scale_factor):
                if s is None:
                    output_shape.append(s)
                elif s % r != 0:
                    raise ValueError(
                        'For downshuffling, all spatial dimensions must be '
                        'divisible by their corresponding scaling factors; '
                        f'Received input shape: {input_shape}, '
                        f'scaling factors: {self.scale_factor}')
                else:
                    output_shape.append(s // r)
            output_shape.append(input_shape[-1] * np.prod(self.scale_factor))

            return tuple(output_shape)

        else:
            if input_shape[-1] % np.prod(self.scale_factor) != 0:
                raise ValueError(
                    'For upshuffling, the number of input channels must be '
                    f'divisible by {np.prod(self.scale_factor)}; '
                    f'Received input shape: {input_shape}')

            return (
                input_shape[0],
                *[None if s is None else s * r
                  for s, r in zip(input_shape[1:-1], self.scale_factor)],
                input_shape[-1] // np.prod(self.scale_factor))

    def get_config(self):
        config = super().get_config()
        config['rank'] = self.rank
        config['scale_factor'] = self.scale_factor
        config['downshuffle'] = self.downshuffle
        return config


@_register_keras_custom_object
class PixelShuffle2D(_PixelShuffle):
    '''
    If `downshuffle` is false, rearranges elements in a tensor of shape
    `(n, h, w, c*sx*sy)` to a tensor of shape `(n, h*sy, w*sx, c)` where `sx`
    and `sy` are scaling factors. Otherwise, rearranges elements in a tensor of
    shape `(n, h*sy, w*sx, c)` to a tensor of shape `(n, h, w, c*sx*sy)`.

    References
    ----------
    Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network
    https://arxiv.org/abs/1609.05158
    '''

    def __init__(self, scale_factor, downshuffle=False, **kwargs):
        super().__init__(2, scale_factor, downshuffle, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop('rank')
        return config


@_register_keras_custom_object
class PixelShuffle3D(_PixelShuffle):
    '''
    If `downshuffle` is false, rearranges elements in a tensor of shape
    `(n, d, h, w, c*sx*sy*sz)` to a tensor of shape `(n, d*sz h*sy, w*sx, c)`
    where `sx`, `sy` and `sz` are scaling factors. Otherwise, rearranges
    elements in a tensor of shape `(n, d*sz h*sy, w*sx, c)` to a tensor of
    shape `(n, d, h, w, c*sx*sy*sz)`.

    References
    ----------
    Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network
    https://arxiv.org/abs/1609.05158
    '''

    def __init__(self, scale_factor, downshuffle=False, **kwargs):
        super().__init__(3, scale_factor, downshuffle, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop('rank')
        return config
