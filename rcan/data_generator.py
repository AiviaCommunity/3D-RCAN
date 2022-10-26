# Copyright 2021 SVision Technologies LLC.
# Copyright 2021-2022 Leica Microsystems, Inc.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import numpy as np
import tensorflow as tf
import warnings

from tensorflow.python.keras.utils.conv_utils import normalize_tuple


class DataGenerator:
    '''
    Generates batches of images with real-time data augmentation.

    Parameters
    ----------
    shape: tuple of int
        Shape of batch images (excluding the channel dimension).
    batch_size: int
        Batch size.
    transform_function: str or callable or None
        Function used for data augmentation. Typically you will set
        ``transform_function='rotate_and_flip'`` to apply combination of
        randomly selected image rotation and flipping.  Alternatively, you can
        specify an arbitrary transformation function which takes two input
        images (source and target) and returns transformed images. If
        ``transform_function=None``, no augmentation will be performed.
    intensity_threshold: float
        If ``intensity_threshold > 0``, pixels whose intensities are greater
        than this threshold will be considered as foreground.
    area_ratio_threshold: float between 0 and 1
        If ``intensity_threshold > 0``, the generator calculates the ratio of
        foreground pixels in a target patch, and rejects the patch if the ratio
        is smaller than this threshold.
    scale_factor: int != 0
        Scale factor for the target patch size. Positive and negative values
        mean up- and down-scaling respectively.
    '''
    def __init__(self,
                 shape,
                 batch_size,
                 transform_function='rotate_and_flip',
                 intensity_threshold=0.0,
                 area_ratio_threshold=0.0,
                 scale_factor=1):
        def rotate_and_flip(x, y, dim):
            if dim == 2:
                k = np.random.randint(0, 4)
                x, y = [None if v is None else np.rot90(v, k=k)
                        for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [None if v is None else np.fliplr(v)
                            for v in (x, y)]
                return x, y
            elif dim == 3:
                k = np.random.randint(0, 4)
                x, y = [None if v is None else np.rot90(v, k=k, axes=(1, 2))
                        for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [None if v is None else np.flip(v, axis=1)
                            for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [None if v is None else np.flip(v, axis=0)
                            for v in (x, y)]
                return x, y
            else:
                raise ValueError('Unsupported dimension')

        self._shape = tuple(shape)
        self._batch_size = batch_size

        dim = len(self._shape)

        if transform_function == 'rotate_and_flip':
            if shape[-2] != shape[-1]:
                raise ValueError(
                    'Patch shape must be square when using `rotate_and_flip`; '
                    f'Received shape: {shape}')
            self._transform_function = lambda x, y: rotate_and_flip(x, y, dim)
        elif callable(transform_function):
            self._transform_function = transform_function
        elif transform_function is None:
            self._transform_function = lambda x, y: (x, y)
        else:
            raise ValueError('Invalid transform function')

        self._intensity_threshold = intensity_threshold

        if not 0 <= area_ratio_threshold <= 1:
            raise ValueError('"area_ratio_threshold" must be between 0 and 1')
        self._area_threshold = area_ratio_threshold * np.prod(shape)

        self._scale_factor = normalize_tuple(scale_factor, dim, 'scale_factor')
        if any(not isinstance(f, int) or f == 0 for f in self._scale_factor):
            raise ValueError('"scale_factor" must be nonzero integer')

    class _Generator:
        def _scale(self, shape):
            return tuple(
                s * f if f > 0 else s // -f
                for s, f in zip(shape, self._scale_factor))

        def __init__(self,
                     x,
                     y,
                     shape,
                     transform_function,
                     intensity_threshold,
                     area_threshold,
                     scale_factor):
            self._transform_function = transform_function
            self._intensity_threshold = intensity_threshold
            self._area_threshold = area_threshold
            self._scale_factor = scale_factor

            for s, f, in zip(shape, self._scale_factor):
                if f < 0 and s % -f != 0:
                    raise ValueError(
                        'When downsampling, all elements in `shape` must be '
                        'divisible by the scale factor; '
                        f'Received shape: {shape}, '
                        f'scale factor: {self._scale_factor}')

            self._x, self._y = [
                list(m) if isinstance(m, (list, tuple)) else [m]
                for m in [x, y]]

            if self._y is not None and len(self._x) != len(self._y):
                raise ValueError(
                    'Different number of images are given: '
                    f'{len(self._x)} vs. {len(self._y)}')

            if len({m.dtype for m in self._x}) != 1:
                raise ValueError('All source images must be the same type')

            if self._y is not None and len({m.dtype for m in self._y}) != 1:
                raise ValueError('All target images must be the same type')

            for i in range(len(self._x)):
                if len(self._x[i].shape) == len(shape):
                    self._x[i] = self._x[i][..., np.newaxis]

                if len(self._x[i].shape) != len(shape) + 1:
                    raise ValueError(f'Source image must be {len(shape)}D')

                if self._x[i].shape[:-1] < shape:
                    raise ValueError(
                        'Source image must be larger than the patch size')

                if self._y is not None:
                    if len(self._y[i].shape) == len(shape):
                        self._y[i] = self._y[i][..., np.newaxis]

                    if len(self._y[i].shape) != len(shape) + 1:
                        raise ValueError(f'Target image must be {len(shape)}D')

                    expected_y_image_size = self._scale(self._x[i].shape[:-1])
                    if self._y[i].shape[:-1] != expected_y_image_size:
                        raise ValueError(
                            'Invalid target image size: '
                            f'expected {expected_y_image_size}, '
                            f'but received {self._y[i].shape[:-1]}')

            if len({m.shape[-1] for m in self._x}) != 1:
                raise ValueError(
                    'All source images must have the same number of channels')

            if (self._y is not None
                    and len({m.shape[-1] for m in self._y}) != 1):
                raise ValueError(
                    'All target images must have the same number of channels')

            output_signature_x = tf.TensorSpec(
                (*shape, self._x[0].shape[-1]), self._x[0].dtype)

            if self._y is None:
                self.output_signature = (output_signature_x,)
            else:
                self.output_signature = (
                    output_signature_x,
                    tf.TensorSpec(
                        (*self._scale(shape), self._y[0].shape[-1]),
                        self._y[0].dtype))

        def __iter__(self):
            while True:
                for _ in range(512):
                    i = np.random.randint(0, len(self._x))

                    tl = [
                        np.random.randint(0, a - b + 1)
                        for a, b in zip(
                            self._x[i].shape, self.output_signature[0].shape)]

                    patch_x_roi = tuple(
                        slice(a, a + b)
                        for a, b in zip(tl, self.output_signature[0].shape))
                    patch_x = np.copy(self._x[i][patch_x_roi])

                    if self._y is not None:
                        patch_y_roi = tuple(
                            slice(a, a + b) for a, b in
                            zip(self._scale(tl),
                                self.output_signature[1].shape))
                        patch_y = np.copy(self._y[i][patch_y_roi])

                    if self._intensity_threshold > 0:
                        foreground_area = np.count_nonzero(
                            (patch_x if self._y is None else patch_y)
                            > self._intensity_threshold)
                        if foreground_area < self._area_threshold:
                            continue

                    break

                else:
                    warnings.warn(
                        'Failed to sample a valid patch',
                        RuntimeWarning,
                        stacklevel=3)

                if self._y is None:
                    yield self._transform_function(patch_x, None)[0]
                else:
                    yield self._transform_function(patch_x, patch_y)

    def flow(self, x, y=None, /):
        '''
        Returns a `tf.data.Dataset` object which generates batches
        infinitely.

        Parameters
        ----------
        x: array_like or list of array_like
            Source image(s).
        y: array_like or list of array_like or None
            Target image(s).

        Returns
        -------
        tf.data.Dataset
            `tf.data.Dataset` yielding image patches.
        '''
        gen = self._Generator(
            x,
            y,
            self._shape,
            self._transform_function,
            self._intensity_threshold,
            self._area_threshold,
            self._scale_factor)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA)

        return (
            tf.data.Dataset
            .from_generator(lambda: gen, output_signature=gen.output_signature)
            .with_options(options)
            .batch(self._batch_size)
            .repeat()
            .prefetch(tf.data.AUTOTUNE))
