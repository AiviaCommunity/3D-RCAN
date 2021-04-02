# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import keras
from keras.utils.conv_utils import normalize_tuple
import numpy as np
import warnings


class DataGenerator:
    '''
    Generates batches of image pairs with real-time data augmentation.

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
                x, y = [np.rot90(v, k=k) for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [np.fliplr(v) for v in (x, y)]
                return x, y
            elif dim == 3:
                k = np.random.randint(0, 4)
                x, y = [np.rot90(v, k=k, axes=(1, 2)) for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [np.flip(v, axis=1) for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [np.flip(v, axis=0) for v in (x, y)]
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

    class _Sequence(keras.utils.Sequence):
        def _scale(self, shape):
            return tuple(
                s * f if f > 0 else s // -f
                for s, f in zip(shape, self._scale_factor))

        def __init__(self,
                     x,
                     y,
                     batch_size,
                     shape,
                     transform_function,
                     intensity_threshold,
                     area_threshold,
                     scale_factor):
            self._batch_size = batch_size
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

            if len(self._x) != len(self._y):
                raise ValueError(
                    'Different number of images are given: '
                    f'{len(self._x)} vs. {len(self._y)}')

            if len({m.dtype for m in self._x}) != 1:
                raise ValueError('All source images must be the same type')
            if len({m.dtype for m in self._y}) != 1:
                raise ValueError('All target images must be the same type')

            for i in range(len(self._x)):
                if len(self._x[i].shape) == len(shape):
                    self._x[i] = self._x[i][..., np.newaxis]

                if len(self._y[i].shape) == len(shape):
                    self._y[i] = self._y[i][..., np.newaxis]

                if len(self._x[i].shape) != len(shape) + 1:
                    raise ValueError(f'Source image must be {len(shape)}D')

                if len(self._y[i].shape) != len(shape) + 1:
                    raise ValueError(f'Target image must be {len(shape)}D')

                if self._x[i].shape[:-1] < shape:
                    raise ValueError(
                        'Source image must be larger than the patch size')

                expected_y_image_size = self._scale(self._x[i].shape[:-1])
                if self._y[i].shape[:-1] != expected_y_image_size:
                    raise ValueError('Invalid target image size: '
                                     f'expected {expected_y_image_size}, '
                                     f'but received {self._y[i].shape[:-1]}')

            if len({m.shape[-1] for m in self._x}) != 1:
                raise ValueError(
                    'All source images must have the same number of channels')
            if len({m.shape[-1] for m in self._y}) != 1:
                raise ValueError(
                    'All target images must have the same number of channels')

            self._batch_x = np.zeros(
                (batch_size, *shape, self._x[0].shape[-1]),
                dtype=self._x[0].dtype)
            self._batch_y = np.zeros(
                (batch_size, *self._scale(shape), self._y[0].shape[-1]),
                dtype=self._y[0].dtype)

        def __len__(self):
            return 1  # return a dummy value

        def __next__(self):
            return self.__getitem__(0)

        def __getitem__(self, _):
            for i in range(self._batch_size):
                for _ in range(512):
                    j = np.random.randint(0, len(self._x))

                    tl = [np.random.randint(0, a - b + 1)
                          for a, b in zip(
                              self._x[j].shape, self._batch_x.shape[1:])]
                    x = np.copy(self._x[j][tuple(
                        [slice(a, a + b) for a, b in zip(
                            tl, self._batch_x.shape[1:])])])

                    y = np.copy(self._y[j][tuple(
                        [slice(a, a + b) for a, b in zip(
                            self._scale(tl), self._batch_y.shape[1:])])])

                    if (self._intensity_threshold <= 0.0 or
                            np.count_nonzero(y > self._intensity_threshold)
                            >= self._area_threshold):
                        break
                else:
                    warnings.warn(
                        'Failed to sample a valid patch',
                        RuntimeWarning,
                        stacklevel=3)

                self._batch_x[i], self._batch_y[i] = \
                    self._transform_function(x, y)

            return self._batch_x, self._batch_y

    def flow(self, x, y):
        '''
        Returns a `keras.utils.Sequence` object which generates batches
        infinitely. It can be used as an input generator for
        `keras.models.Model.fit_generator()`.

        Parameters
        ----------
        x: array_like or list of array_like
            Source image(s).
        y: array_like or list of array_like
            Target image(s).

        Returns
        -------
        keras.utils.Sequence
            `keras.utils.Sequence` object which generates tuples of source and
            target image patches.
        '''
        return self._Sequence(x,
                              y,
                              self._batch_size,
                              self._shape,
                              self._transform_function,
                              self._intensity_threshold,
                              self._area_threshold,
                              self._scale_factor)
