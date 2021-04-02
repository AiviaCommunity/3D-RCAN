# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import fractions
import h5py
import itertools
import json
import keras
import numexpr
import numpy as np
import pathlib
import re
import tifffile
import tqdm
import tqdm.utils

from tensorflow.python.client.device_lib import list_local_devices


def get_gpu_count():
    '''Returns the number of available GPUs.'''
    return len([x for x in list_local_devices() if x.device_type == 'GPU'])


def is_multi_gpu_model(model):
    '''Checks if the model supports multi-GPU data parallelism.'''
    return hasattr(model, 'is_multi_gpu_model') and model.is_multi_gpu_model


def convert_to_multi_gpu_model(model, gpus=None):
    '''
    Converts a model into a multi-GPU version if possible.

    Parameters
    ----------
    model: keras.Model
        Model to be converted.
    gpus: int or None
        Number of GPUs used to create model replicas. If None, all GPUs
        available on the device will be used.

    Returns
    -------
    keras.Model
        Multi-GPU model.
    '''

    gpus = gpus or get_gpu_count()

    if gpus <= 1 or is_multi_gpu_model(model):
        return model

    multi_gpu_model = keras.utils.multi_gpu_model(
        model, gpus=gpus, cpu_relocation=True)

    # copy weights
    multi_gpu_model.layers[
        -(len(multi_gpu_model.outputs) + 1)].set_weights(model.get_weights())

    setattr(multi_gpu_model, 'is_multi_gpu_model', True)
    setattr(multi_gpu_model, 'gpus', gpus)

    return multi_gpu_model


def normalize(image, p_min=2, p_max=99.9, dtype='float32'):
    '''
    Normalizes the image intensity so that the `p_min`-th and the `p_max`-th
    percentiles are converted to 0 and 1 respectively.

    References
    ----------
    Content-Aware Image Restoration: Pushing the Limits of Fluorescence
    Microscopy
    https://doi.org/10.1038/s41592-018-0216-7
    '''
    low, high = np.percentile(image, (p_min, p_max))
    return numexpr.evaluate(
        '(image - low) / (high - low + 1e-6)').astype(dtype)


def rescale(restored, gt):
    '''Affine rescaling to minimize the MSE to the GT'''
    cov = np.cov(restored.flatten(), gt.flatten())
    a = cov[0, 1] / cov[0, 0]
    b = gt.mean() - a * restored.mean()
    return a * restored + b


def staircase_exponential_decay(n):
    '''
    Returns a scheduler function to drop the learning rate by half
    every `n` epochs.
    '''
    return lambda epoch, lr: lr / 2 if epoch != 0 and epoch % n == 0 else lr


def save_model(filename, model, weights_only=False):
    if is_multi_gpu_model(model):
        m = model.layers[-(len(model.outputs) + 1)]
    else:
        m = model

    if weights_only:
        m.save_weights(filename, overwrite=True)
    else:
        m.save(filename, overwrite=True)


def get_model_path(directory, model_type='best'):
    '''
    Finds a model file in the given directory.

    Parameters
    ----------
    directory: str
        Directory where model files are located.
    model_type: str
        One of 'best' or 'newest'.

    Returns
    -------
    pathlib.Path
        Path of the model file.
    '''

    if model_type not in ('best', 'newest'):
        raise ValueError('`model_type` must be either "best" or "newest"')

    def get_value(path):
        match = re.match(
            r'weights_(\d+)_([+-]?\d+(?:\.\d+)?)\.hdf5', path.name)
        if match:
            return float(match.group(2 if model_type == 'best' else 1))
        else:
            return np.inf

    try:
        files = pathlib.Path(directory).glob('*.hdf5')
        return (min if model_type == 'best' else max)(files, key=get_value)
    except ValueError:
        raise RuntimeError(f'Unable to find model file in {directory}')


def load_model(filename, input_shape=None):
    '''
    Loads a model from a file.

    Parameters
    ----------
    filename: str
        Model file to be loaded.
    input_shape: tuple of int or None
        Optional parameter to specify model's input shape (excluding the
        channel dimension).

    Returns
    -------
    keras.Model
        Keras model instance.
    '''

    with h5py.File(filename, mode='r') as f:
        model_config = f.attrs.get('model_config')
        model_config = json.loads(model_config.decode('utf-8'))

        # overwrite model's input shape
        if input_shape is not None:
            for layer in model_config['config']['layers']:
                if layer['class_name'] == 'InputLayer':
                    shape = layer['config']['batch_input_shape']
                    if len(shape) - 2 != len(input_shape):
                        raise ValueError(
                            f'Input shape must be {len(shape) - 2}D; '
                            f'Received input shape: {input_shape}')
                    shape[1:-1] = input_shape

        model = keras.models.model_from_config(model_config)
        model.load_weights(filename)
        return model


def apply(model, data, overlap_shape=None, verbose=False):
    '''
    Applies a model to an input image. The input image stack is split into
    sub-blocks with model's input size, then the model is applied block by
    block. The sizes of input and output images are assumed to be the same
    while they can have different numbers of channels.

    Parameters
    ----------
    model: keras.Model
        Keras model.
    data: array_like or list of array_like
        Input data. Either an image or a list of images.
    overlap_shape: tuple of int or None
        Overlap size between sub-blocks in each dimension. If not specified,
        a default size ((32, 32) for 2D and (2, 32, 32) for 3D) is used.
        Results at overlapped areas are blended together linearly.

    Returns
    -------
    ndarray
        Result image.
    '''

    model_input_image_shape = tuple(model.input.shape.as_list()[1:-1])
    model_output_image_shape = tuple(model.output.shape.as_list()[1:-1])

    if len(model_input_image_shape) != len(model_output_image_shape):
        raise NotImplementedError

    image_dim = len(model_input_image_shape)
    num_input_channels = model.input.shape[-1].value
    num_output_channels = model.output.shape[-1].value

    scale_factor = tuple(
        fractions.Fraction(o, i) for i, o in zip(
            model_input_image_shape, model_output_image_shape))

    def _scale_tuple(t):
        t = [v * f for v, f in zip(t, scale_factor)]

        if not all([v.denominator == 1 for v in t]):
            raise NotImplementedError

        return tuple(v.numerator for v in t)

    def _scale_roi(roi):
        roi = [slice(r.start * f, r.stop * f)
               for r, f in zip(roi, scale_factor)]

        if not all([
                r.start.denominator == 1 and
                r.stop.denominator == 1 for r in roi]):
            raise NotImplementedError

        return tuple(slice(r.start.numerator, r.stop.numerator) for r in roi)

    if overlap_shape is None:
        if image_dim == 2:
            overlap_shape = (32, 32)
        elif image_dim == 3:
            overlap_shape = (2, 32, 32)
        else:
            raise NotImplementedError
    elif len(overlap_shape) != image_dim:
        raise ValueError(f'Overlap shape must be {image_dim}D; '
                         f'Received shape: {overlap_shape}')

    step_shape = tuple(
        m - o for m, o in zip(
            model_input_image_shape, overlap_shape))

    block_weight = np.ones(
        [m - 2 * o for m, o
         in zip(model_output_image_shape, _scale_tuple(overlap_shape))],
        dtype=np.float32)

    block_weight = np.pad(
        block_weight,
        [(o + 1, o + 1) for o in _scale_tuple(overlap_shape)],
        'linear_ramp'
    )[(slice(1, -1),) * image_dim]

    batch_size = model.gpus if is_multi_gpu_model(model) else 1
    batch = np.zeros(
        (batch_size, *model_input_image_shape, num_input_channels),
        dtype=np.float32)

    if isinstance(data, (list, tuple)):
        input_is_list = True
    else:
        data = [data]
        input_is_list = False

    result = []

    for image in data:
        # add the channel dimension if necessary
        if len(image.shape) == image_dim:
            image = image[..., np.newaxis]

        if (len(image.shape) != image_dim + 1
                or image.shape[-1] != num_input_channels):
            raise ValueError(f'Input image must be {image_dim}D with '
                             f'{num_input_channels} channels; '
                             f'Received image shape: {image.shape}')

        input_image_shape = image.shape[:-1]
        output_image_shape = _scale_tuple(input_image_shape)

        applied = np.zeros(
            (*output_image_shape, num_output_channels), dtype=np.float32)
        sum_weight = np.zeros(output_image_shape, dtype=np.float32)

        num_steps = tuple(
            i // s + (i % s != 0)
            for i, s in zip(input_image_shape, step_shape))

        # top-left corner of each block
        blocks = list(itertools.product(
            *[np.arange(n) * s for n, s in zip(num_steps, step_shape)]))

        for chunk_index in tqdm.trange(
                0, len(blocks), batch_size, disable=not verbose,
                dynamic_ncols=True, ascii=tqdm.utils.IS_WIN):
            rois = []
            for batch_index, tl in enumerate(
                    blocks[chunk_index:chunk_index + batch_size]):
                br = [min(t + m, i) for t, m, i
                      in zip(tl, model_input_image_shape, input_image_shape)]
                r1, r2 = zip(
                    *[(slice(s, e), slice(0, e - s)) for s, e in zip(tl, br)])

                m = image[r1]
                if model_input_image_shape != m.shape[-1]:
                    pad_width = [(0, b - s) for b, s
                                 in zip(model_input_image_shape, m.shape[:-1])]
                    pad_width.append((0, 0))
                    m = np.pad(m, pad_width, 'reflect')

                batch[batch_index] = m
                rois.append((r1, r2))

            p = model.predict(batch, batch_size=batch_size)

            for batch_index in range(len(rois)):
                for channel in range(num_output_channels):
                    p[batch_index, ..., channel] *= block_weight

                r1, r2 = [_scale_roi(roi) for roi in rois[batch_index]]

                applied[r1] += p[batch_index][r2]
                sum_weight[r1] += block_weight[r2]

        for channel in range(num_output_channels):
            applied[..., channel] /= sum_weight

        if applied.shape[-1] == 1:
            applied = applied[..., 0]

        result.append(applied)

    return result if input_is_list else result[0]


def save_imagej_hyperstack(filename, image):
    assert image.ndim in [3, 4]
    if image.ndim == 4:
        image = np.transpose(image, (1, 0, 2, 3))

    tifffile.imwrite(str(filename), image, imagej=True)


def save_ome_tiff(filename, image):
    assert image.ndim in [3, 4]
    image = np.expand_dims(image, (1, 2) if image.ndim == 3 else 1)
    c, t, z, y, x = image.shape

    pixel_type = {
        np.dtype('uint8'): 'Uint8',
        np.dtype('uint16'): 'Uint16',
        np.dtype('float32'): 'Float'
    }[image.dtype]

    channel_names = ['Raw', 'Restored', 'Ground Truth']
    lsid_base = 'ome.drvtechnologies.com:'

    channel_info = ''
    for i, name in enumerate(channel_names[:c]):
        channel_info += f'''\
    <ChannelInfo Name="{name}" ID="{lsid_base}ChannelInfo:{i + 3}">
      <ChannelComponent Index="{i}" Pixels="{lsid_base}Pixels:2"/>
    </ChannelInfo>
'''
    description = f'''\
<OME xmlns="http://www.openmicroscopy.org/XMLschemas/OME/FC/ome.xsd">
  <Image Name="Unnamed [{pixel_type} {x}x{y}x{z}x{t} Channels]"
         ID="{lsid_base}Image:1">
{channel_info}\
    <Pixels DimensionOrder="XYZTC" PixelType="{pixel_type}"
            SizeX="{x}" SizeY="{y}" SizeZ="{z}" SizeT="{t}" SizeC="{c}"
            BigEndian="false" ID="{lsid_base}Pixels:2">
      <TiffData IFD="0" NumPlanes="{z * c * t}"/>
    </Pixels>
  </Image>
</OME>
'''

    tifffile.imwrite(
        filename,
        data=image,
        description=description,
        metadata=None)


def save_tiff(filename, image, format):
    {
        'imagej': save_imagej_hyperstack,
        'ome': save_ome_tiff
    }[format](filename, image)
