# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

from rcan.utils import (
    apply,
    convert_to_multi_gpu_model,
    get_model_path,
    load_image,
    load_model,
    rescale,
    save_imagej_hyperstack,
    save_ome_tiff)

import argparse
import itertools
import numpy as np
import pathlib
import scipy.ndimage


def tuple_of_ints(string):
    return tuple(int(s) for s in string.split(','))


def percentile(x):
    x = float(x)
    if 0.0 <= x <= 100.0:
        return x
    else:
        raise argparse.ArgumentTypeError(f'{x} not in range [0.0, 100.0]')


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', type=str, required=True)
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-g', '--ground_truth', type=str)
parser.add_argument('-d', '--input_data_format', type=str, default='')
parser.add_argument('-D', '--ground_truth_data_format', type=str, default='')
parser.add_argument(
    '-f', '--output_tiff_format', type=str,
    choices=['imagej', 'ome'], default='imagej')
parser.add_argument('-b', '--bpp', type=int, choices=[8, 16, 32], default=32)
parser.add_argument('-B', '--block_shape', type=tuple_of_ints)
parser.add_argument('-O', '--block_overlap_shape', type=tuple_of_ints)
parser.add_argument('--p_min', type=percentile, default=2.0)
parser.add_argument('--p_max', type=percentile, default=99.9)
parser.add_argument('--rescale', action='store_true')
parser.add_argument(
    '--normalize_output_range_between_zero_and_one', action='store_true')
args = parser.parse_args()

if args.rescale and args.normalize_output_range_between_zero_and_one:
    raise ValueError(
        'You cannot set both `rescale` and '
        '`normalize_output_range_between_zero_and_one` at the same time')

input_path = pathlib.Path(args.input)
output_path = pathlib.Path(args.output)

if input_path.is_dir() and not output_path.exists():
    print('Creating output directory', output_path)
    output_path.mkdir(parents=True)

if input_path.is_dir() != output_path.is_dir():
    raise ValueError('Mismatch between input and output path types')

if args.ground_truth is None:
    gt_path = None
else:
    gt_path = pathlib.Path(args.ground_truth)
    if input_path.is_dir() != gt_path.is_dir():
        raise ValueError('Mismatch between input and ground truth path types')

if input_path.is_dir():
    raw_files = sorted(input_path.glob('*.tif'))

    if gt_path is None:
        data = itertools.zip_longest(raw_files, [])
    else:
        gt_files = sorted(gt_path.glob('*.tif'))

        if len(raw_files) != len(gt_files):
            raise ValueError(
                'Mismatch between raw and ground truth file counts '
                f'({len(raw_files)} vs. {len(gt_files)})')

        data = zip(raw_files, gt_files)
else:
    data = [(input_path, gt_path)]

model_path = get_model_path(args.model_dir)
print('Loading model from', model_path)
model = convert_to_multi_gpu_model(
    load_model(str(model_path), input_shape=args.block_shape))
image_ndim = len(model.input_shape) - 2

if args.block_overlap_shape is None:
    overlap_shape = [
        max(1, x // 8) if x > 2 else 0
        for x in model.input.shape.as_list()[1:-1]]
else:
    overlap_shape = args.block_overlap_shape

for raw_file, gt_file in data:
    print('Loading raw image from', raw_file)
    raw = load_image(
        str(raw_file),
        image_ndim,
        args.input_data_format,
        args.p_min,
        args.p_max)

    print('Applying model')
    restored = apply(model, raw, overlap_shape=overlap_shape, verbose=True)

    if raw.shape[:-1] != restored.shape[:-1]:
        print('Upsampling raw image')
        raw_upsampled = np.empty(
            (*restored.shape[:-1], raw.shape[-1]), dtype='float32')
        for c in range(raw.shape[-1]):
            raw_upsampled[..., c] = scipy.ndimage.zoom(
                raw[..., c],
                [b / a for a, b in zip(raw.shape[:-1], restored.shape[:-1])],
                order=0)
        raw = raw_upsampled

    if gt_file is None:
        result = [raw, restored]
    else:
        print('Loading ground truth image from', gt_file)
        gt = load_image(
            str(gt_file),
            image_ndim,
            args.ground_truth_data_format,
            args.p_min,
            args.p_max)

        if args.rescale:
            print('Performing affine rescaling')
            for c in range(gt.shape[-1]):
                restored[..., c] = rescale(restored[..., c], gt[..., c])

        result = [raw, restored, gt]

    if args.normalize_output_range_between_zero_and_one:
        print('Normalizing output range to [0, 1]')
        for m in result:
            for c in range(m.shape[-1]):
                max_val, min_val = m[..., c].max(), m[..., c].min()
                diff = max_val - min_val
                if diff > 0:
                    m[..., c] = (m[..., c] - min_val) / diff
                else:
                    m[..., c] = 0

    result = np.concatenate(result, axis=-1)

    if args.bpp == 8:
        result = np.clip(255 * result, 0, 255).astype('uint8')
    elif args.bpp == 16:
        result = np.clip(65535 * result, 0, 65535).astype('uint16')

    if output_path.is_dir():
        output_file = output_path / raw_file.name
    else:
        output_file = output_path

    print('Saving output image to', output_file)
    if args.output_tiff_format == 'ome':
        def generate_channel_names(name, n):
            channel_names = []
            if n == 1:
                channel_names.append(name)
            else:
                for i in range(n):
                    channel_names.append(f'{name} Channel {i + 1}')
            return channel_names

        channel_names = generate_channel_names('Raw', raw.shape[-1])
        channel_names.extend(
            generate_channel_names('Restored', restored.shape[-1]))
        if gt_file is not None:
            channel_names.extend(
                generate_channel_names('Ground Truth', gt.shape[-1]))

        save_ome_tiff(
            str(output_file),
            result,
            channel_names=channel_names,
            data_format='YXC' if image_ndim == 2 else 'ZYXC')
    else:
        save_imagej_hyperstack(
            str(output_file),
            result,
            'YXC' if image_ndim == 2 else 'ZYXC')
