# Copyright 2020 DRVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

from rcan.utils import apply, get_model_path, normalize, load_model

import argparse
import itertools
import numpy as np
import pathlib
import tifffile

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', type=str, required=True)
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-g', '--ground_truth', type=str)
parser.add_argument('-b', '--bpp', type=int, choices=[8, 16, 32], default=32)
args = parser.parse_args()

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
model = load_model(str(model_path), input_shape=None)

overlap_shape = [
    max(1, x // 8) if x > 2 else 0 for x in model.input.shape.as_list()[1:-1]]

for raw_file, gt_file in data:
    print('Loading raw image from', raw_file)
    raw = normalize(tifffile.imread(str(raw_file)))

    print('Applying model')
    restored = apply(model, raw, overlap_shape=overlap_shape, verbose=True)

    result = [raw, restored]

    if gt_file is not None:
        print('Loading ground truth image from', gt_file)
        gt = tifffile.imread(str(gt_file))
        if raw.shape == gt.shape:
            result.append(normalize(gt))
        else:
            print('Ground truth image discarded due to image shape mismatch')

    result = np.stack(result)
    if result.ndim == 4:
        result = np.transpose(result, (1, 0, 2, 3))

    if args.bpp == 8:
        result = np.clip(255 * result, 0, 255).astype('uint8')
    elif args.bpp == 16:
        result = np.clip(65535 * result, 0, 65535).astype('uint16')

    if output_path.is_dir():
        output_file = output_path / raw_file.name
    else:
        output_file = output_path

    print('Saving output image to', output_file)
    tifffile.imwrite(str(output_file), result, imagej=True)
