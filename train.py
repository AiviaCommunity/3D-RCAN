# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

from rcan.callbacks import ModelCheckpoint, TqdmCallback
from rcan.data_generator import DataGenerator
from rcan.losses import mae, mse
from rcan.metrics import psnr, ssim
from rcan.model import build_rcan
from rcan.utils import (
    convert_to_multi_gpu_model,
    get_gpu_count,
    normalize,
    staircase_exponential_decay)

import argparse
import itertools
import json
import jsonschema
import keras
import numpy as np
import pathlib
import tifffile


def load_data(config, data_type):
    image_pair_list = config.get(data_type + '_image_pairs', [])

    if data_type + '_data_dir' in config:
        raw_dir, gt_dir = [
            pathlib.Path(config[data_type + '_data_dir'][t])
            for t in ['raw', 'gt']]

        raw_files, gt_files = [
            sorted(d.glob('*.tif')) for d in [raw_dir, gt_dir]]

        if not raw_files:
            raise RuntimeError(f'No TIFF file found in {raw_dir}')

        if len(raw_files) != len(gt_files):
            raise RuntimeError(
                f'"{raw_dir}" and "{gt_dir}" must contain the same number of '
                'TIFF files')

        for raw_file, gt_file in zip(raw_files, gt_files):
            image_pair_list.append({'raw': str(raw_file), 'gt': str(gt_file)})

    if not image_pair_list:
        return None

    print(f'Loading {data_type} data')

    data = []
    for p in image_pair_list:
        raw_file, gt_file = [p[t] for t in ['raw', 'gt']]

        print('  - raw:', raw_file)
        print('    gt:', gt_file)

        raw, gt = [tifffile.imread(p[t]) for t in ['raw', 'gt']]

        if raw.shape != gt.shape:
            raise ValueError(
                'Raw and GT images must be the same size: '
                f'{p["raw"]} {raw.shape} vs. {p["gt"]} {gt.shape}')

        data.append([normalize(m) for m in [raw, gt]])

    return data


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
args = parser.parse_args()

schema = {
    'type': 'object',
    'properties': {
        'training_image_pairs': {'$ref': '#/definitions/image_pairs'},
        'validation_image_pairs': {'$ref': '#/definitions/image_pairs'},
        'training_data_dir': {'$ref': '#/definitions/raw_gt_pair'},
        'validation_data_dir': {'$ref': '#/definitions/raw_gt_pair'},
        'input_shape': {
            'type': 'array',
            'items': {'type': 'integer', 'minimum': 1},
            'minItems': 2,
            'maxItems': 3
        },
        'num_channels': {'type': 'integer', 'minimum': 1},
        'num_residual_blocks': {'type': 'integer', 'minimum': 1},
        'num_residual_groups': {'type': 'integer', 'minimum': 1},
        'channel_reduction': {'type': 'integer', 'minimum': 1},
        'epochs': {'type': 'integer', 'minimum': 1},
        'steps_per_epoch': {'type': 'integer', 'minimum': 1},
        'data_augmentation': {'type': 'boolean'},
        'intensity_threshold': {'type': 'number'},
        'area_ratio_threshold': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'initial_learning_rate': {'type': 'number', 'minimum': 1e-6},
        'loss': {'type': 'string', 'enum': ['mae', 'mse']},
        'metrics': {
            'type': 'array',
            'items': {'type': 'string', 'enum': ['psnr', 'ssim']}
        }
    },
    'additionalProperties': False,
    'anyOf': [
        {'required': ['training_image_pairs']},
        {'required': ['training_data_dir']}
    ],
    'definitions': {
        'raw_gt_pair': {
            'type': 'object',
            'properties': {
                'raw': {'type': 'string'},
                'gt': {'type': 'string'},
            }
        },
        'image_pairs': {
            'type': 'array',
            'items': {'$ref': '#/definitions/raw_gt_pair'},
            'minItems': 1
        }
    }
}

with open(args.config) as f:
    config = json.load(f)

jsonschema.validate(config, schema)
config.setdefault('epochs', 300)
config.setdefault('steps_per_epoch', 256)
config.setdefault('num_channels', 32)
config.setdefault('num_residual_blocks', 3)
config.setdefault('num_residual_groups', 5)
config.setdefault('channel_reduction', 8)
config.setdefault('data_augmentation', True)
config.setdefault('intensity_threshold', 0.25)
config.setdefault('area_ratio_threshold', 0.5)
config.setdefault('initial_learning_rate', 1e-4)
config.setdefault('loss', 'mae')
config.setdefault('metrics', ['psnr'])

training_data = load_data(config, 'training')
validation_data = load_data(config, 'validation')

ndim = training_data[0][0].ndim

for p in itertools.chain(training_data, validation_data or []):
    if p[0].ndim != ndim:
        raise ValueError('All images must have the same number of dimensions')

if 'input_shape' in config:
    input_shape = config['input_shape']
    if len(input_shape) != ndim:
        raise ValueError(
            f'`input_shape` must be a {ndim}D array; received: {input_shape}')
else:
    input_shape = (16, 256, 256) if ndim == 3 else (256, 256)

for p in itertools.chain(training_data, validation_data or []):
    input_shape = np.minimum(input_shape, p[0].shape)

print('Building RCAN model')
print('  - input_shape =', input_shape)
for s in ['num_channels',
          'num_residual_blocks',
          'num_residual_groups',
          'channel_reduction']:
    print(f'  - {s} =', config[s])

model = build_rcan(
    (*input_shape, 1),
    num_channels=config['num_channels'],
    num_residual_blocks=config['num_residual_blocks'],
    num_residual_groups=config['num_residual_groups'],
    channel_reduction=config['channel_reduction'])

gpus = get_gpu_count()
model = convert_to_multi_gpu_model(model, gpus)

model.compile(
    optimizer=keras.optimizers.Adam(lr=config['initial_learning_rate']),
    loss={'mae': mae, 'mse': mse}[config['loss']],
    metrics=[{'psnr': psnr, 'ssim': ssim}[m] for m in config['metrics']])

data_gen = DataGenerator(
    input_shape,
    gpus,
    transform_function=(
        'rotate_and_flip' if config['data_augmentation'] else None),
    intensity_threshold=config['intensity_threshold'],
    area_ratio_threshold=config['area_ratio_threshold'])

training_data = data_gen.flow(*list(zip(*training_data)))

if validation_data is not None:
    validation_data = data_gen.flow(*list(zip(*validation_data)))
    checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
else:
    checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'

steps_per_epoch = config['steps_per_epoch'] // gpus
validation_steps = None if validation_data is None else steps_per_epoch

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print('Training RCAN model')
model.fit_generator(
    training_data,
    epochs=config['epochs'],
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_data,
    validation_steps=validation_steps,
    verbose=0,
    callbacks=[
        keras.callbacks.LearningRateScheduler(
            staircase_exponential_decay(config['epochs'] // 4)),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir),
            write_graph=False),
        ModelCheckpoint(
            str(output_dir / checkpoint_filepath),
            monitor='loss' if validation_data is None else 'val_loss',
            save_best_only=True),
        TqdmCallback()
    ])
