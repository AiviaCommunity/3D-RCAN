# Copyright 2020 DRVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) International Public License (https://creativecommons.org/licenses/by-nc/4.0/)

from rcan.data_generator import DataGenerator
from rcan.losses import mae
from rcan.metrics import psnr
from rcan.model import build_rcan
from rcan.utils import normalize, staircase_exponential_decay

import argparse
import functools
import itertools
import json
import jsonschema
import keras
import numpy as np
import pathlib
import tifffile

from tqdm import tqdm as std_tqdm
from tqdm.keras import TqdmCallback
from tqdm.utils import IS_WIN
tqdm = functools.partial(std_tqdm, dynamic_ncols=True, ascii=IS_WIN)


def load_data(image_pair_list):
    data = []
    for p in tqdm(image_pair_list):
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
        'training_data': {'$ref': '#/definitions/image_pairs'},
        'validation_data': {'$ref': '#/definitions/image_pairs'},
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
        'steps_per_epoch': {'type': 'integer', 'minimum': 1}
    },
    'additionalProperties': False,
    'required': ['training_data'],
    'definitions': {
        'image_pairs': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'raw': {'type': 'string'},
                    'gt': {'type': 'string'},
                }
            },
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

print('Loading training data')
training_data = load_data(config['training_data'])

if 'validation_data' in config:
    print('Loading validation data')
    validation_data = load_data(config['validation_data'])
else:
    validation_data = None

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

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    loss=mae,
    metrics=[psnr])

data_gen = DataGenerator(
    input_shape,
    1,
    intensity_threshold=0.25,
    area_ratio_threshold=0.05)

training_data = data_gen.flow(*list(zip(*training_data)))

if validation_data is not None:
    validation_data = data_gen.flow(*list(zip(*validation_data)))
    checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
else:
    checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'

output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print('Training RCAN model')
model.fit_generator(
    training_data,
    epochs=config['epochs'],
    steps_per_epoch=config['steps_per_epoch'],
    validation_data=validation_data,
    validation_steps=config['steps_per_epoch'],
    verbose=0,
    callbacks=[
        keras.callbacks.LearningRateScheduler(
            staircase_exponential_decay(config['epochs'] // 4)),
        keras.callbacks.ModelCheckpoint(
            str(output_dir / checkpoint_filepath),
            monitor='loss' if validation_data is None else 'val_loss',
            save_best_only=True),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir),
            write_graph=False),
        TqdmCallback(tqdm_class=tqdm)
    ])
