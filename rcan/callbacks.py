# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import functools
import keras
import warnings

import tqdm
from tqdm.utils import IS_WIN
from tqdm.keras import TqdmCallback as _TqdmCallback

from .utils import save_model


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s '
                                  'available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to '
                                  '%0.5f, saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        save_model(
                            filepath, self.model, self.save_weights_only)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f'
                                  % (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s'
                          % (epoch + 1, filepath))
                save_model(filepath, self.model, self.save_weights_only)


class TqdmCallback(_TqdmCallback):
    def __init__(self):
        super().__init__(
            tqdm_class=functools.partial(
                tqdm.tqdm, dynamic_ncols=True, ascii=IS_WIN))
        self.on_batch_end = self.bar2callback(
            self.batch_bar, pop=['batch', 'size'])
