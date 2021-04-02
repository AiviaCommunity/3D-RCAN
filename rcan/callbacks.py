# Copyright 2020 DRVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import warnings

import keras
import numpy as np

from .utils import save_model


class ModelCheckpoint(keras.callbacks.Callback):
    '''
    Save the model after every epoch.

    `filepath` can contain named formatting options, which will be filled the
    value of `epoch` and keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and the
    validation loss in the filename.

    Parameters
    ----------
    filepath: str
        Path to save the model file.
    monitor: str
        Quantity to monitor.
    verbose: int
        Verbosity mode, 0 or 1.
    save_best_only: bool
        If true, the latest best model according to the quantity monitored will
        not be overwritten.
    mode: str
        One of {``auto``, ``min``, ``max``}. If ``save_best_only=True``, the
        decision to overwrite the current save file is made based on either the
        maximization or the minimization of the monitored quantity. For
        ``val_acc``, this should be ``max``, for ``val_loss`` this should be
        ``min``, etc. In ``auto`` mode, the direction is automatically inferred
        from the name of the monitored quantity.
    save_weights_only: bool
        If True, then only the model's weights will be saved, else the full
        model is saved.
    period: int
        Interval (number of epochs) between checkpoints.
    '''

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

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
