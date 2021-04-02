# Copyright 2021 SVision Technologies LLC.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import keras


def mae(y_true, y_pred):
    '''Mean absolute error'''
    return keras.losses.mae(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])


def mse(y_true, y_pred):
    '''Mean squared error'''
    return keras.losses.mse(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])
