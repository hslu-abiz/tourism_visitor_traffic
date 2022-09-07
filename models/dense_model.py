# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 18.11.2019.
#
from typing import Callable, Optional, Sequence, Union

import tensorflow as tf


class DenseModel(tf.keras.Model):

    def __new__(
            cls,
            name: str,
            input_shape: Sequence[Optional[int]],
            outputs: Sequence[str],
            batch_size: Optional[int] = None,
            activation: Union[str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
            architecture: Sequence[int] = tuple(),
            dropout: float = 0.,
            dropout_before_last: bool = False,
            l1_regularization: float = 0.,
            seed: int = 42,
            dtype: tf.keras.backend.dtype = tf.float32,
    ):
        regularizer = None
        if l1_regularization != 0.0:
            regularizer = tf.keras.regularizers.l1(l1_regularization)
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1e-3, mode='fan_avg', seed=seed)
        bias_initializer = tf.keras.initializers.zeros()
        layer_opts = {
            'activation': activation,
            'kernel_regularizer': regularizer,
            'bias_regularizer': None,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'dtype': dtype,
        }
        batch_shape = (batch_size, ) + input_shape
        input = tf.keras.layers.Input(batch_shape=batch_shape)
        last = input
        for units in architecture:
            if dropout != 0.:
                last = tf.keras.layers.Dropout(dropout)(last)
            last = tf.keras.layers.Dense(units=units, **layer_opts)(last)
        if dropout != 0. and dropout_before_last:
            last = tf.keras.layers.Dropout(dropout)(last)
        layer_opts['activation'] = 'linear'
        outputs = {
            output: tf.keras.layers.Dense(
                units=1, name=output, **layer_opts
            )(last)
            for output in outputs
        }
        return tf.keras.Model(name=name, inputs=input, outputs=outputs)
