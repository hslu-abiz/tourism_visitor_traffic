# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 18.11.2019.

from typing import Iterable, Optional, Sequence

import tensorflow as tf


class RNNModel(tf.keras.Model):

    unit_types = {
        'SimpleRNN': tf.keras.layers.SimpleRNN,
        'GRU': tf.keras.layers.GRU,
        'LSTM': tf.keras.layers.LSTM,
    }

    def __new__(
            cls,
            name: str,
            input_shape: Sequence[Optional[int]],
            outputs: Iterable[str],
            batch_size: Optional[int] = None,
            unit_type: str = 'SimpleRNN',
            architecture: Sequence[int] = tuple(),
            kernel_dropout: float = 0.,
            recurrent_dropout: float = 0.,
            kernel_regularization: float = 0.,
            recurrent_regularization: float = 0.,
            layer_normalization: bool = False,
            activation: str = 'relu',
            seed: int = 42,
    ):
        if unit_type not in cls.unit_types.keys():
            raise ValueError(
                'unit_type must be one of {}'.format(cls.unit_types.keys()))
        kernel_regularizer = None
        if kernel_regularization != 0.:
            kernel_regularizer = tf.keras.regularizers.l1(kernel_regularization)
        recurrent_regularizer = None
        if recurrent_regularization != 0.:
            recurrent_regularizer = tf.keras.regularizers.l1(
                recurrent_regularization)
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1e-3, mode='fan_avg', seed=seed)
        recurrent_initializer = tf.keras.initializers.orthogonal(seed=seed)
        bias_initializer = tf.keras.initializers.zeros()
        layer_opts = {
            'use_bias': True,
            'return_sequences': True,
            'stateful': True,
            'dropout': kernel_dropout,
            'recurrent_dropout': recurrent_dropout,
            'kernel_regularizer': kernel_regularizer,
            'recurrent_regularizer': recurrent_regularizer,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_initializer': kernel_initializer,
            'recurrent_initializer': recurrent_initializer,
            'bias_initializer': bias_initializer,
        }
        layer_class = cls.unit_types[unit_type]
        batch_shape = (batch_size, ) + input_shape
        input = tf.keras.layers.Input(batch_shape=batch_shape)
        last = input
        for units in architecture:
            last = layer_class(
                units=units,
                **layer_opts,
            )(last)
            if layer_normalization:
                last = tf.keras.layers.LayerNormalization(
                    axis=-1,
                    epsilon=tf.keras.backend.epsilon(),
                    center=False,
                    scale=False,
                )(last)
        outputs = {
            output: tf.keras.layers.Dense(
                units=1,
                name=output,
                activation='linear',
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(last)
            for output in outputs
        }
        return tf.keras.Model(name=name, inputs=input, outputs=outputs)
