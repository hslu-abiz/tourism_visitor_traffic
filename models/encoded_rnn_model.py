# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 18.11.2019.

from typing import Iterable, Optional, Sequence

import tensorflow as tf


class EncodedRNNModel(tf.keras.Model):

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
            hidden_size: int = 128,
            dropout: float = 0.,
            kernel_regularization: float = 0.,
            recurrent_regularization: float = 0.,
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
            recurrent_regularizer = tf.keras.regularizers.l1(recurrent_regularization)
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1e-3, mode='fan_avg', seed=seed)
        recurrent_initializer = tf.keras.initializers.orthogonal(seed=seed)
        bias_initializer = tf.keras.initializers.zeros()
        dense_layer_opts = {
            'units': hidden_size,
            'activation': activation,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': None,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
        }
        rnn_layer_opts = {
            'units': hidden_size,
            'use_bias': True,
            'return_sequences': True,
            'stateful': True,
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
        inputs = tf.keras.layers.Input(batch_shape=batch_shape)
        inputs_dropout = tf.keras.layers.Dropout(dropout)(inputs)
        encoded = tf.keras.layers.Dense(**dense_layer_opts)(inputs_dropout)
        encoded_dropout = tf.keras.layers.Dropout(dropout)(encoded)
        rnn = layer_class(**rnn_layer_opts)(encoded_dropout)
        rnn_dropout = tf.keras.layers.Dropout(dropout)(rnn)
        decoded = tf.keras.layers.Dense(**dense_layer_opts)(rnn_dropout)
        decoded_dropout = tf.keras.layers.Dropout(dropout)(decoded)
        outputs = {
            output: tf.keras.layers.Dense(
                units=1,
                name=output,
                activation='linear',
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(decoded_dropout)
            for output in outputs
        }
        return tf.keras.Model(name=name, inputs=inputs, outputs=outputs)
