# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 18.11.2019.

from typing import Optional, Sequence

import tensorflow as tf

from models.dense_model import DenseModel


class LinearModel(DenseModel):

    def __new__(
            cls,
            name: str,
            input_shape: Sequence[Optional[int]],
            outputs: Sequence[str],
            batch_size: Optional[int] = None,
            l1_regularization: float = 0.,
            dropout: float = 0.,
            seed: int = 42,
            dtype: tf.keras.backend.dtype = tf.float32,
    ):
        return DenseModel(
            name,
            input_shape,
            outputs,
            batch_size=batch_size,
            activation='linear',
            architecture=tuple(),
            dropout=dropout,
            dropout_before_last=True,
            l1_regularization=l1_regularization,
            seed=seed,
            dtype=dtype,
        )
