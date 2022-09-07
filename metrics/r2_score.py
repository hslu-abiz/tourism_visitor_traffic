# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 21.11.2019.

import tensorflow as tf


K = tf.keras.backend


class R2(tf.keras.metrics.Metric):

    def __init__(self, name='r2', **kwargs):
        super().__init__(name=name, **kwargs)
        self.squared_sum = self.add_weight("squared_sum", initializer="zeros")
        self.sum = self.add_weight("sum", initializer="zeros")
        self.res = self.add_weight("residual", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_t = K.cast(y_true, self.dtype)
        y_p = K.cast(y_pred, self.dtype)
        self.squared_sum.assign_add(K.sum(y_t ** 2))
        self.sum.assign_add(K.sum(y_t))
        self.res.assign_add(K.sum(K.square(y_t - y_p)))
        self.count.assign_add(K.cast(K.shape(y_t)[0], self.dtype))

    def result(self):
        mean = self.sum / self.count
        total = self.squared_sum - self.sum * mean
        total += K.epsilon()
        return 1 - (self.res / total)

    def reset_states(self):
        self.squared_sum.assign(0.0)
        self.sum.assign(0.0)
        self.res.assign(0.0)
        self.count.assign(0.0)
