# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 21.11.2019.

import tensorflow as tf

K = tf.keras.backend


class RegularizedMAPE(tf.keras.metrics.Metric):

    def __init__(self, name='r2', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight("sum", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        PEs = (y_true - y_pred) / (y_true + 1)
        APEs = K.abs(PEs)
        self.sum.assign_add(K.sum(APEs))
        self.count.assign_add(K.cast(K.shape(y_true)[0], self.dtype))

    def result(self):
        return self.sum / self.count

    def reset_states(self):
        self.sum.assign(0.0)
        self.count.assign(0.0)


class PatchedMAPE(tf.keras.metrics.Metric):

    def __init__(self, name='r2', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight("sum", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        denominators = tf.math.maximum(y_true, 1)
        PEs = (y_true - y_pred) / denominators
        APEs = K.abs(PEs)
        self.sum.assign_add(K.sum(APEs))
        self.count.assign_add(K.cast(K.shape(y_true)[0], self.dtype))

    def result(self):
        return self.sum / self.count

    def reset_states(self):
        self.sum.assign(0.0)
        self.count.assign(0.0)
