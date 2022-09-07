# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 21.11.2019.
import tensorflow as tf


K = tf.keras.backend


class PoissonRootMeanSquare(tf.keras.metrics.Metric):

    def __init__(self, name: str = 'PRMS', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sigma2_sum = self.add_weight("sigma2_sum", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_t = K.cast(y_true, self.dtype)
        y_p = K.cast(y_pred, self.dtype)
        residual_squares = (y_t - y_p) ** 2
        variance_estimates = y_t + 1
        normalized_squared_residuals = residual_squares / variance_estimates
        self.sigma2_sum.assign_add(K.sum(normalized_squared_residuals))
        self.count.assign_add(K.cast(K.shape(y_t)[0], self.dtype))

    def result(self):
        return (self.sigma2_sum  / self.count) ** 0.5

    def reset_states(self):
        self.sigma2_sum.assign(0.0)
        self.count.assign(0.0)


class PoissonLogLikelihood(tf.keras.metrics.Metric):

    def __init__(self, name: str = 'PLL', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight("sum", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_t = K.cast(y_true, self.dtype)
        y_p = K.cast(y_pred, self.dtype)
        log_likelihood = y_t * K.log(y_p) - y_p - tf.math.lgamma(y_t + 1)
        self.sum.assign_add(K.sum(log_likelihood))
        self.count.assign_add(K.cast(K.shape(y_t)[0], self.dtype))

    def result(self):
        return self.sum / self.count

    def reset_states(self):
        self.sum.assign(0.0)
        self.count.assign(0.0)
