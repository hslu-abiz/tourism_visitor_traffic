# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
import tensorflow as tf

import metrics


def poisson_loss_from_logits(y_true, y_pred):
    return tf.reduce_mean(
        tf.exp(y_pred) - y_pred * y_true + tf.math.lgamma(y_true + 1)
    )


def regularized_poisson_loss_from_logits(y_true, y_pred):
    r_true = tf.random.poisson(tuple(), y_true)
    return tf.reduce_mean(
        tf.exp(y_pred) - y_pred * r_true + tf.math.lgamma(r_true + 1)
    )


def MSE_loss_from_logits(y_true, y_pred):
    return tf.reduce_mean((tf.exp(y_pred) - y_true) ** 2)


def regularized_MSE_loss_from_logits(y_true, y_pred):
    actual_pred = tf.exp(y_pred)
    regularized_true = tf.random.poisson(tuple(), y_true)
    return tf.reduce_mean((actual_pred - regularized_true) ** 2)


FLOAT_TYPE = tf.float32

CUSTOM_OBJECTS = {
    'R2': metrics.R2,
    'RdMAPE': metrics.RegularizedMAPE,
    'PdMAPE': metrics.PatchedMAPE,
    'PRMS': metrics.PoissonRootMeanSquare,
    'PLL': metrics.PoissonLogLikelihood,
}

LOSS_DICT = {
    'MSE': MSE_loss_from_logits,
    'Poisson': poisson_loss_from_logits,
    'RdMSE': regularized_MSE_loss_from_logits,
    'RdPoisson': regularized_poisson_loss_from_logits,
}

METRICS = {
    'RMSE': tf.keras.metrics.RootMeanSquaredError,
    'PRMS': metrics.PoissonRootMeanSquare,
    'RdMAPE': metrics.RegularizedMAPE,
    'PdMAPE': metrics.PatchedMAPE,
    'R2': metrics.R2,
    'PLL': metrics.PoissonLogLikelihood,
}
