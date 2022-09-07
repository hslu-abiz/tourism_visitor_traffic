# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import tensorflow as tf

from training.training_configuration import FLOAT_TYPE


@tf.function
def apply_masks(tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    not_mask = tf.equal(mask, 0.)
    return tf.boolean_mask(tensor, not_mask)


@tf.function
def train_step(
        model: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
        loss_metric: tf.keras.metrics.Metric,
        regularization_loss_metric: tf.keras.metrics.Metric,
        metrics_dict: Dict[str, Iterable[tf.keras.metrics.Metric]],
        features: tf.Tensor,
        targets: Dict[str, tf.Tensor],
        loss_weights: Dict[str, float],
):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        tape.watch(model.trainable_variables)
        model_loss_values = model.losses
        regularization_loss = tf.reduce_sum(model_loss_values)
        total_loss = tf.reduce_sum(model_loss_values)
        for k in targets.keys():
            prediction_values = predictions[k]
            prediction_values = tf.keras.backend.reshape(
                prediction_values, prediction_values.shape[:-1])
            target_values = targets[k][:, :, 0]
            masks = targets[k][:, :, 1]
            masked_prediction = apply_masks(prediction_values, masks)
            masked_target = apply_masks(target_values, masks)
            loss_weight = loss_weights[k]
            total_loss += loss_weight * loss(masked_target, masked_prediction)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    regularization_loss_metric(regularization_loss)
    loss_metric(total_loss)
    for k in targets.keys():
        prediction_values = predictions[k]
        prediction_values = tf.keras.backend.reshape(
            prediction_values, prediction_values.shape[:-1])
        target_values = targets[k][:, :, 0]
        masks = targets[k][:, :, 1]
        masked_prediction = tf.math.exp(apply_masks(prediction_values, masks))
        masked_target = apply_masks(target_values, masks)
        for metric in metrics_dict[k]:
            metric(masked_target, masked_prediction)


@tf.function
def evaluate_step(
        model: tf.keras.models.Model,
        metrics_dict: Dict[str, Iterable[tf.keras.metrics.Metric]],
        features: tf.Tensor,
        targets: Dict[str, tf.Tensor],
        loss: Optional[tf.keras.losses.Loss] = None,
        loss_metric: Optional[tf.keras.metrics.Metric] = None,
        loss_weights: Dict[str, float] = None,
):
    predictions = model(features, training=False)
    total_loss = None
    if loss is not None:
        total_loss = tf.reduce_sum(model.losses)
    for k in targets.keys():
        # Casting is necessary for different precision required in linear_fit
        prediction_values = tf.keras.backend.cast(predictions[k], FLOAT_TYPE)
        prediction_values = tf.keras.backend.reshape(
            prediction_values, prediction_values.shape[:-1])
        target_values = targets[k][:, :, 0]
        masks = targets[k][:, :, 1]
        masked_prediction = apply_masks(prediction_values, masks)
        masked_target = apply_masks(target_values, masks)
        if loss is not None:
            total_loss += loss(masked_target, masked_prediction) if loss_weights is None \
                else loss_weights[k] * loss(masked_target, masked_prediction)
        masked_prediction = tf.math.exp(masked_prediction)
        for metric in metrics_dict[k]:
            metric(masked_target, masked_prediction)
    if loss_metric is not None:
        loss_metric(total_loss)


def train_epoch(
        model: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
        loss_metric: tf.keras.metrics.Metric,
        regularization_loss_metric: tf.keras.metrics.Metric,
        metrics_dict: Dict[str, Iterable[tf.keras.metrics.Metric]],
        dataset: tf.data.Dataset,
        loss_weights: Dict[str, float],
):
    for data_dict in dataset:
        features = data_dict['features']
        targets = {k: v for k, v in data_dict.items() if k != 'features'}
        train_step(
            model, optimizer, loss, loss_metric, regularization_loss_metric,
            metrics_dict, features, targets, loss_weights,
        )


def evaluate(
        model: tf.keras.models.Model,
        metrics_dict: Dict[str, Iterable[tf.keras.metrics.Metric]],
        dataset: tf.data.Dataset,
        loss: Optional[tf.keras.losses.Loss] = None,
        loss_metric: Optional[tf.keras.metrics.Metric] = None,
        loss_weights: Dict[str, float] = None,
):
    for data_dict in dataset:
        features = data_dict['features']
        targets = {k: v for k, v in data_dict.items() if k != 'features'}
        evaluate_step(model, metrics_dict, features, targets, loss, loss_metric,
                      loss_weights)


def predict(
        model: tf.keras.models.Model,
        dataset: tf.data.Dataset,
):
    predictions = {}
    for batch in dataset:
        result = model(batch['features'], training=False)
        for k, v in result.items():
            if k in predictions:
                predictions[k] = np.concatenate(
                    [predictions[k], v.numpy().flatten()]
                )
            else:
                predictions[k] = v.numpy().flatten()
            predictions[k] = np.exp(predictions[k])
    return predictions


def get_loss_and_metrics(
        loss_metrics: Sequence[tf.keras.metrics.Metric] = tuple(),
        metrics_dict: Optional[Dict[str, Iterable[tf.keras.metrics.Metric]]] = None,
        epoch: Optional[int] = None,
) -> Dict[str, float]:
    if epoch is not None:
        for loss_metric in loss_metrics:
            tf.summary.scalar(loss_metric.name, loss_metric.result(), step=epoch)
        for metrics in metrics_dict.values():
            for metric in metrics:
                tf.summary.scalar(metric.name, metric.result(), step=epoch)
    result = {loss_metric.name: loss_metric.result().numpy()
              for loss_metric in loss_metrics}
    if metrics_dict is not None:
        for metrics in metrics_dict.values():
            for metric in metrics:
                result[metric.name] = metric.result().numpy()
    return result


def reset_loss_and_metrics(
        loss_metric: Optional[tf.keras.metrics.Metric] = None,
        metrics_dict: Optional[Dict[str, Iterable[tf.keras.metrics.Metric]]] = None,
) -> None:
    if loss_metric is not None:
        loss_metric.reset_states()
    if metrics_dict is not None:
        for metrics in metrics_dict.values():
            for metric in metrics:
                metric.reset_states()
