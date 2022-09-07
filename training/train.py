# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
import logging
import math
import pathlib
import sys
import time
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf

from datapreparation.datasets.column_information import ColumnInformation
from inputpipelines.column_information_pipeline import ColumnInformationTrainValidPipeline
import models
import parsing
from training.optimizer import Optimizer
import training.step_operations as ops
from training.training_configuration import FLOAT_TYPE, LOSS_DICT, METRICS
import training.training_utils as utils

# Set tensorflow configs
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu in physical_devices:
  try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
  except Exception as ex:  
    # Invalid device or cannot modify virtual devices once initialized. 
    warnings.warn('Set memory growth on gpu did not work.')
    warnings.warn(ex)
    pass 

class Trainer:

    def __init__(
            self,
            model: tf.keras.models.Model,
            optimizer: tf.keras.optimizers.Optimizer,
            pipeline: ColumnInformationTrainValidPipeline,
            metrics: Dict[str, tf.keras.metrics.Metric] = METRICS,
    ):
        self.model = model
        self.optimizer = optimizer
        self.pipeline = pipeline
        # Setup loss
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=FLOAT_TYPE)
        self.train_regularization_loss = tf.keras.metrics.Mean('train_regularization_loss', dtype=FLOAT_TYPE)
        self.valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=FLOAT_TYPE)
        # Setup metrics
        self.targets = pipeline.target_names
        self.train_metrics = {
            target: [
                metric_class(target + '_train_' + metric_name)
                for metric_name, metric_class in metrics.items()
            ]
            for target in self.targets
        }
        self.valid_metrics = {
            target: [
                metric_class(target + '_valid_' + metric_name)
                for metric_name, metric_class in metrics.items()
            ]
            for target in self.targets
        }

    def _check_loss_weights(self, loss_weights: Dict[str, float]):
        if loss_weights is None:
            return {name: 1.0 for name in self.pipeline.target_names}
        exists_in_loss_weights = [name in loss_weights for name in self.pipeline.target_names]
        exists_in_targets = [name in self.pipeline.target_names for name in loss_weights]
        if not np.all(exists_in_targets):
            extra_names = np.array(list(loss_weights.keys()))[exists_in_targets]
            warnings.warn('Weighted loss is used but keys {} were not found in targets. '
                          'Valid are {}'.format(
                ', '.join(extra_names), ', '.join(self.pipeline.target_names)))
        if not np.all(exists_in_loss_weights):
            missing_names = np.array(self.pipeline.target_names)[np.array(exists_in_loss_weights)==False]
            warnings.warn('Weighted loss is used but no weight is defined for {}. '
                          'They are set to 1.0.'.format(', '.join(missing_names)))
            defaults = {name: 1.0 for name in missing_names}
            loss_weights.update(defaults)
        return loss_weights

    def initialize_biases(
            self,
            column_information_arguments: Dict[str, Any],
    ):
        column_information = ColumnInformation(**column_information_arguments)
        df = column_information.df
        try:
            for target in self.targets:
                row = df.loc[df[column_information.column_name] == target]
                mean = row[column_information.column_mean].values
                layer = self.model.get_layer(target)
                weights = layer.get_weights()
                layer.set_weights([weights[0], np.log(mean)])
        except ValueError:
            warnings.warn('Could not initialize model biases with means')

    def train(
            self,
            epochs: int = 0,
            loss: str = 'MSE',
            loss_weights: Optional[Dict[str, float]] = None,
            tensorboard_path: Union[str, pathlib.Path] = '',
            checkpoint_path: Union[str, pathlib.Path] = '',
            checkpoint_frequency: Optional[int] = None,
            max_to_keep: Optional[int] = 2,
            resume_if_possible: bool = True,
    ):
        # Sanity check
        loss_weights_dict = self._check_loss_weights(loss_weights)

        loss_function = LOSS_DICT[loss]
        log_path = str(tensorboard_path)
        summary_writer = tf.summary.create_file_writer(log_path)
        ckpt = tf.train.Checkpoint(model=self.model,
                                   optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path.parent, max_to_keep=max_to_keep)
        start_epoch = 1
        if resume_if_possible and ckpt_manager.latest_checkpoint:
            print("Restoring model from latest checkpoint...")
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) + 1
            ckpt.restore(ckpt_manager.latest_checkpoint)
        for epoch in range(start_epoch, epochs+1):
            print('Epoch: {}'.format(epoch))
            start_time = time.time()
            # Perform training and compute training metrics
            ops.train_epoch(
                model=self.model,
                optimizer=self.optimizer,
                loss=loss_function,
                loss_weights=loss_weights_dict,
                loss_metric=self.train_loss,
                regularization_loss_metric=self.train_regularization_loss,
                metrics_dict=self.train_metrics,
                dataset=self.pipeline.train,
            )
            # Save model and checkpoint
            if ((checkpoint_frequency is not None
                and epoch % checkpoint_frequency == 0)
                or epoch == epochs):
                self.model.save(str(checkpoint_path).format(epoch=epoch))
            ckpt_manager.save()
            # Log training metrics to screen and tensorboard
            with summary_writer.as_default():
                train_dict = ops.get_loss_and_metrics(
                    [self.train_loss, self.train_regularization_loss],
                    self.train_metrics,
                    epoch
                )
            print(train_dict)
            # Reset training metrics
            ops.reset_loss_and_metrics(self.train_loss, self.train_metrics)
            # Compute validation metrics
            ops.evaluate(
                model=self.model,
                loss=loss_function,
                loss_metric=self.valid_loss,
                metrics_dict=self.valid_metrics,
                dataset=self.pipeline.valid,
                loss_weights=loss_weights_dict
            )
            # Log validation metrics to screen and tensorboard
            with summary_writer.as_default():
                valid_dict = ops.get_loss_and_metrics(
                   [self.valid_loss], self.valid_metrics, epoch)
            print(valid_dict)
            # Reset validation metrics
            ops.reset_loss_and_metrics(self.valid_loss, self.valid_metrics)
            # Reset internal model states
            self.model.reset_states()
            # Print some timing information
            end_time = time.time()
            epoch_time = end_time - start_time
            print('Epoch {} took {} s.'.format(epoch, epoch_time))
            # Abort if infs or nans are produced in the training loss
            if not (math.isfinite(train_dict['train_loss'])):
                break


def main():
    # Parse arguments
    parser = parsing.Parser('Train a baseline for a single dataset')
    parsing.add_information(parser)
    parser.add_group('Target', parsing.TARGET_ARGUMENTS)
    parser.add_group('Column information', parsing.COLUMN_INFORMATION_ARGUMENTS)
    parser.add_group('Pipeline', parsing.PIPELINE_ARGUMENTS)
    parser.add_group('Model', parsing.BUILD_MODEL_ARGUMENTS)
    parser.add_group('Optimizer', parsing.OPTIMIZER_ARGUMENTS)
    parser.add_group('Training', parsing.TRAINING_ARGUMENTS)
    parser.parse()
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    # Train
    target_arguments = parser.get_group('Target')
    column_information_arguments = parser.get_group('Column information')
    pipeline_arguments = parser.get_group('Pipeline')
    model_arguments = parser.get_group('Model')
    optimizer_arguments = parser.get_group('Optimizer')
    training_arguments = parser.get_group('Training')
    # Setup the train and valid pipelines
    logger.info('Setting up the pipelines')
    utils.update_pipeline_arguments_from_column_information_and_target(
        pipeline_arguments, target_arguments, column_information_arguments)
    pipeline = ColumnInformationTrainValidPipeline(**pipeline_arguments)
    # Setup the model
    logger.info('Setting up the model')
    utils.update_model_arguments_from_pipeline(
        model_arguments, pipeline_arguments)
    model = models.make_model(**model_arguments)
    # Setup the optimizer
    logger.info('Setting up the optimizer')
    optimizer = Optimizer(**optimizer_arguments)
    # Run the training
    logger.info('Setting up the training')
    training = Trainer(model, optimizer, pipeline)
    logger.info('Fitting the model')
    training.train(**training_arguments)


if __name__ == '__main__':
    main()
