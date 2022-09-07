# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
import logging
import sys

from inputpipelines.column_information_pipeline import ColumnInformationTrainValidPipeline
import models
import parsing
import scripts.paths as paths
from training.optimizer import Optimizer
from training.train import Trainer
import training.training_utils as utils
import tensorflow as tf

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

def main():
    # Parse arguments
    parser = parsing.Parser('Train a baseline for a single dataset')
    parsing.add_information(parser)
    parser.add_group('Dataset', parsing.SCRIPT_DATASET_ARGUMENTS)
    parser.add_group('Target', parsing.TARGET_ARGUMENTS)
    parser.add_group('Fold', parsing.FOLD_ARGUMENTS)
    parser.add_group('Pipeline', parsing.PIPELINE_SHAPE_ARGUMENTS)
    parser.add_group('Model', parsing.BUILD_MODEL_ARGUMENTS)
    parser.add_group('Optimizer', parsing.OPTIMIZER_ARGUMENTS)
    parser.add_group('Training', parsing.SCRIPT_TRAINING_ARGUMENTS)
    parser.parse()
    # Set debug messages for tensorflow
    #tf.debugging.set_log_device_placement(True)
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    # Train
    dataset_arguments = parser.get_group('Dataset')
    target_arguments = parser.get_group('Target')
    fold_arguments = parser.get_group('Fold')
    pipeline_arguments = parser.get_group('Pipeline')
    model_arguments = parser.get_group('Model')
    optimizer_arguments = parser.get_group('Optimizer')
    training_arguments = parser.get_group('Training')
    base_path = dataset_arguments['base_path']
    dataset = dataset_arguments['dataset']
    target_names = target_arguments['target_names']
    loss = training_arguments['loss']
    loss_weights = training_arguments['loss_weights']
    path_manager = paths.PathManager(base_path)
    nums_train_years = fold_arguments['num_train_years']
    num_valid_years = fold_arguments['num_valid_years']
    lags = fold_arguments['lags']
    for num_train_years in nums_train_years:
        # Setup the train and valid pipelines
        logger.info('Setting up the pipelines')
        column_information_arguments = path_manager.get_processed_column_information_arguments(
            dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        path_manager.update_pipeline_arguments(
            arguments=pipeline_arguments, dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
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
        optimizer_name = Optimizer.get_name(**optimizer_arguments)
        # Setup the training
        logger.info('Setting up the training')
        training = Trainer(model, optimizer, pipeline)
        # Run the training
        logger.info('Running the training')
        training_arguments['tensorboard_path'] = path_manager.tensorboard_path(
            dataset=dataset, model=model.name, target_names=target_names,
            loss=loss, loss_weights=loss_weights, optimizer=optimizer_name,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        training_arguments['checkpoint_path'] = path_manager.checkpoint_path(
            dataset=dataset, model=model.name, target_names=target_names,
            loss=loss, loss_weights=loss_weights, optimizer=optimizer_name,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        training.train(**training_arguments)


if __name__ == '__main__':
    main()
