# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
import logging
import sys
import os
import warnings
# Suppress Tensorflow's exceedingly verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import models
import parsing
import scripts.paths as paths
from training.evaluate import Evaluator
from training.optimizer import Optimizer
import training.training_utils as utils


def main():
    # Parse arguments
    parser = parsing.Parser('Evaluate models on a dataset')
    parsing.add_information(parser)
    parser.add_group('Dataset', parsing.SCRIPT_DATASET_ARGUMENTS)
    parser.add_group('Fold', parsing.FOLD_ARGUMENTS)
    parser.add_group('Model', parsing.BUILD_MODEL_ARGUMENTS)
    parser.add_group('Optimizer', parsing.OPTIMIZER_ARGUMENTS)
    parser.add_group('Pipeline', parsing.PIPELINE_SHAPE_ARGUMENTS)
    parser.add_group('Target', parsing.TARGET_ARGUMENTS)
    parser.add_group('Training', parsing.SCRIPT_TRAINING_ARGUMENTS)
    parser.parse()
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    # Retrieve arguments
    dataset_arguments = parser.get_group('Dataset')
    fold_arguments = parser.get_group('Fold')
    model_arguments = parser.get_group('Model')
    optimizer_arguments = parser.get_group('Optimizer')
    pipeline_arguments = parser.get_group('Pipeline')
    target_arguments = parser.get_group('Target')
    training_arguments = parser.get_group('Training')
    # Set up paths
    base_path = dataset_arguments['base_path']
    dataset = dataset_arguments['dataset']
    target_names = target_arguments['target_names']
    loss = training_arguments['loss']
    loss_weights = training_arguments['loss_weights']
    path_manager = paths.PathManager(base_path)
    nums_train_years = fold_arguments['num_train_years']
    num_valid_years = fold_arguments['num_valid_years']
    lags = fold_arguments['lags']
    optimizer = Optimizer.get_name(**optimizer_arguments)
    # For each requested fold
    results = list()
    for num_train_years in nums_train_years:
        logger.info(
            'Processing fold with {} training '.format(num_train_years) +
            'and {} validation years'.format(num_valid_years)
        )
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
        evaluator = Evaluator(
            logger=logger,
            column_information_arguments=column_information_arguments,
            pipeline_arguments=pipeline_arguments,
            target_arguments=target_arguments,
        )
        utils.update_model_arguments_from_pipeline(
            model_arguments, pipeline_arguments)
        if optimizer_arguments['optimizer'] == 'sklearn':
            del model_arguments['batch_size']
        # Evaluate the models
        model = models.get_name(**model_arguments)
        model_path_template = path_manager.checkpoint_path(
            dataset=dataset, model=model, target_names=target_names,
            loss=loss, loss_weights=loss_weights, optimizer=optimizer,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        epochs = training_arguments['epochs']
        model_path = paths.Path(str(model_path_template).format(epoch=epochs))
        result = evaluator.evaluate(model_path)
        results.append(result)
    # Compute statistics
    df = pd.DataFrame(data=results, index=nums_train_years).T
    if len(nums_train_years) > 1:
        means = df.mean(numeric_only=True, axis=1)
        stds = df.std(numeric_only=True, axis=1)
        df['mean'] = means
        df['std'] = stds
    pd.options.display.float_format = '{:.3f}'.format
    print(df)
    tex_path = model_path.parent / (model_path.stem + '.tex')
    with open(tex_path, 'w') as tex_file:
        tex_file.write(df.to_latex())


if __name__ == '__main__':
    main()
