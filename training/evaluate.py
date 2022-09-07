# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 26.11.2019.
#
import logging
import pathlib
import sys
from typing import Dict, Union

import tensorflow as tf

from datapreparation.datasets.column_information import ColumnInformation
from inputpipelines.column_information_pipeline import ColumnInformationTrainValidPipeline
import parsing
from training.training_configuration import METRICS
import training.training_utils as utils
import training.step_operations as ops


class Evaluator:

    def __init__(
            self,
            logger: logging.Logger,
            column_information_arguments: dict,
            pipeline_arguments: dict,
            target_arguments: dict,
            metrics: Dict[str, tf.keras.metrics.Metric] = METRICS,
    ):
        # Setup logger
        self.logger = logger
        # Read column information files
        self.logger.info('Reading the column information file')
        self.column_information = ColumnInformation(**column_information_arguments)
        self.pipeline_arguments = pipeline_arguments
        self.target_arguments = target_arguments
        # Setup metrics
        self.targets = target_arguments['target_names']
        if not self.targets:
            self.targets = self.column_information.get_all_target_column_names()
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

    def evaluate(
            self,
            model_path: Union[pathlib.Path, str],
            model_arg_file: Union[pathlib.Path, str] = None
    ):
        # Load the model
        self.logger.info('Loading model {}'.format(model_path))
        model_path = pathlib.Path(model_path)

        if model_path.suffix == ".hdf5":
            model = utils.load_model(model_path)
            # Pipeline arguments
            new_pipeline_args = {k: v for k, v in self.pipeline_arguments.items()}
            utils.update_pipeline_arguments_from_model(
                new_pipeline_args, model)
            utils.update_pipeline_arguments_from_column_information_and_target(
                pipeline_arguments=new_pipeline_args,
                target_arguments=self.target_arguments,
                column_information=self.column_information,
            )
        else:
            new_pipeline_args = self.pipeline_arguments.copy()
            utils.update_pipeline_arguments_from_column_information_and_target(
                new_pipeline_args, self.target_arguments, column_information = self.column_information)
            model = utils.load_model_from_checkpoint(model_path, model_arg_file, new_pipeline_args)
            if not 'batch_size' in new_pipeline_args:
                new_pipeline_args.setdefault('batch_size', 1)
            if not 'time_steps' in new_pipeline_args:
                new_pipeline_args.setdefault('time_steps', None)

        # Setup the train and valid pipelines
        self.logger.info('Setting up the pipelines')
        pipeline = ColumnInformationTrainValidPipeline(**new_pipeline_args)

        # Compute training metrics
        ops.evaluate(model, self.train_metrics, pipeline.train)
        result = ops.get_loss_and_metrics(metrics_dict=self.train_metrics)
        ops.reset_loss_and_metrics(metrics_dict=self.train_metrics)
        # Compute validation metrics
        ops.evaluate(model, self.valid_metrics, pipeline.valid)
        result.update(ops.get_loss_and_metrics(metrics_dict=self.valid_metrics))
        ops.reset_loss_and_metrics(metrics_dict=self.valid_metrics)
        return result


def main():
    # Parse arguments
    parser = parsing.Parser('Evaluate models on a dataset')
    parsing.add_information(parser)
    parser.add_group('Target', parsing.TARGET_ARGUMENTS)
    parser.add_group('Column information', parsing.COLUMN_INFORMATION_ARGUMENTS)
    parser.add_group('Pipeline', parsing.PIPELINE_ARGUMENTS)
    parser.add_group('Model', parsing.LOAD_MODEL_ARGUMENTS)
    parser.parse()
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    # Retrieve arguments
    target_arguments = parser.get_group('Target')
    column_information_arguments = parser.get_group('Column information')
    pipeline_arguments = parser.get_group('Pipeline')
    load_model_arguments = parser.get_group('Model')
    # Compute results per model
    evaluator = Evaluator(
        logger=logger,
        column_information_arguments=column_information_arguments,
        pipeline_arguments=pipeline_arguments,
        target_arguments=target_arguments,
    )
    results = dict()
    model_path_args = load_model_arguments['model_paths']
    model_arg_file_args = load_model_arguments['model_arg_file']
    if model_arg_file_args is None:
        model_arg_file_args = [None]*len(model_arg_file_args)
    else:
        assert len(model_arg_file_args)==len(model_path_args)

    for model_path, model_arg_file in zip(model_path_args, model_arg_file_args):
        evaluations = evaluator.evaluate(model_path, model_arg_file)
        results[model_path] = evaluations
    # Print results
    for k, v in results.items():
        print('{}:\t{}'.format(k, v))


if __name__ == '__main__':
    main()
