# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 25.11.2019.
#
import cycler
import logging
import pathlib
import sys
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt

from datapreparation.datasets.column_information import ColumnInformation
import datapreparation.datasets.dataset_loaders as dataset_loaders
from datapreparation.datasets.helper.get_dates import get_dates
from inputpipelines.column_information_pipeline import ColumnInformationTrainValidPipeline
import parsing
from plotting.utils.plot_time_series import plot_time_series
import training.training_utils as utils
import training.step_operations as ops


class Predictor:

    def __init__(
            self,
            logger: logging.Logger,
            column_information_arguments: dict,
            pipeline_arguments: dict,
            target_arguments: dict,
    ):
        self.logger = logger
        # Read column information files
        self.logger.info('Reading the column information file')
        self.column_information = ColumnInformation(**column_information_arguments)
        self.target_arguments = target_arguments
        self.target_names = target_arguments.get('target_names', None)
        if not self.target_names:
            self.target_names = self.column_information.get_all_target_column_names()
        # Read the datasets
        self.logger.info('Reading the datasets')
        train_dataset = dataset_loaders.create_csv_dataset(
            dataset_path=pipeline_arguments['train_path'],
            dataset_delimiter=pipeline_arguments['train_delimiter'],
        )
        self.train_dataframe = train_dataset.load()
        valid_dataset = dataset_loaders.create_csv_dataset(
            dataset_path=pipeline_arguments['valid_path'],
            dataset_delimiter=pipeline_arguments['valid_delimiter'],
        )
        self.valid_dataframe = valid_dataset.load()
        # Convert dates
        self.logger.info('Converting dates')
        train_dates = get_dates(self.column_information, self.train_dataframe)
        valid_dates = get_dates(self.column_information, self.valid_dataframe)
        self.train_dates = train_dates.values
        self.valid_dates = valid_dates.values
        # Initialize plot line list
        self.plot_lines = {}

    def plot_data_lines(self):
        # Retrieve target data
        self.logger.info('Retrieving target data')
        for target_name in self.target_names:
            train_data = self.train_dataframe[target_name].values
            valid_data = self.valid_dataframe[target_name].values
            # Append to plot list
            self.plot_lines[target_name] = [
                {'x': self.train_dates, 'y': train_data, 'label': 'Train data'},
                {'x': self.valid_dates, 'y': valid_data, 'label': 'Valid data'},
            ]

    def plot_prediction_lines(
            self,
            pipeline_arguments: dict,
            model_path: Union[pathlib.Path, str],
    ):
        # Load the model
        self.logger.info('Loading the model from {}'.format(model_path))
        model = utils.load_model(model_path)
        # Setup the train and valid pipelines
        self.logger.info('Setting up the pipelines')
        new_pipeline_args = {k: v for k, v in pipeline_arguments.items()}
        utils.update_pipeline_arguments_from_model(
            new_pipeline_args, model)
        utils.update_pipeline_arguments_from_column_information_and_target(
            pipeline_arguments=new_pipeline_args,
            target_arguments=self.target_arguments,
            column_information=self.column_information,
        )
        pipeline = ColumnInformationTrainValidPipeline(**new_pipeline_args)
        # First predict the train data to update the state
        train_predictions = ops.predict(model, pipeline.train)
        valid_predictions = ops.predict(model, pipeline.valid)
        for target_name in self.target_names:
            try:
                train_prediction = train_predictions[target_name]
                valid_prediction = valid_predictions[target_name]
            except KeyError:
                self.logger.info(f'No predictions for {target_name}.')
                continue
            self.plot_lines[target_name] += [
                {
                    'x': self.train_dates, 'y': train_prediction,
                    'label': model.name + ' fit', 'linestyle': 'dotted',
                },
                {
                    'x': self.valid_dates, 'y': valid_prediction,
                    'label': model.name + ' inference', 'linestyle': 'dotted',
                },
            ]

    def make_plot(
            self,
            plotting_arguments: dict,
    ):
        # Produce the plot
        self.logger.info('Producing the plot')
        cm = plt.cm.get_cmap('tab20')
        color_cycler = cycler.cycler('color', [cm.colors[i] for i in range(20)])
        plt.rc('axes', prop_cycle=color_cycler)
        nrows = len(self.plot_lines)
        figure, axes = plt.subplots(nrows=nrows, sharex='row')
        if nrows == 1:
            axes = (axes, )
        for i, target in enumerate(self.plot_lines.keys()):
            axes[i].set_title(target)
            axes[i].label_outer()
            plot_time_series(
                self.plot_lines[target],
                axes[i],
                start_date=plotting_arguments['start_date'],
                end_date=plotting_arguments['end_date'],
            )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels)
        if plotting_arguments['show']:
            self.logger.info('Displaying the plot')
            plt.show()
        if plotting_arguments['save']:
            path = plotting_arguments['save']
            self.logger.info('Saving the plot to {}'.format(path))
            figure.savefig(path)


def main():
    # Parse arguments
    parser = parsing.Parser('Plot predictions from models on a dataset')
    parsing.add_information(parser)
    parser.add_group('Target', parsing.TARGET_ARGUMENTS)
    parser.add_group('Column information', parsing.COLUMN_INFORMATION_ARGUMENTS)
    parser.add_group('Pipeline', parsing.PIPELINE_ARGUMENTS)
    parser.add_group('Model', parsing.LOAD_MODEL_ARGUMENTS)
    parser.add_group('Plotting', parsing.PLOTTING_ARGUMENTS)
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
    plotting_arguments = parser.get_group('Plotting')
    # Predict
    predictor = Predictor(
        logger=logger,
        column_information_arguments=column_information_arguments,
        pipeline_arguments=pipeline_arguments,
        target_arguments=target_arguments,
    )
    predictor.plot_data_lines()
    for model_path in load_model_arguments['model_paths']:
        predictor.plot_prediction_lines(pipeline_arguments, model_path)
    predictor.make_plot(plotting_arguments)


if __name__ == '__main__':
    main()
