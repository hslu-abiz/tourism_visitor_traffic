# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 22.11.2019.
#
import logging
import sys

import matplotlib.pyplot as plt

from datapreparation.datasets.column_information import ColumnInformation
import datapreparation.datasets.dataset_loaders as dataset_loaders
from datapreparation.datasets.helper.get_dates import get_date_column_name, get_dates
from plotting.utils.plot_time_series import plot_time_series
import parsing


def plot_data(
        logger: logging.Logger,
        column_information_arguments: dict,
        dataset_arguments: dict,
        plotting_arguments: dict,
        target_arguments: dict,
):
    # Check that the plot is either shown or saved
    if not plotting_arguments['show'] and not plotting_arguments['save']:
        logger.error('The plot will neither be shown nor saved.')
    # Read column information files
    logger.info('Reading the column information file')
    column_information = ColumnInformation(**column_information_arguments)
    # Read the dataset
    logger.info('Reading the dataset')
    dataset = dataset_loaders.create_csv_dataset(**dataset_arguments)
    dataframe = dataset.load()
    dataframe = dataframe.sort_values(get_date_column_name(column_information))
    # Convert dates
    logger.info('Converting dates')
    dates = get_dates(column_information, dataframe).values
    # Select the targets to plot
    logger.info('Selecting the columns to plot')
    columns = target_arguments['target_names']
    if not columns:
        columns = column_information.get_all_target_column_names()
    all_time_series = {
        column: [{'x': dates, 'y': dataframe[column].values}, ]
        for column in columns
    }
    # Produce the plot
    logger.info('Producing the plot')
    nrows = len(all_time_series)
    figure, axes = plt.subplots(nrows=nrows, sharex='row', figsize=(12, 8))
    if nrows == 1:
        axes = (axes,)
    for i, column in enumerate(columns):
        axes[i].set_title(column)
        axes[i].label_outer()
        plot_time_series(
            all_time_series[column],
            axes[i],
            start_date=plotting_arguments['start_date'],
            end_date=plotting_arguments['end_date'],
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels)
    # Show and save the plot if requested
    if plotting_arguments['show']:
        logger.info('Displaying the plot')
        plt.show()
    if plotting_arguments['save']:
        logger.info('Saving the plot to {}'.format(plotting_arguments['save']))
        figure.savefig(plotting_arguments['save'])


def main():
    # Parse arguments
    parser = parsing.Parser('Plot time series data for a dataset')
    parsing.add_information(parser)
    parser.add_group('Column information', parsing.DATASET_ARGUMENTS)
    parser.add_group('Dataset', parsing.DATASET_ARGUMENTS)
    parser.add_group('Plotting', parsing.PLOTTING_ARGUMENTS)
    parser.add_group('Target', parsing.TARGET_ARGUMENTS)
    parser.parse()
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    plot_data(
        logger=logger,
        column_information_arguments=parser.get_group('Column information'),
        dataset_arguments=parser.get_group('Dataset'),
        plotting_arguments=parser.get_group('Plotting'),
        target_arguments=parser.get_group('Target'),
    )


if __name__ == '__main__':
    main()
