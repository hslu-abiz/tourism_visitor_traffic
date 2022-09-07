# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 19.11.2019.
#
import logging
import sys

import parsing
import scripts.paths as paths
from plotting.plot_data import plot_data


def main():
    # Parse arguments
    parser = parsing.Parser('Plot time series data for a dataset')
    parsing.add_information(parser)
    parser.add_group('Dataset', parsing.SCRIPT_DATASET_ARGUMENTS)
    parser.add_group('Plotting', parsing.PLOTTING_ARGUMENTS)
    parser.add_group('Target', parsing.TARGET_ARGUMENTS)
    parser.parse()
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    # Determine file names
    script_arguments = parser.get_group('Dataset')
    base_path = script_arguments['base_path']
    dataset = script_arguments['dataset']
    path_manager = paths.PathManager(base_path)
    # Plot data
    logger.info('Processing dataset {}'.format(dataset))
    plot_data(
        logger=logger,
        column_information_arguments=path_manager.get_raw_column_information_arguments(dataset),
        dataset_arguments=path_manager.get_raw_dataset_arguments(dataset),
        plotting_arguments=parser.get_group('Plotting'),
        target_arguments=parser.get_group('Target'),
    )


if __name__ == '__main__':
    main()
