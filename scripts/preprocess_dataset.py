# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 19.11.2019.
#
import logging
import sys
import warnings

import parsing
import scripts.paths as paths
from datapreparation.datasets.preprocess_dataset import preprocess_dataset


def main():
    # Parse arguments
    parser = parsing.Parser('Split a single dataset into folds')
    parsing.add_information(parser)
    parser.add_group('Fold', parsing.FOLD_ARGUMENTS)
    parser.add_group('Dataset', parsing.SCRIPT_DATASETS_ARGUMENTS)
    parser.parse()
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    # Determine file names
    script_arguments = parser.get_group('Dataset')
    base_path = script_arguments['base_path']
    datasets = script_arguments['dataset']
    path_manager = paths.PathManager(base_path)
    fold_arguments = parser.get_group('Fold')
    nums_train_years = fold_arguments['num_train_years']
    num_valid_years = fold_arguments['num_valid_years']
    lags = fold_arguments['lags']
    for dataset in datasets:
        logger.info('Processing dataset {}'.format(dataset))
        column_information_arguments = path_manager.get_raw_column_information_arguments(dataset)
        dataset_arguments = path_manager.get_raw_dataset_arguments(dataset)
        out_column_information_paths = []
        out_train_paths = []
        out_valid_paths = []
        for num_train_years in nums_train_years:
            out_column_information_paths.append(path_manager.processed_path(
                what=paths.Processed.COLUMN_INFORMATION,
                dataset=dataset,
                num_train_years=num_train_years,
                num_valid_years=num_valid_years,
                lags=lags,
            ))
            out_train_paths.append(path_manager.processed_path(
                what=paths.Processed.TRAIN,
                dataset=dataset,
                num_train_years=num_train_years,
                num_valid_years=num_valid_years,
                lags=lags,
            ))
            out_valid_paths.append(path_manager.processed_path(
                what=paths.Processed.VALID,
                dataset=dataset,
                num_train_years=num_train_years,
                num_valid_years=num_valid_years,
                lags=lags,
            ))
        output_arguments = {
            'out_column_information_paths': out_column_information_paths,
            'out_train_paths': out_train_paths,
            'out_valid_paths': out_valid_paths,
        }
        # Preprocess datasets
        with warnings.catch_warnings():
            # Ignore all warnings due to bad pandas programming
            warnings.simplefilter('ignore')
            preprocess_dataset(
                logger=logger,
                column_information_arguments=column_information_arguments,
                dataset_arguments=dataset_arguments,
                fold_arguments=fold_arguments,
                output_arguments=output_arguments,
            )


if __name__ == '__main__':
    main()
