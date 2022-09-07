# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 19.11.2019.
#
import logging
import pathlib
import sys
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from datapreparation.datasets.helper.get_dates import get_date_column_name, get_dates
import datapreparation.datasets.dataset_transformations as dt
import datapreparation.datasets.column_information_transformations as ct
import datapreparation.datasets.dataset_loaders as dataset_loaders
from datapreparation.datasets.dataset import Dataset
from datapreparation.datasets.column_information import ColumnInformation
import parsing


def pre_split_pipeline(
        dataset: Dataset,
        column_information: ColumnInformation,
        lags: Sequence[int] = tuple(),
) -> Dataset:
    dataset = ct.ColumnVerificationCheck(dataset, column_information)
    dataset = ct.RemoveColumns(dataset, column_information)
    dataset = dt.MaskNullValues(dataset)
    dataset = ct.OneToMany(dataset, column_information)
    dataset = ct.UpdateOneToManyColumnStatistics(dataset, column_information)
    if lags:
        dataset = dt.ShiftValues(
            dataset=dataset,
            shift_column_names=column_information.get_all_target_column_names(),
            shift_by=lags,
            is_normalised=False,
            # transformation=np.log1p,
        )
    dataset = dt.Checkpoint(dataset, name="pre_split")
    return dataset


def post_split_pipeline(
        dataset: Dataset,
        column_information: ColumnInformation,
        is_validation: bool,
) -> Dataset:
    if not is_validation:
        dataset = ct.UpdateColumnStatistics(dataset, column_information)
    dataset = ct.MeanNormalisation(dataset, column_information,
                                   use_learnt_values=is_validation)
    dataset = ct.MinMaxNormalisation(dataset, column_information,
                                     use_learnt_values=is_validation)
    dataset = ct.NullValuesReplacement(dataset, column_information)
    dataset = ct.ReorderColumn(dataset, column_information)
    dataset = ct.SortColumns(dataset, column_information)
    dataset = dt.Checkpoint(dataset, name="post_split")
    return dataset


def read_dataset(
        column_information_path: pathlib.Path,
        dataset_path: pathlib.Path,
        column_information_delimiter: str,
        dataset_delimiter: str,
) -> (ColumnInformation, pd.DataFrame):
    # Read column information files
    column_information = ColumnInformation(
        column_information_path, column_information_delimiter)
    # Read the dataset
    dataset = dataset_loaders.create_csv_dataset(
        dataset_path, dataset_delimiter)
    dataframe = dataset.load()
    return column_information, dataframe


def pre_process_dataset(
        column_information: ColumnInformation,
        dataframe: pd.DataFrame,
        lags: Sequence[int] = tuple(),
) -> pd.DataFrame:
    # Read column information files
    dataset = dataset_loaders.DataFrameLoader(dataframe)
    dataset = pre_split_pipeline(dataset, column_information, lags)
    return dataset.load()


def process_dataset_fold(
        column_information: ColumnInformation,
        dataframe: pd.DataFrame,
        train_indices: Sequence[int],
        valid_indices: Sequence[int],
        out_column_information_path: pathlib.Path,
        out_train_path: pathlib.Path,
        out_valid_path: pathlib.Path,
) -> None:
    # Select the relevant rows of the dataframe
    train_dataframe = dataframe.loc[train_indices]
    valid_dataframe = dataframe.loc[valid_indices]
    train_dataset = dataset_loaders.DataFrameLoader(train_dataframe)
    valid_dataset = dataset_loaders.DataFrameLoader(valid_dataframe)
    # Normalize the data independently per-fold
    train_dataset = post_split_pipeline(train_dataset,
                                        column_information,
                                        is_validation=False)
    processed_train_dataset = train_dataset.load()
    valid_dataset = post_split_pipeline(valid_dataset,
                                        column_information,
                                        is_validation=True)
    processed_valid_dataset = valid_dataset.load()
    # Save the column information with the normalization information
    column_information.save_df(out_column_information_path)
    # Save the processed data to csv
    dt.SaveDataFrame.save_dataframe(
        df=processed_train_dataset, save_file_to=out_train_path)
    dt.SaveDataFrame.save_dataframe(
        df=processed_valid_dataset, save_file_to=out_valid_path)


def get_indices_by_year(
        year_column: pd.Series,
        years: Sequence[int],
) -> Sequence[int]:
    return year_column[year_column.isin(years)].index


def get_folds_by_year(
        year_column: pd.Series,
        nums_train_years: Sequence[int],
        num_valid_years: int,
) -> Dict[int, Dict[str, Sequence[int]]]:
    years = sorted(year_column.unique())
    fold_indices = {}
    for num_train_years in nums_train_years:
        train_years = years[:num_train_years]
        valid_years = years[num_train_years:num_train_years+num_valid_years]
        fold_indices[num_train_years] = {
            'train': get_indices_by_year(year_column, train_years),
            'valid': get_indices_by_year(year_column, valid_years),
        }
    return fold_indices


def preprocess_dataset(
        logger: logging.Logger,
        column_information_arguments: Dict[str, Any],
        dataset_arguments: Dict[str, Any],
        fold_arguments: Dict[str, Any],
        output_arguments: Dict[str, Any],
):
    # Preprocess prior to splitting
    logger.info('Pre-processing dataset')
    column_information, dataframe = read_dataset(
        **column_information_arguments, **dataset_arguments)
    # Generate folds
    # NOTE This must be done _before_ the year column is removed
    nums_train_years = fold_arguments['num_train_years']
    num_valid_years = fold_arguments['num_valid_years']
    lags = fold_arguments['lags']
    if 'x_c_y' not in dataframe.columns:
        dates = get_dates(column_information, dataframe)
        years = dates.apply(lambda x: x.year)
        dataframe['x_c_y'] = years
    year_column = dataframe['x_c_y']
    fold_dict = get_folds_by_year(
        year_column, nums_train_years, num_valid_years)
    # Start preprocessing
    dataframe = dataframe.sort_values(get_date_column_name(column_information))
    dataframe = pre_process_dataset(column_information, dataframe, lags)
    if lags:
        targets = column_information.get_all_target_column_names()
        target_rows_mask = column_information.df[column_information.column_name].isin(targets)
        target_rows = column_information.df.loc[target_rows_mask]
        lag_rows = []
        for _, target_row in target_rows.iterrows():
            for lag in lags:
                lag_row = target_row.copy()
                lag_row[column_information.column_name] += f".l{lag}"
                lag_row[column_information.column_operation] = None
                lag_rows.append(lag_row)
        column_information.df = pd.concat([column_information.df, pd.DataFrame(lag_rows)], ignore_index=True)
    # Iterate over folds
    all_out_paths = zip(
        output_arguments['out_column_information_paths'],
        output_arguments['out_train_paths'],
        output_arguments['out_valid_paths'],
    )
    for num_train_years, fold_indices in fold_dict.items():
        logger.info(
            'Processing fold with {} training '.format(num_train_years) +
            'and {} validation years'.format(num_valid_years)
        )
        try:
            out_column_path, out_train_path, out_valid_path = next(all_out_paths)
        except:
            raise ValueError('Insufficient number of paths provided.')
        process_dataset_fold(
            column_information=column_information,
            dataframe=dataframe,
            train_indices=fold_indices['train'],
            valid_indices=fold_indices['valid'],
            out_column_information_path=out_column_path,
            out_train_path=out_train_path,
            out_valid_path=out_valid_path,
        )


def main():
    # Parse arguments
    parser = parsing.Parser('Split a single dataset into folds')
    parsing.add_information(parser)
    parser.add_group('Column information', parsing.COLUMN_INFORMATION_ARGUMENTS)
    parser.add_group('Dataset', parsing.DATASET_ARGUMENTS)
    parser.add_group('Fold', parsing.FOLD_ARGUMENTS)
    parser.add_group('Output', parsing.PREPROCESS_OUTPUT_ARGUMENTS)
    parser.parse()
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    # Preprocess datasets
    preprocess_dataset(
        logger=logger,
        column_information_arguments=parser.get_group('Column information'),
        dataset_arguments=parser.get_group('Dataset'),
        fold_arguments=parser.get_group('Fold'),
        output_arguments=parser.get_group('Output')
    )


if __name__ == '__main__':
    main()
