# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 17.10.2019.
#  
# Project: tourism_workflow
#
# Description: 
#
from datapreparation.datasets.dataset_transformations import Checkpoint, MaskNullValues, DropMaskedColumns, ShiftValues
from datapreparation.datasets.column_information_transformations import *
from datapreparation.datasets.column_information import *

from datapreparation.datasets.helper.errors import *

import logging
import pathlib


def _create_dataset_from_raw_file(
        raw_file_path: pathlib.Path,
        column_information: ColumnInformation,
        delimiter: str = ';',
        is_validation: bool = False,
):
    dataset = create_csv_dataset(
        path_to_file=raw_file_path,
        delimiter=delimiter,
        force_numeric=False,
        mask_null_values=False,
        has_index_col=False,
    )
    dataset = ColumnVerificationCheck(dataset, column_information)
    dataset = RemoveColumns(dataset, column_information)
    dataset = MaskNullValues(dataset)
    dataset = OneToMany(dataset, column_information)
    if not is_validation:
        dataset = UpdateOneToManyColumnStatistics(dataset, column_information)
    dataset = MeanNormalisation(dataset, column_information,
                                use_learnt_values=is_validation)
    dataset = MinMaxNormalisation(dataset, column_information,
                                  use_learnt_values=is_validation)
    dataset = NullValuesReplacement(dataset, column_information)
    dataset = ReorderColumn(dataset, column_information)
    dataset = SortColumns(dataset, column_information)
    dataset = Checkpoint(dataset, name="processing")
    return dataset


def create_base_dataset(
        column_information: ColumnInformation,
        raw_file: pathlib.Path = pathlib.Path("undefined"),
        delimiter: str = ';',
        checkpoint_file: pathlib.Path = pathlib.Path("undefined"),
        save_column_information: bool = False,
        column_information_output_file: pathlib.Path = None,
        is_validation: bool = False
):
    """

    :param column_information: Filepath to column info file
    :param raw_file: Filepath of source file containing all data.
        Can be undefined if 'intermediate_processed_checkpoint_file' is set.
    :param delimiter: Delimiter used for files
    :param checkpoint_file: If set and the file exists the dataset is loaded
        without processing the raw_file.
    :param save_column_information: If True and a checkpoint is _not_ loaded,
        the updated column information is saved to 'column_information_output_file'.
    :param column_information_output_file: Filepath where the column statistics are written to.
        If None 'column_information.path' is taken.
    :param is_validation: If True all normalisation operations will use stored training values.
        Make sure these are available in the 'column_information' dataset.

    :return: dataset pipeline
    """
    if not raw_file.is_file() and not checkpoint_file.is_file():
        raise ValueError("Please provide a valid raw_file or checkpoint_file")
    log = logging.getLogger(__name__)
    # Decide whether to use an already processed state or to start from scratch
    if checkpoint_file.is_file():
        log.info("Loading pipeline from checkpoint...")
        # Use an already processed state
        date_columns = column_information.get_column_operation(enm_op=Operation.DATE)
        if len(date_columns) > 1:
            raise ValueError(
                "The column information file has {:d} date columns. Only 1 is allowed."
                .format(len(date_columns)))
        dataset = create_csv_dataset(
            path_to_file=checkpoint_file, delimiter=delimiter,
            force_numeric=False, mask_null_values=False)
    else:
        log.info("Preparing pipeline from raw data...")
        # Start from scratch
        dataset = _create_dataset_from_raw_file(
            raw_file_path=raw_file, column_information=column_information,
            delimiter=delimiter, is_validation=is_validation)
        # Save the column information file if requested
        if save_column_information:
            if column_information_output_file is None:
                column_information_output_file = column_information.path
            dataset = SaveColumnInformation(
                dataset=dataset, column_information=column_information,
                column_information_output_file=column_information_output_file)
    return dataset
