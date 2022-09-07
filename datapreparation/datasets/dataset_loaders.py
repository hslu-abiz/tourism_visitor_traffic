# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 10.10.2019.
#  
# Project: tourism_workflow
#
# Description: 
#
from datapreparation.datasets.dataset import Dataset
from datapreparation.datasets.dataset_transformations import DateColumnExtractor, MakeNumeric, MaskNullValues

import pandas as pd
import pathlib
import time


class DataFrameLoader(Dataset):
    """Load a pandas.DataFrame."""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def load(self):
        return self.df


class CsvLoader(Dataset):
    """ Loads CSV files. """

    def __init__(
            self,
            path: pathlib.Path,
            delimiter: str,
            has_index_col: bool = False,
    ):
        super().__init__()
        self.path = path
        self.delimiter = delimiter
        self.has_index_col = has_index_col
        if not self.path.is_file():
            raise ValueError('File does not exist', self.path)

    def load(self):
        """Load the csv file and return the corresponding pandas.DataFrame."""
        # timer_start = time.process_time()
        iter_csv = pd.read_csv(
            self.path,
            sep=self.delimiter,
            iterator=True,
            index_col=0 if self.has_index_col else None,
            chunksize=10000,
        )
        df = pd.concat([chunk for chunk in iter_csv])
        # elapsed = time.process_time() - timer_start
        # load_str = "Load dataset {} took {:0.4f}sec."
        # self.log.info(load_str.format(self.path.name, elapsed))
        return df


def create_csv_dataset(
        dataset_path: pathlib.Path,
        dataset_delimiter: str,
        force_numeric: bool = False,
        date_column: str = "",
        mask_null_values: bool = False,
        has_index_col: bool = False,
) -> Dataset:
    dataset = CsvLoader(
        path=dataset_path,
        delimiter=dataset_delimiter,
        has_index_col=has_index_col,
    )
    if len(date_column) > 0:
        dataset = DateColumnExtractor(
            dataset=dataset,
            date_column_name=date_column,
        )
    dataset = MakeNumeric(dataset=dataset, force_numeric=force_numeric)
    if mask_null_values:
        dataset = MaskNullValues(dataset=dataset)
    return dataset
