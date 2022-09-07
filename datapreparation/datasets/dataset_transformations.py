# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 10.10.2019.
#  
# Project: tourism_workflow
#
# Description: 
#
from typing import Callable, Optional, Sequence

from datapreparation.datasets.dataset import Transformation, Dataset, Visitor
from datapreparation.datasets.helper.nullvalues import mask_nullvalues_median
from datapreparation.datasets.helper.one_to_many import one_to_many, one_to_many_limits

from datapreparation.datasets.helper.normaliser import normalise_meanstd
from datapreparation.datasets.helper.nullvalues import set_nullvalues_mean
from datapreparation.datasets.helper.shift import shift

import pandas as pd
import pathlib
import re


class ColumnNotFoundError(Exception):
    pass


class DateColumnExtractor(Transformation):
    """Extracts the date column."""
    def __init__(self, dataset:Dataset, date_column_name:str="x_c_dat"):
        super().__init__(dataset=dataset)
        self.dates = None
        self.date_column_name=date_column_name

    def load(self) -> pd.DataFrame:
        df = super().load()
        if self.date_column_name in df.columns:
            self.dates=df[self.date_column_name]
        else:
            raise ColumnNotFoundError("Could not load date column! Column name {} was not found.".format(self.date_column_name))
        return df


class MakeNumeric(Transformation):
    """Coerce to numeric."""
    def __init__(self, dataset:Dataset, force_numeric:bool=True):
        """

        :param dataset:
        :param force_numeric: If true all values are forced to be numeric or an error is raised.
        """
        super().__init__(dataset = dataset)
        self.force_numeric = force_numeric

    def load(self) -> pd.DataFrame:
        df = super().load()
        if self.force_numeric:
            self.log.info("Force to numeric...")
            df = df.apply(pd.to_numeric, errors='coerce')
        else:
            df = df.apply(pd.to_numeric, errors='ignore')
        return df


class MaskNullValues(Transformation):
    def __init__(self, dataset:Dataset):
        super().__init__(dataset=dataset)
        self.counted_null_values= 0

    def load(self):
        df = super().load()
        self.log.info("Mask null values...")
        self.counted_null_values, df = mask_nullvalues_median(df)
        self.log.info("Null values after median replacement: {:d}".format( df.isnull().sum().sum() ))
        return df


class ShiftValues(Transformation):
    """Given a time-series the value are shifted by x steps."""

    def __init__(
            self,
            dataset: Dataset,
            shift_column_names: Sequence[str],
            shift_by: Sequence[int],
            is_normalised: bool,
            transformation: Optional[Callable[[float], float]] = None
    ):
        super().__init__(dataset=dataset)
        self.shift_column_names = shift_column_names
        self.shift_by = shift_by
        self.is_normalised = is_normalised
        self.transformation = transformation

    def load(self):
        df = super().load()
        shifted = shift(df=df, columns=self.shift_column_names, shift_by=self.shift_by,
                        transformation=self.transformation)
        if self.is_normalised:
            shifted, means, stds = normalise_meanstd(shifted, columns=shifted.columns)
        shifted = set_nullvalues_mean(shifted, columns=shifted.columns)
        return pd.concat([df, shifted], axis=1)


class Checkpoint(Transformation):
    def __init__(self, dataset:Dataset, name:str):
        super().__init__(dataset = dataset)
        self.checkpoint_df = None
        self.name = name

    def load(self):
        df = super().load()
        self.checkpoint_df = df.copy(deep = True)
        return df


class OneToMany(Transformation):
    def __init__(self, dataset:Dataset, columns:list):
        super().__init__(dataset = dataset)
        self.columns = columns
        self.created_columns_tuple = None

    def load(self):
        df = super().load()
        df, tuples_columns = one_to_many(df = df, columns = self.columns)
        self.created_columns_tuple = tuples_columns
        return df


class DropMaskedColumns(Transformation):
    def load(self):
        df = super().load()
        # remove all masked column
        cols_masked = [col for col in df.columns if re.match(".*(_mask)$", col)]
        for col in cols_masked:
            del df[col]
        return df


class SaveDataFrame(Transformation):
    def __init__(self, dataset:Dataset, save_file_to: pathlib.Path,
                 delimiter: chr = ';', encoding: str = "utf-8"):
        super().__init__(dataset = dataset)
        if not save_file_to.parent.exists():
            raise ValueError("Parent directory of path must exists. Given 'save_file_to': {}".format(str(save_file_to)))
        self.save_file_to = save_file_to
        self.delimiter = delimiter
        self.encoding = encoding

    @staticmethod
    def save_dataframe(df:pd.DataFrame, save_file_to: pathlib.Path, delimiter:chr=';', encoding:str="utf-8"):
        df.to_csv(save_file_to, index = False, header = True, sep = delimiter, encoding = encoding,
                  chunksize = 10000)

    def load(self):
        df = super().load()
        self.save_dataframe(df=df, save_file_to=self.save_file_to,
                            delimiter=self.delimiter, encoding=self.encoding)
        return df

