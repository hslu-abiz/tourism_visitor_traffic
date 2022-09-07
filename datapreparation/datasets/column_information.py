# Copyright 2016 Pfäffli. All Rights Reserved.
#
# Created by Daniel Pfäffli on 01.06.17
#
from datapreparation.datasets.dataset_loaders import create_csv_dataset
from datapreparation.datasets.helper.errors import ArgumentNullError

from enum import Enum
import pandas as pd
import numpy as np

import pathlib
import re


class Operation(Enum):
    BATCH = 1
    REMOVE = 2
    TARGET = 3
    ONETOMANY = 4
    DATE = 5
    TARGET_MASK = 6


class Normalisation(Enum):
    STD = 1
    MINMAX = 2


class NullValue(Enum):
    ZERO = 1
    MEAN = 2


class ForecastOperation(Enum):
    GE = 1
    EQ = 2
    LE = 3


class SortOperation(Enum):
    ASC = 1


class ColumnInformation(object):

    def __init__(self, column_information_path: pathlib.Path, column_information_delimiter: str = ';'):
        self.path = column_information_path
        self.delimiter = column_information_delimiter
        self.column_name = "Column Header"
        self.column_forecast = "KnownDaysInAdvanced"
        self.column_operation = "Operation"
        self.column_normalisation = "Normalisation"
        self.column_nullvalue = "NullValue"
        self.column_sort = "Sort"
        self.column_mean = "mean"
        self.column_std = "std"
        self.column_min = "min"
        self.column_max = "max"
        self.column_cardinality = "cardinality"
        self.column_nan = "nan_values"
        self.normalisation_switcher = {
            Normalisation.STD: "std",
            Normalisation.MINMAX: "minmax",
        }
        self.nullvalue_switcher = {
            NullValue.ZERO: "zero",
            NullValue.MEAN: "mean",
        }
        self.operation_switcher = {
            Operation.BATCH: "batch",
            Operation.REMOVE: "remove",
            Operation.TARGET: "target",
            Operation.ONETOMANY: "onetomany",
            Operation.DATE: "date",
            Operation.TARGET_MASK: "target_mask"
        }
        self.sort_switcher = {
            SortOperation.ASC: "asc",
        }
        
        self.df = create_csv_dataset(column_information_path, column_information_delimiter).load() \
                if column_information_path is not None and column_information_path.exists() \
                else \
                None

    def set_dataframe(self, df: pd.DataFrame):
        if df is None:
            raise ArgumentNullError(argument_name="df")
        self.df = df

    def get_inforows_for_operation(self, enm_op):
        if enm_op is None:
            raise ValueError("Please give an operation!")
        cur_series = self.df[self.column_operation]
        cur_operation = self.operation_switcher.get(enm_op)
        return self.df.loc[cur_series == cur_operation, :]

    def get_column_operation(self, enm_op):
        return self.get_inforows_for_operation(enm_op = enm_op).loc[:, self.column_name].values

    def get_column_operation_onetomany_prefix(self):
        cur_series = self.df[self.column_operation]
        features = self.df[self.column_name]
        cur_operation = "onetomany"
        num_rows = self.df.shape[0]
        features_to_modify = []
        limits = []
        for row in range(0, num_rows):
            operation = str(cur_series[row])

            if operation.startswith(cur_operation + ":"):
                tokens = operation.split(":")
                features_to_modify.append(features[row])
                limit_pair = (int(tokens[1]), int(tokens[2]))
                limits.append(limit_pair)
        return features_to_modify, limits

    def get_column_normalisation(self, op: Normalisation):
        if op is None:
            raise ValueError("Please give an operation!")

        series = self.df[self.column_normalisation]
        operation = self.normalisation_switcher.get(op)

        return self.df.loc[series == operation, self.column_name].values

    def get_learnt_normalisation_values(self, op: Normalisation) -> np.ndarray:
        series = self.df[self.column_normalisation]
        operation = self.normalisation_switcher.get(op)

        if op is Normalisation.MINMAX:
            return self.df.loc[series == operation, [self.column_min, self.column_max]].values
        elif op is Normalisation.STD:
            return self.df.loc[series == operation, [self.column_mean, self.column_std]].values
        else:
            raise NotImplementedError("Operation {} is not implemented.".format(op))

    def get_column_nullvalue(self, op: NullValue):
        if op is None:
            raise ValueError("Please give an operation!")

        series = self.df[self.column_nullvalue]
        operation = self.nullvalue_switcher.get(op)

        return self.df.loc[series == operation, self.column_name].values

    def get_column_sort(self, op: SortOperation):
        if op is None:
            raise ValueError("Please give an operation!")

        series = self.df[self.column_sort]
        operation = self.sort_switcher.get(op)
        selection = series[series.str.contains('^(' + operation + '[0-9]+)$',
                                               flags = re.IGNORECASE, regex = True, na = False)].sort_values(
            ascending = True, axis = 0)
        return self.df.loc[selection.index, self.column_name].values

    def get_column_forecast(self, number_of_days_forecast: int, op: ForecastOperation = ForecastOperation.GE):
        if number_of_days_forecast is None:
            raise ValueError("Please give an operation!")

        series = self.df[self.column_forecast]

        if op == ForecastOperation.GE:
            return self.df.loc[series >= number_of_days_forecast, self.column_name].values
        elif op == ForecastOperation.EQ:
            return self.df.loc[series == number_of_days_forecast, self.column_name].values
        else:
            return self.df.loc[series <= number_of_days_forecast, self.column_name].values

    def get_column_statistics(self):
        """Get an array of all column names with at least one statistic set."""
        mean_series = self.df[self.column_mean]
        std_series = self.df[self.column_std]
        min_series = self.df[self.column_min]
        max_series = self.df[self.column_max]
        has_stats_series = (mean_series.notna() | std_series.notna() |
                            min_series.notna() | max_series.notna())
        return self.df.loc[has_stats_series, self.column_name].values

    def set_column_values(self, columns: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None,
                          min_values: np.ndarray = None, max_values: np.ndarray = None,
                          cardinality: np.ndarray =  None, nan_values: np.ndarray = None ):
        if columns is None:
            raise ValueError("Columns must be set.")
        columns = np.array(columns)
        mean = np.array(mean) if mean is not None else None
        std = np.array(std) if std is not None else None        
        min_values = np.array(min_values) if min_values is not None else None
        max_values = np.array(max_values) if max_values is not None else None
        
        assert mean is None or columns.shape[0] == mean.shape[0]
        assert std is None or columns.shape[0] == std.shape[0]
        assert min_values is None or columns.shape[0] == min_values.shape[0]
        assert max_values is None or columns.shape[0] == max_values.shape[0]
        assert cardinality is None or columns.shape[0] == cardinality.shape[0]
        assert nan_values is None or columns.shape[0] == nan_values.shape[0]
        
        data_dict = {"cols":columns}
        if mean is not None:
            data_dict.update({self.column_mean:mean})

        if std is not None:
            data_dict.update({self.column_std: std})

        if min_values is not None:
            data_dict.update({self.column_min: min_values})

        if max_values is not None:
            data_dict.update({self.column_max: max_values})
        
        if cardinality is not None:
            data_dict.update({self.column_cardinality: cardinality})
        
        if nan_values is not None:
            data_dict.update({self.column_nan: nan_values})
        
        #self.column_cardinality = "cardinality"
        #self.column_nan = "nan_values"

        stats = pd.DataFrame(data_dict)
        stats.set_index("cols", inplace=True)
        
        self.df.loc[:, "index"] = self.df.index
        self.df.set_index(self.column_name, drop=False, append=False, inplace=True)
        self.df.loc[stats.index, stats.columns] = stats.loc[stats.index, stats.columns]
        self.df.set_index("index", drop=True, inplace=True)

    def replace_column(self, column: str, new_columns: np.array):
        if column is None:
            raise ValueError("Please give column name to replace.")
        if new_columns is None:
            raise ValueError("Please give columns which replace the old one.")

        series = self.df[self.column_name]
        index = series[series == column].index
        row = self.df.loc[index, :].values
        self.df.drop(index, inplace = True)

        new_rows = np.repeat(a = row, repeats = len(new_columns), axis = 0)
        tmp_df = pd.DataFrame(new_rows, columns = self.df.columns)
        tmp_df[self.column_name] = new_columns

        self.df = self.df.append(tmp_df, ignore_index = True).reindex()

    def get_all_feature_rows(self):
        cur_series = self.df[self.column_operation]
        cur_operation = self.operation_switcher.get(Operation.ONETOMANY)

        cond = cur_series == cur_operation
        cond = cond | (cur_series == "") | (cur_series.isnull())
        return self.df.loc[cond, :]

    def get_all_feature_column_names(self):
        return self.get_all_feature_rows().loc[:, self.column_name].values

    def get_all_target_column_names(self):
        return self.get_column_operation(Operation.TARGET)

    def get_num_features(self):
        return len(self.get_all_feature_column_names())

    def get_num_targets(self):
        return len(self.get_all_target_column_names())

    def save_df(self, result_filepath: pathlib.Path):
        if result_filepath is None:
            raise ArgumentNullError(argument_name = "result_filepath")
        if not result_filepath.parent.exists():
            raise ValueError(
                "Parent directory of 'result_filepath' does not exist. "
                "Make sure that the parent directory is created before calling save_df. "
                "Argument passed: {}.".format(result_filepath)
            )
        self.df.to_csv(result_filepath, index=False, header=True, sep=';',encoding='utf-8')
