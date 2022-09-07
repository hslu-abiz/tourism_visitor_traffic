# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 14.10.2019.
#  
# Project: tourism_workflow
#
# Description: 
#
from datapreparation.datasets.dataset import Dataset, Transformation
from datapreparation.datasets.column_information import ColumnInformation, Operation, Normalisation, \
    NullValue, SortOperation

from datapreparation.datasets.helper.normaliser import normalise_meanstd, normalise_minmax
from datapreparation.datasets.helper.one_to_many import one_to_many
from datapreparation.datasets.helper.nullvalues import set_nullvalues_mean, set_nullvalues_zero
from datapreparation.datasets.helper.split_dataframe import split_dataframe_by_number

import numpy as np
import pathlib


class ColumnInformationTransformation(Transformation):
    def __init__(self, dataset: Dataset,
                 column_information: ColumnInformation):
        super().__init__(dataset=dataset)
        self.column_information = column_information


class ColumnVerificationCheck(ColumnInformationTransformation):
    """ Verify that all columns exists. """
    def __init__(self, dataset: Dataset,
                 column_information: ColumnInformation):
        super().__init__(dataset=dataset, column_information=column_information)
        self.columns_not_in_df = []
        self.columns_not_in_verfication_list = []

    def load(self):
        df = super().load()
        raw_columns_to_verify = self.column_information.df[self.column_information.column_name]
        columns_to_verify = [str(feature).strip()
                             for feature in raw_columns_to_verify]
        # check if all columns show up in the information file
        for column_name in df.columns:
            if not str(column_name).strip() in columns_to_verify:
                self.log.warning("feature %s was not found in verification list!", column_name)
                self.columns_not_in_verfication_list.append(column_name)
        # check if verification columns show up in the dataframe
        df_cols = [str(feature).strip() for feature in df.columns]
        for column_name in columns_to_verify:
            if not str(column_name).strip() in df_cols:
                self.log.warning("Feature %s was not found in dataframe!", column_name)
                self.columns_not_in_df.append(column_name)
        return df


class RemoveColumns(ColumnInformationTransformation):
    """ Remove all mentioned columns by name. """
    def load(self):
        df = super().load()
        cols = self.column_information.get_column_operation(Operation.REMOVE)
        for val in cols:
            self.log.info("remove column: %s", val)
            if val not in df.columns:
                self.log.info("Cannot find column %s", val)
            else:
                del df[val]
        return df


class OneToMany(ColumnInformationTransformation):
    def load(self):
        df = super().load()
        col = self.column_information.get_column_operation(Operation.ONETOMANY)
        if col.shape[0] <= 0:
            return df

        df, tuples_columns = one_to_many(df=df, columns=col)

        for old, new in tuples_columns:
            self.column_information.replace_column(old, new)

        return df


class UpdateOneToManyColumnStatistics(ColumnInformationTransformation):
    def load(self):
        df = super().load()
        col = self.column_information.get_column_operation(Operation.ONETOMANY)
        if col.shape[0] <= 0:
            return df
        means = df[col].mean(axis=0, skipna=True).values
        std = df[col].std(axis=0, skipna=True).values
        min_values = df[col].min(axis=0, skipna=True).values
        max_values = df[col].max(axis=0, skipna=True).values
        self.column_information.set_column_values(columns=col,
                                                  mean=means,
                                                  std=std,
                                                  min_values=min_values,
                                                  max_values=max_values)
        return df


class UpdateColumnStatistics(ColumnInformationTransformation):
    def load(self):
        df = super().load()
        col = self.column_information.get_column_statistics()
        df_columns = df.columns
        not_updated_columns = [c for c in col if c not in df_columns]
        if not_updated_columns:
            log_str = "Will not update statistics for columns {}"
            self.log.info(log_str.format(not_updated_columns))
        col = [c for c in col if c in df_columns]
        if not col:
            return df
        means = df[col].mean(axis=0, skipna=True).values
        std = df[col].std(axis=0, skipna=True).values
        min_values = df[col].min(axis=0, skipna=True).values
        max_values = df[col].max(axis=0, skipna=True).values

        unique_count = df[col].apply(lambda x: x.dropna().unique().shape[0], axis = 0).values
        nan_values = df[col].apply(lambda x: x.isnull().sum(), axis = 0).values

        self.column_information.set_column_values(columns=col,
                                                  mean=means,
                                                  std=std,
                                                  min_values=min_values,
                                                  max_values=max_values,
                                                  cardinality = unique_count,
                                                  nan_values = nan_values)
        return df


class MinMaxNormalisation(ColumnInformationTransformation):
    def __init__(self, dataset: Dataset,
                 column_information: ColumnInformation,
                 use_learnt_values: bool):
        super().__init__(dataset=dataset, column_information=column_information)
        self.use_learnt_values = use_learnt_values

    def load(self):
        df = super().load()
        # TODO(tapfaeff, 17.10.2019): For train-test splitting the normalisation should be learnt. Therefore, it is
        # necessary that a learnt mean and std can be used instead of the sample mean and std. Make it configurable.
        col = self.column_information.get_column_normalisation(Normalisation.MINMAX)
        if col.shape[0]==0:
            self.log.info("No columns found for minmax-normalisation.")
            return df

        mins, maxs = None, None
        if self.use_learnt_values:
            mins_maxs = self.column_information.get_learnt_normalisation_values(
                Normalisation.MINMAX)
            mins = mins_maxs[:, 0]
            maxs = mins_maxs[:, 1]

        df, min_values, max_values = normalise_minmax(
            df=df, columns=col, min_cols=mins, max_cols=maxs)
        columns2 = [c for c in col if c in df]

        self.column_information.set_column_values(columns = columns2, min_values = min_values, max_values = max_values)

        return df


class MeanNormalisation(ColumnInformationTransformation):
    def __init__(self, dataset: Dataset,
                 column_information: ColumnInformation,
                 use_learnt_values: bool):
        super().__init__(dataset=dataset, column_information=column_information)
        self.use_learnt_values = use_learnt_values

    def load(self):
        df = super().load()

        col = self.column_information.get_column_normalisation(Normalisation.STD)
        if col.shape[0]==0:
            self.log.info("No columns found for standard-normalisation.")
            return df

        means, stds = None, None
        if self.use_learnt_values:
            means_stds = self.column_information.get_learnt_normalisation_values(
                Normalisation.STD)
            means = means_stds[:, 0]
            stds = means_stds[:, 1]

        df, means, stds = normalise_meanstd(
            df=df, columns=col, means=means, stds=stds)
        columns2 = [c for c in col if c in df]

        self.column_information.set_column_values(
            columns=columns2, mean=means, std=stds)

        return df


class NullValuesReplacement(ColumnInformationTransformation):
    def load(self):
        df = super().load()
        col = self.column_information.get_column_nullvalue(NullValue.ZERO)
        if col.shape[0]>0:
            df = set_nullvalues_zero(df = df, columns = col)
        col = self.column_information.get_column_nullvalue(NullValue.MEAN)
        if col.shape[0] > 0:
            df = set_nullvalues_mean(df = df, columns = col)
        return df


class ReorderColumn(ColumnInformationTransformation):
    def __init__(self, dataset:Dataset, column_information:ColumnInformation):
        super().__init__(dataset=dataset, column_information=column_information)
        self.dates = None

    def load(self):
        df = super().load()

        batch_col = self.column_information.get_column_operation(Operation.BATCH).tolist()

        date_col = self.column_information.get_column_operation(Operation.DATE)
        if date_col.shape[0] > 0:
            self.dates = df[date_col].values
            self.log.info("date column: %s", date_col)

        target_col = self.column_information.get_column_operation(Operation.TARGET)
        if target_col.shape[0] <= 0:
            raise ValueError("There must be at least one target column. Found none.")
        tmp = np.append(batch_col, date_col)

        target_mask_col = self.column_information.get_column_operation(Operation.TARGET_MASK)
        if len(target_mask_col)>0:
            tmp = np.append(tmp, target_mask_col)

        columns = df.columns
        indices_delete = [np.where(columns == col) for col in np.append(tmp, target_col)]

        feature_columns = np.delete(columns, indices_delete)
        # define new order
        column_ordered = np.append(np.append(tmp, target_col), feature_columns)

        # Effectivly reorder columns
        df = df[column_ordered]
        return df


class SortColumns(ColumnInformationTransformation):
    def load(self):
        df = super().load()
        sort_col = self.column_information.get_column_sort(SortOperation.ASC)
        if sort_col.shape[0] <= 0:
            return df # Nothing to sort

        sort_asc = [True] * len(sort_col)
        sort_col2 = []
        sort_asc2 = []
        for col, asc in zip(sort_col, sort_asc):
            if col in df.columns.values:
                sort_col2.append(col)
                sort_asc2.append(asc)

        df.sort_values(by = sort_col2, ascending = sort_asc2, axis = 0, inplace = True)
        df.reset_index(inplace = True, drop = True)
        return df


class RemoveTrains(Transformation):
    def __init__(self, dataset: Dataset, train_numbers: list, train_number_column:str):
        super().__init__(dataset = dataset)
        self.train_numbers = train_numbers
        self.train_number_column = train_number_column

    def load(self):
        df = super().load()

        _, df = split_dataframe_by_number(df = df, column = self.train_number_column, values = [2212, 2214])
        df.reset_index(inplace = True, drop = True)

        return df

class SaveColumnInformation(Transformation):
    def __init__(self, dataset: Dataset, column_information: ColumnInformation,
                 column_information_output_file: pathlib.Path):
        super().__init__(dataset = dataset)
        self.column_information = column_information
        self.column_information_output_file = column_information_output_file

    def load(self):
        df = super().load()
        self.column_information.save_df(result_filepath = self.column_information_output_file)
        return df
