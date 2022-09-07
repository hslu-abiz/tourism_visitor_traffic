# Copyright 2016 Pfäffli. All Rights Reserved.
#
# Created by Daniel Pfäffli on 01.06.17
#

#
# NullValues processing
#
from datapreparation.datasets.helper.errors import *
import pandas as pd
import logging


def mask_nullvalues_median(df: pd.DataFrame, columns: list = None) -> [int, pd.DataFrame]:
    """ Searches for null-values, mask them and fill with median.

    :param df: DataFrame
    :param columns: Columns
    :return: Total null-values count
    """
    if columns is None:
        columns = df.columns
    count_nullvalues = df.isnull().sum().sum()

    log = logging.getLogger(__name__)
    log.info("CountNullValues: %i" % count_nullvalues)

    for col in columns:
        mask = df[col].isnull()
        if mask.any():
            df[col].loc[mask] = df[col].median(skipna=True)
            df.loc[:, col + '_mask'] = mask.astype(int)

    return count_nullvalues, df


def mask_nullvalues(df: pd.DataFrame, columns: list = None) -> [int, pd.DataFrame]:
    """ Searches for null-values, mask them

    :param df: DataFrame
    :param columns: Columns
    :return: Total null-values count
    """
    if columns is None:
        columns = df.columns
    count_nullvalues = df.isnull().sum().sum()

    log = logging.getLogger(__name__)
    log.info("CountNullValues: %i" % count_nullvalues)

    for col in columns:
        mask = df[col].isnull()
        index = pd.np.where(mask)[0]
        if (len(index) > 0):
            df.loc[:, col + '_mask'] = mask.astype(int)

    return count_nullvalues, df


def _set_nullvalues_with_value_or_from_function(df: pd.DataFrame,
                                                columns: list = None,
                                                learnt_values: list = None,
                                                replacement_func = None) -> [pd.DataFrame]:
    """ Searches for null-values and fill them up with the minimum value.

    :param df: DataFrame
    :param columns: Columns to operate on. If none all columns are taken
    :param learnt_values: List of values to set as replacement of null values.
    :param replacement_func: Function of type f(x) -> R with x:pd.Series
    :return: Dataframe with replaced null values.
    """
    if df is None:
        raise ArgumentNullError(argument_name = "df")

    if columns is None:
        columns = df.columns

    if learnt_values is None:
        learnt_values = [None] * len(columns)

    if len(columns) != len(learnt_values):
        raise ValueError("Found {:d} columns but only {:d} values to set. Please provide as many values as columns or "
                         "set 'set_with_values' to none.".format(len(columns), len(learnt_values)))

    columns2 = []
    values2 = []
    for i, col in enumerate(columns):
        if col in df.columns.values:
            columns2.append(col)
            values2.append(learnt_values[i])
        else:
            logging.getLogger(__name__).warning("Column {} not found in dataframe.".format(col))

    if len(columns) <= 0:
        logging.getLogger(__name__).info(
            "No columns to perform the null values replacement. Operation skipped.")
        return df

    for i, col in enumerate(columns2):
        mask = df[col].isnull()
        index = pd.np.where(mask)[0]

        if len(index) > 0:
            df.loc[df.index[index], col] = values2[i] if values2[i] is not None else replacement_func(df[col])

    return df


def set_nullvalues_zero(df: pd.DataFrame, columns: list = None) -> [pd.DataFrame]:
    """ Searches for null-values and fill them up with 0.

    :param df: DataFrame
    :param columns: Columns
    :return: Dataframe
    """
    df = _set_nullvalues_with_value_or_from_function(
        df = df,
        columns = columns,
        replacement_func = lambda x: 0
    )

    return df


def set_nullvalues_mean(df: pd.DataFrame, columns: list = None, learnt_means:list = None ) -> [pd.DataFrame]:
    """ Searches for null-values and fill them up with the mean or the learnt means.

    :param df: DataFrame
    :param columns: Columns to operate on. If none all columns are taken
    :param learnt_means: List of values to set as replacement of null values.
    :return: Dataframe with replaced null values.
    """
    df = _set_nullvalues_with_value_or_from_function(
        df = df,
        columns = columns,
        learnt_values = learnt_means,
        replacement_func = lambda x: x.mean(skipna = True)
    )

    return df


def set_nullvalues_minimum(df: pd.DataFrame, columns: list = None, learnt_minimums: list = None) -> [pd.DataFrame]:
    """ Searches for null-values and fill them up with the minimum value.

    :param df: DataFrame
    :param columns: Columns to operate on. If none all columns are taken
    :param learnt_minimums: List of values to set as replacement of null values.
    :return: Dataframe with replaced null values.
    """

    df = _set_nullvalues_with_value_or_from_function(
        df=df,
        columns=columns,
        learnt_values = learnt_minimums,
        replacement_func = lambda x: x.min(skipna = True)
    )

    return df
