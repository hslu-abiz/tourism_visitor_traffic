# Copyright 2016 Pfäffli. All Rights Reserved.
#
# Created by Daniel Pfäffli on 01.06.17
#
"""Normalization functions."""

import logging

import numpy as np
import pandas as pd


def normalise_meanstd(df: pd.DataFrame, columns: np.ndarray, means: np.ndarray = None, stds: np.ndarray = None) -> [
    pd.DataFrame, np.array, np.array]:
    """ Makes a mean, standard-deviation normalisation to mean 0 and std 1.

    :param df: Dataframe
    :param columns: Columns to perform the operation on
    :param means: Mean values for columns. If none it is estimated from data.
    :param stds: Standard deviation values for columns. If none it is estimated from data.
    :return: dataframe,
             means as np.array,
             stds as np.array
    """
    if columns is None:
        raise ValueError("Columns must be defined for this operation.")
    if df is None:
        raise ValueError("Dataframe must be defined for this operation.")
    if means is None and stds is not None or stds is None and means is not None:
        raise ValueError("Please provide valid mean and std values (parameter 'means' and 'stds'). "
                         "It is not valid to have means set but stds is None, or vice-versa.")
    if means is not None and stds is not None and means.shape[0] != stds.shape[0]:
        raise ValueError("Parameter 'means' is of length {:d} but parameter 'stds' length {:d}. "
                         "Provide two arrays of equal size.".format(means.shape[0], stds.shape[0]))

    columns = np.array(columns)
    columns2 = []
    for col in columns:
        if not col in df:
            logging.getLogger(__name__).warning("column: {} not found in dataframe.".format(col))
        else:
            columns2.append(col)

    subset_df = df[columns2].apply(pd.to_numeric, errors = 'coerce')

    # Test if one column contains all null values
    null_per_columns = subset_df.isnull().sum(axis = 0)
    cond = null_per_columns == subset_df.shape[0]
    if np.any(cond):
        # raise ValueError("Columns {} contain all None values and cannot be normalised.".format(', '.join(subset_df.columns[cond].values)))
        logging.getLogger(__name__).warning("Columns {} contain all None values and "
                                            "cannot be normalised. Mean and standard deviation "
                                            "is set to mean=0, std=1.".format(
            ', '.join(subset_df.columns[cond].values)))

    # When everything is ok we can start normalising the values
    subset = subset_df.values

    # Test if means and stds are provided.
    if means is None:
        means = np.nanmean(subset, axis = 0, dtype = np.float32)
        stds = np.nanstd(subset, axis = 0, dtype = np.float32)  # Sample std
    else:
        means = np.array(means, dtype = np.float32)
        stds = np.array(stds, dtype = np.float32)

    means[np.isnan(means)] = 0.0
    stds[np.isnan(stds)] = 1.0
    stds[stds == 0.0] = 1.0

    if means.shape[0] != subset.shape[1]:
        raise ValueError("Parameter 'means' is of length {:d} but the subset to normalize has {:d} columns. "
                         "Provide the mean and std values for all columns.".format(means.shape[0], subset.shape[1]))

    subset = (subset - means) / stds
    df[columns2] = subset

    return df, means, stds  # elementwise operations


def _min_max_scale(X: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, range: np.ndarray = np.array([-1., 1.])):
    assert len(range) == 2
    shift = (X - x_min) * (range[1] - range[0])
    scaler = x_max - x_min
    scaler[scaler == 0] = 1
    return range[0] + shift / scaler


def normalise_minmax(df: pd.DataFrame, columns: np.ndarray, min_cols: np.ndarray = None, max_cols: np.ndarray = None,
                     range: list = None) \
        -> [pd.DataFrame, np.ndarray, np.ndarray]:
    """ Makes a robust min-max normalisation to [0, 1] ignoring nans.

    :param df: Dataframe
    :param columns: Columns to perform the operation on
    :param min_cols: Min values for columns. If none it is estimated from data.
    :param max_cols: Max values for columns. If none it is estimated from data.
    :param range: Range of normalised values. Default range is [-1., 1.]
    :return: dataframe,
             min as np.array,
             max as np.array
    """
    if columns is None:
        raise ValueError("Columns must be defined for this operation.")
    if df is None:
        raise ValueError("DataFrame must be defined for this operation.")
    if min_cols is None and max_cols is not None or min_cols is None and max_cols is not None:
        raise ValueError("Please provide valid min and max values (parameter 'min_cols' and 'max_cols'). It is not "
                         "valid to have min_cols set but max_cols is None, or vice-versa.")

    if min_cols is not None and max_cols is not None and min_cols.shape[0] != max_cols.shape[0]:
        raise ValueError("Parameter 'min_cols' is of length {:d} but parameter 'max_cols' length {:d}. "
                         "Provide two array of equal size.".format(min_cols.shape[0], max_cols.shape[0]))
    # Set default values
    if range is None:
        range = [-1., 1.]

    columns = np.array(columns)
    columns2 = []
    for col in columns:
        if not col in df:
            logging.getLogger(__name__).warning("column: {} not found in dataframe.".format(col))
        else:
            columns2.append(col)

    subset_df = df[columns2].apply(pd.to_numeric, errors = 'coerce')

    # Test if one column contains all null values
    null_per_columns = subset_df.isnull().sum(axis = 0)
    cond = null_per_columns == subset_df.shape[0]
    if np.any(cond):
        logging.getLogger(__name__).warning("Columns {} contain all None values and "
                                            "cannot be normalised. Min and max value "
                                            "is set to min=0, max=1.".format(
            ', '.join(subset_df.columns[cond].values)))

    # When everything is ok we can start normalising the values
    subset = subset_df.values
    # Test if min and max values are provided.
    if min_cols is None:
        min_cols = np.nanmin(subset, axis = 0)
        max_cols = np.nanmax(subset, axis = 0)
    else:
        min_cols = np.array(min_cols, dtype = np.float32)
        max_cols = np.array(max_cols, dtype = np.float32)


    subset = _min_max_scale(X = subset, x_min = min_cols, x_max = max_cols, range = np.array(range))
    df[columns2] = subset

    return df, min_cols, max_cols  # elementwise operations
