# Copyright 2016 Pfäffli. All Rights Reserved.
#
# Created by Daniel Pfäffli on 01.06.17
#

#
# One to many operation
#
import pandas as pd
import numpy as np

def one_to_many_limits(df: pd.DataFrame, columns: list, limits: list) -> [pd.DataFrame]:
    """ Searches for null-values, mask them and fill with median.

    :param df: DataFrame
    :param columns: Columns
    :return: Total null-values count
             replaced columns as list of tuple (old column, new columns)
    """
    if columns is None:
        raise ValueError("Columns must be defined for this operation.")
    if df is None:
        raise ValueError("Dataframe must be defined for this operation.")
    columns_old_new = []

    for col,limit_pair in zip(columns,limits):
        values=list(range(limit_pair[0],limit_pair[1]))
        lower_limit=limit_pair[0]
        upper_limit=limit_pair[1]
        series = df[col]
        row_col_new = []
        for val in values:
            new_column_name = col + '_' + str(val)
            mask = series == val
            index = pd.np.where(mask)[0]
            if (len(index) > 0):
                df[new_column_name] = mask.astype(int)
                row_col_new = np.append(row_col_new, [new_column_name])
            else:
                df[new_column_name]=[0]*df.shape[0]
                row_col_new = np.append(row_col_new, [new_column_name])
        if len(row_col_new) > 0:
            columns_old_new.append((col, row_col_new))
        del df[col]
    return df, columns_old_new

                                                                                                                                                                                                    

def one_to_many(df: pd.DataFrame, columns: list) -> [pd.DataFrame]:
    """ Searches for null-values, mask them and fill with median.

    :param df: DataFrame
    :param columns: Columns
    :return: Total null-values count
             replaced columns as list of tuple (old column, new columns)
    """
    if columns is None:
        raise ValueError("Columns must be defined for this operation.")
    if df is None:
        raise ValueError("Dataframe must be defined for this operation.")

    columns_old_new = []
    for col in columns:
        series = df[col]
        values = series.dropna().unique()
        row_col_new = []
        for val in values:
            new_column_name = col + '_' + str(val)
            mask = series == val
            index = pd.np.where(mask)[0]
            if (len(index) > 0):
                df[new_column_name] = mask.astype(int)
                row_col_new = np.append(row_col_new, [new_column_name])

        if len(row_col_new) > 0:
            columns_old_new.append((col, row_col_new))

        del df[col]

    return df, columns_old_new
