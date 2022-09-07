# Copyright 2016 Pfäffli. All Rights Reserved.
#
# Created by Daniel Pfäffli on 01.06.17
#

#
# Provides functions for holdoutset
#
import pandas as pd
import numpy as np
import logging

def split_dataframe_by_number(df:pd.DataFrame, column:str, values:np.array)->[pd.DataFrame, pd.DataFrame]:
    """ Splits the data because of specific values in the given column
    
    :param df: Dataframe to split
    :param column: Column to look up for values
    :param values: Values to look for 
    :return: dataframe containing the specific values
             dataframe without the specific values
    """
    if not isinstance(column, str):
        raise ValueError("Column must be a string. List are not accepted.")

    df_with_values = df[df[column].isin(values)]
    df_with_values.reset_index(inplace=True, drop=True)
    df_without_values = df[~df[column].isin(values)]
    df_without_values.reset_index(inplace=True, drop=True)

    return df_with_values, df_without_values
