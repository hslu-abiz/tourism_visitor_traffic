# Copyright 2016 Pfäffli. All Rights Reserved.
#
# Created by Daniel Pfäffli on 07.06.17
#

#
# Shift operation
#
from typing import Callable, Sequence, Optional

import pandas as pd
import numpy as np


def shift(
        df: pd.DataFrame,
        shift_by: Sequence[int],
        columns: Optional[Sequence[str]] = None,
        transformation: Optional[Callable[[float], float]] = None
) -> Optional[pd.DataFrame]:
    """ Shifts all columns of the given dataset by the shift_by values. Index size is kept.

    :param df: Dataset to be shifted
    :param shift_by: Sequence with the numbers of shift steps that should be taken.
    :param columns: Sequence with the names of columns to shift. If None all columns will be taken.
    :return: Dataframe with all shifted columns
    """
    if df is None:
        raise ValueError("Dataset is none.")
    if shift_by is None:
        raise ValueError("shift_by is none.")
    if type(df) is not pd.DataFrame:
        raise ValueError("Dataset is of wrong type. Must be pandas.Dataframe.")

    if columns is None:
        columns = df.columns
    existing_columns = [col in df.columns for col in columns]
    if not np.all(existing_columns):
        raise ValueError("The following columns were not found in the dataframe. "
        "Columns: {}\nFunction aborted.".format(', '.join(columns[existing_columns == False])) )

    df_new = None
    for col in columns:
        x = df[col]
        for shift in shift_by:
            colname = col + '.l' + str(shift)
            shifted_x = x.shift(shift)
            if transformation is not None:
                shifted_x = transformation(shifted_x)
            if (df_new is None):
                df_new = pd.DataFrame({colname: shifted_x})
            else:
                df_temp = pd.DataFrame({colname: shifted_x})
                df_new = pd.concat([df_new, df_temp], axis=1)
    return df_new
