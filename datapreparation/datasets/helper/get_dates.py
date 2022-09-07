# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 25.11.2019.
#
import pandas as pd

import datapreparation.datasets.helper.excel_dates as excel_dates
from datapreparation.datasets.column_information import ColumnInformation, Operation


def get_date_column_name(column_information: ColumnInformation) -> str:
    date_columns = column_information.get_column_operation(Operation.DATE)
    if len(date_columns) != 1:
        raise ValueError('Found {} date columns.'.format(len(date_columns)))
    return date_columns[0]


def get_dates(
        column_information: ColumnInformation,
        dataframe: pd.DataFrame,
) -> pd.Series:
    date_column = get_date_column_name(column_information)
    excel_days = dataframe[date_column]
    return excel_days.apply(excel_dates.from_excel_days)
