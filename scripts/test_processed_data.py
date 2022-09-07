import datetime

import numpy as np
import pandas as pd

import datapreparation.datasets.helper.excel_dates as excel_dates

DATASETS = ['dataset', 'dataset']
FOLDS = tuple(range(4, 9))
NUM_VALID_YEARS = 1
FIRST_YEAR = 2007


def check_days(days: np.ndarray, start_year: int, stop_year: int):
    start_datetime = datetime.datetime(year=start_year, month=1, day=1)
    stop_datetime = datetime.datetime(year=stop_year, month=1, day=1)
    start_excel_day = excel_dates.to_excel_days(start_datetime)
    stop_excel_day = excel_dates.to_excel_days(stop_datetime)
    excel_days = np.array(range(start_excel_day, stop_excel_day), dtype=days.dtype)
    for day, excel_day in zip(days, excel_days):
        if day != excel_day:
            out_day = excel_dates.from_excel_days(day)
            out_excel_day = excel_dates.from_excel_days(excel_day)
            print(f'Expected {out_excel_day}, found {out_day}.')
            return False
    return len(days) == len(excel_days)


for dataset in DATASETS:
    print(f'Checking {dataset}...')
    for num_train_years in FOLDS:
        print(f'Processing fold {num_train_years}')
        df = pd.read_csv(
            f'../data/processed/{dataset}_train_{num_train_years}ty_{NUM_VALID_YEARS}vy.csv',
            sep=';',
            index_col='x_c_dat1',
        )
        days = df.index.values
        start_year = FIRST_YEAR
        stop_year = FIRST_YEAR + num_train_years
        result = check_days(days, start_year, stop_year)
        if result:
            print(f'Train set for fold {num_train_years} ({start_year} to {stop_year-1}): OK')
        else:
            print(f'Train set for fold {num_train_years} ({start_year} to {stop_year-1}): ERROR')
        df = pd.read_csv(
            f'../data/processed/{dataset}_valid_{num_train_years}ty_{NUM_VALID_YEARS}vy.csv',
            sep=';',
            index_col='x_c_dat1',
        )
        days = df.index.values
        start_year = stop_year
        stop_year = start_year + NUM_VALID_YEARS
        result = check_days(days, start_year, stop_year)
        if result:
            print(f'Valid set for fold {num_train_years} ({start_year} to {stop_year-1}): OK')
        else:
            print(f'Valid set for fold {num_train_years} ({start_year} to {stop_year-1}): ERROR')
