import datetime

import numpy as np
import pandas as pd

import datapreparation.datasets.helper.excel_dates as excel_dates
from training.training_configuration import METRICS

DATASETS = ['dataset', 'dataset']
TARGETS = ['target', 'target']
TEST_YEARS = tuple(range(2011, 2016))
LAGS = [1, 7, 364, 365]


def score(y_true: np.array, y_pred: np.array):
    scores = {}
    for metric_name, metric_class in METRICS.items():
        metric_object = metric_class(dtype='float64')
        if metric_name in ['PRMS', 'PLL']:
            y_pred = y_pred + 1
        metric_object.update_state(y_true, y_pred)
        scores[metric_name] = metric_object.result().numpy()
    return scores


models = ['mean', 'regularized'] + [f'lag {lag}' for lag in LAGS]
dataset_stats = []
for dataset, target in zip(DATASETS, TARGETS):
    print(f'Processing {dataset}...')
    all_scores = []
    dataset_scores = {model: [] for model in models}
    df = pd.read_csv(
        f'../data/processed/{dataset}_train_10ty_0vy.csv',
        sep=';',
        index_col='x_c_dat1',
    )
    target_values = df[target].values
    for test_year in TEST_YEARS:
        print(f'Processing test year {test_year}')
        start_datetime = datetime.datetime(year=test_year, month=1, day=1)
        stop_datetime  = datetime.datetime(year=test_year+1, month=1, day=1)
        start_excel_day = excel_dates.to_excel_days(start_datetime)
        stop_excel_day  = excel_dates.to_excel_days(stop_datetime)
        excel_days = tuple(range(start_excel_day, stop_excel_day))
        all_days = df.index.values
        for day in excel_days:
            if day not in all_days:
                print(f'day {day} ({excel_dates.from_excel_days(day)}) missing!!')
        target_values = df.loc[excel_days, target].values
        target_mask = target + '_masked'
        if target_mask in df.columns:
            target_masks = df.loc[excel_days, target_mask].values
        else:
            target_masks = np.array([0 for _ in excel_days])
        # Mean model
        previous_rows = df.loc[df.index < start_excel_day]
        previous_targets = previous_rows[target].values
        if target_mask in df.columns:
            previous_mask = previous_rows[target_mask] == 0
            previous_targets = previous_targets[previous_mask]
        start_sum = np.sum(previous_targets)
        start_count = len(previous_targets)
        y_true = np.array(target_values[np.logical_not(target_masks)], dtype='float64')
        running_means = []
        for y in y_true:
            running_means.append(float(start_sum) / float(start_count))
            start_count += 1
            start_sum += y
        y_pred = np.array(running_means, dtype='float64')
        scores = score(y_true, y_pred)
        dataset_scores['mean'].append(scores)
        # Regularized model
        y_reg = np.random.poisson(y_true)
        scores = score(y_true, y_reg)
        dataset_scores['regularized'].append(scores)
        # Lag models
        for lag in LAGS:
            lag_excel_days = tuple(range(start_excel_day-lag, stop_excel_day-lag))
            prediction_values = df.loc[lag_excel_days, target].values
            if target_mask in df.columns:
                prediction_masks = df.loc[lag_excel_days, target_mask].values
            else:
                prediction_masks = np.array([0 for _ in lag_excel_days])
            masks = np.logical_not(np.logical_and(target_masks, prediction_masks))
            y_true = np.array(target_values[masks], dtype='float64')
            y_pred = np.array(prediction_values[masks], dtype='float64')
            scores = score(y_true, y_pred)
            dataset_scores[f'lag {lag}'].append(scores)
    for model in models:
        scores_df = pd.DataFrame(data=dataset_scores[model], index=TEST_YEARS).T
        score_means = scores_df.mean(numeric_only=True, axis=1)
        score_stds = scores_df.std(numeric_only=True, axis=1)
        scores_df['mean'] = score_means
        scores_df['std'] = score_stds
        scores_df.reset_index(inplace=True)
        scores_df.insert(0, column='model', value=model)
        scores_df.rename(columns={'index': 'metric'}, inplace=True)
        all_scores.append(scores_df)
    all_scores_df = pd.concat(all_scores)
    all_scores_df.set_index(['model', 'metric'], inplace=True)
    # print(all_scores_df.to_string(float_format='%.3f', index=False))
    print(all_scores_df.to_latex(float_format='%.3f', multirow=True))
