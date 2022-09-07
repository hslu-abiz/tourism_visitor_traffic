# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
import logging
import sys

import numpy as np
import pandas as pd
from fbprophet import Prophet

from datapreparation.datasets.column_information import ColumnInformation
import datapreparation.datasets.dataset_loaders as dataset_loaders
import parsing
import scripts.paths as paths
from scripts.sklearn_fit import is_basic_weather, FEATURE_ARGUMENTS, METRICS, OUTPUT_ARGUMENTS
import datapreparation.datasets.helper.excel_dates as excel_dates


def main():
    # Parse arguments
    parser = parsing.Parser('Fit a linear model baseline for a single dataset')
    parsing.add_information(parser)
    parser.add_group('Dataset', parsing.SCRIPT_DATASET_ARGUMENTS)
    parser.add_group('Fold', parsing.FOLD_ARGUMENTS)
    parser.add_group("Features", FEATURE_ARGUMENTS)
    parser.add_group("Targets", parsing.TARGET_ARGUMENTS)
    parser.add_group("Output", OUTPUT_ARGUMENTS)
    parser.parse()
    # Configure logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(parser.args.log_level)
    # Extract arguments
    dataset_arguments = parser.get_group('Dataset')
    fold_arguments = parser.get_group('Fold')
    target_arguments = parser.get_group('Targets')
    feature_arguments = parser.get_group('Features')
    output_arguments = parser.get_group('Output')
    base_path = dataset_arguments['base_path']
    dataset = dataset_arguments['dataset']
    path_manager = paths.PathManager(base_path)
    nums_train_years = fold_arguments['num_train_years']
    num_valid_years = fold_arguments['num_valid_years']
    lags = fold_arguments['lags']
    target_columns = target_arguments["target_names"]
    feature_columns = feature_arguments.get("features", None)
    feature_groups = feature_arguments.get("feature_groups", None)
    train_scores = []
    valid_scores = []
    # feature_importances = []
    # Prepare folds
    for num_train_years in nums_train_years:
        logger.info(f'Fold {num_train_years}')
        column_information_arguments = path_manager.get_processed_column_information_arguments(
            dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        column_information = ColumnInformation(**column_information_arguments)
        new_dataset_arguments = {}
        path_manager.update_pipeline_arguments(
            new_dataset_arguments,
            dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        train_dataset = dataset_loaders.create_csv_dataset(
            dataset_path=new_dataset_arguments["train_path"],
            dataset_delimiter=new_dataset_arguments["train_delimiter"],
            has_index_col=True,
        )
        train_dataframe = train_dataset.load()
        valid_dataset = dataset_loaders.create_csv_dataset(
            dataset_path=new_dataset_arguments["valid_path"],
            dataset_delimiter=new_dataset_arguments["valid_delimiter"],
            has_index_col=True,
        )
        valid_dataframe = valid_dataset.load()
        # Select features and targets
        if not target_columns:
            target_columns = list(column_information.get_all_target_column_names())
        all_feature_columns = list(column_information.get_all_feature_column_names())
        if feature_columns is None and feature_groups is None:
            feature_columns = all_feature_columns
        if feature_groups is not None:
            if feature_columns is None:
                feature_columns = []
            feature_group_dict = {
                "calendar": [f for f in all_feature_columns if f.startswith("x_c_")],
                "weather": [f for f in all_feature_columns if f.startswith("x_w_")],
                "vacation": [f for f in all_feature_columns if f.startswith("x_v_")],
                "event": [f for f in all_feature_columns if f.startswith("x_ev_")],
                "holiday": [f for f in all_feature_columns if f.startswith("x_ho_")],
                "lagged": [f for f in all_feature_columns if f[:-1].endswith(".l")],  # Breaks down for lags > 9
                "basic_weather": [f for f in all_feature_columns if is_basic_weather(f)],
            }
            for feature_group in feature_groups:
                feature_columns += feature_group_dict[feature_group]
            feature_columns = list(set(feature_columns))
        all_columns = feature_columns + target_columns
        train_dataframe = train_dataframe[all_columns]
        valid_dataframe = valid_dataframe[all_columns]
        train_datetime_index = np.array([excel_dates.from_excel_days(x) for x in train_dataframe.index])
        train_dataframe.reset_index()
        train_dataframe["ds"] = train_datetime_index
        train_dataframe.rename(columns={target_columns[0]: "y"}, inplace=True)
        # print(train_dataframe.columns)
        # print(columns_to_drop)
        # for column in columns_to_drop:
        #     print(column in train_dataframe.columns)
        valid_datetime_index = np.array([excel_dates.from_excel_days(x) for x in valid_dataframe.index])
        valid_dataframe.reset_index()
        valid_dataframe["ds"] = valid_datetime_index
        valid_dataframe.rename(columns={target_columns[0]: "y"}, inplace=True)
        # Define model, perform fit and evaluate
        fold_model = Prophet()
        for feature_column in feature_columns:
            fold_model.add_regressor(feature_column)
        fold_model.fit(train_dataframe)
        train_valid_dataframe = pd.concat([train_dataframe, valid_dataframe])
        train_valid_features = train_valid_dataframe.copy().drop(columns=["y"])
        # The following lines are to check that features in the future are not being used for predictions
        # forecast = []
        # train_len = len(train_dataframe)
        # for valid_len in range(1, len(valid_dataframe)+1):
        #     rolling_len = train_len + valid_len
        #     rolling_features = train_valid_features.iloc[:rolling_len]
        #     forecast.append(fold_model.predict(rolling_features).iloc[-1])
        #     if valid_len == 10:
        #         break
        # forecast = pd.DataFrame(forecast)
        # print(forecast)
        # forecast = fold_model.predict(train_valid_features[train_len:train_len+10])
        # print(forecast)
        # abort
        forecast = fold_model.predict(train_valid_features)
        # import matplotlib.pyplot as plt
        # fold_model.plot(forecast)
        # plt.scatter(valid_dataframe["ds"], valid_dataframe["y"], s=4, c="red")
        # plt.show()
        train_predictions = forecast.loc[forecast["ds"].isin(train_datetime_index)]["yhat"]
        fold_train_scores = {
            metric_name: metric_function(y_pred=train_predictions, y_true=train_dataframe["y"])
            for metric_name, metric_function in METRICS.items()
        }
        train_scores.append(fold_train_scores)
        valid_predictions = forecast.loc[forecast["ds"].isin(valid_datetime_index)]["yhat"]
        fold_valid_scores = {
            metric_name: metric_function(y_pred=valid_predictions, y_true=valid_dataframe["y"])
            for metric_name, metric_function in METRICS.items()
        }
        valid_scores.append(fold_valid_scores)
    feature_groups = ["all", ] if feature_groups is None else feature_groups
    train_scores_df = pd.DataFrame(train_scores)
    train_scores_df["dataset"] = dataset
    train_scores_df["target"] = target_columns[0] if dataset == "dataset" else dataset
    train_scores_df["model"] = "prophet"
    train_scores_df["features"] = "_".join(feature_groups)
    if output_arguments["print_train"]:
        print("Training:")
        print(train_scores_df.to_string())
    if output_arguments["print_train_stats"]:
        print("Training stats:")
        print(train_scores_df.describe().loc[["mean", "std"]])
    valid_scores_df = pd.DataFrame(valid_scores)
    valid_scores_df["dataset"] = dataset
    valid_scores_df["target"] = target_columns[0] if dataset == "dataset" else dataset
    valid_scores_df["model"] = "prophet"
    valid_scores_df["features"] = "_".join(feature_groups)
    if output_arguments["print_valid"]:
        print("Validation:")
        print(valid_scores_df.to_string())
    if output_arguments["print_valid_stats"]:
        print("Validation stats:")
        print(valid_scores_df.describe().loc[["mean", "std"]])


if __name__ == '__main__':
    main()
