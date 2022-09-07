# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
import logging
import sys
from typing import Callable

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.tree
from tqdm import tqdm
from xgboost import XGBRegressor

from datapreparation.datasets.column_information import ColumnInformation
import datapreparation.datasets.dataset_loaders as dataset_loaders
import parsing
import scripts.paths as paths


def rmse(y_pred, y_true):
    return -sklearn.metrics.mean_squared_error(y_pred=y_pred, y_true=y_true) ** 0.5


metric_type = Callable[[np.ndarray, np.ndarray], float]

METRICS = {
    "MAE": sklearn.metrics.mean_absolute_error,
    "RMSE": rmse,
    "R2": sklearn.metrics.r2_score,
}

MODELS = {
    "linear": sklearn.linear_model.LinearRegression,
    "lasso": sklearn.linear_model.MultiTaskLasso,
    "tree": sklearn.tree.DecisionTreeRegressor,
    "xgboost": XGBRegressor,
    "lgbm": LGBMRegressor,
}

MODEL_ARGUMENTS = {
    'model': dict(type=str, default='linear', choices=MODELS.keys(), help='Name of the model.'),
    'importance': dict(action='store_true', default=False, help='Report feature importance results.'),
    'alpha': dict(type=float, help='L1 regularization weight.'),
    'max_iter': dict(type=int, help='Max number of iterations.'),
    'max_depth': dict(type=int, help='Max depth of trees, default = 6.'),
    'min_child_weight': dict(type=float, help='Minimum sum of instance weight (hessian) needed in a child, default = 1.0.'),
    'eta': dict(type=float, help='Step size shrinkage used in update to prevents overfitting, default = 0.3.'),
    'subsample': dict(type=float, help='Subsample ratio of the training instances, default = 0.0.'),
    'min_impurity_decrease': dict(type=float, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value.'),
    'boosting_type': dict(type=str, help='Boosting type for LightGBM.')
}

FEATURE_ARGUMENTS = {
    'features': dict(type=str, nargs='*', help='Features to be used.'),
    'feature_groups': dict(type=str, nargs='*', help='Feature groups to be used.'),
    'fakes': dict(action='store_true', default=False, help='Add fake train_features using random shuffling.'),
}

OUTPUT_ARGUMENTS = {
    'print_train': dict(action='store_true', default=False, help='Print training scores.'),
    'print_valid': dict(action='store_true', default=False, help='Print validation scores.'),
    'print_train_stats': dict(action='store_true', default=False, help='Print training score statistics.'),
    'print_valid_stats': dict(action='store_true', default=False, help='Print validation score statistics.'),
}


def is_basic_weather(name: str) -> bool:
    prefixes = ["x_w_11_", "x_w_21_", "x_w_41_", "x_w_51_"]
    suffixes = ["_l0"]
    prefix = None
    for x in prefixes:
        if name.startswith(x):
            prefix = x
    if prefix is None:
        return False
    suffix = None
    for x in suffixes:
        if name.endswith(x):
            suffix = x
    if suffix is None:
         return False
    return len(name) == len(f"{prefix}xxxx{suffix}")


def add_fake_columns(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    new_df = df.copy()
    for column in df.columns:
        data = df[column].values.copy()
        rng.shuffle(data)
        new_df[column + "_fake"] = data
    return new_df


def backward_selection(
    model: sklearn.base.RegressorMixin,
    train_features: pd.DataFrame,
    train_targets: pd.DataFrame,
    valid_features: pd.DataFrame,
    valid_targets: pd.DataFrame,
    metric: metric_type = METRICS["R2"],
):
    current_columns = [col for col in train_features.columns]
    while len(current_columns):
        current_scores = {}
        for column_left_out in tqdm(current_columns):
            new_columns  = [col for col in current_columns if col != column_left_out]
            new_train_features = train_features[new_columns]
            model.fit(new_train_features, train_targets)
            new_valid_features = valid_features[new_columns]
            predictions = model.predict(new_valid_features)
            current_scores[column_left_out] = metric(valid_targets, predictions)
        column_to_remove = max(current_scores, key=current_scores.get)
        print(f"Least important feature is {column_to_remove}, removing it decreases score to {current_scores[column_to_remove]}")
        current_columns = [col for col in current_columns if col != column_to_remove]


def forward_selection(
    model: sklearn.base.RegressorMixin,
    train_features: pd.DataFrame,
    train_targets: pd.DataFrame,
    valid_features: pd.DataFrame,
    valid_targets: pd.DataFrame,
    metric: metric_type = METRICS["R2"],
):
    missing_columns = [col for col in train_features.columns]
    while len(missing_columns):
        current_scores = {}
        for column_to_add in tqdm(missing_columns):
            new_columns  = [col for col in train_features.columns if col not in missing_columns or col == column_to_add]
            new_train_features = train_features[new_columns]
            model.fit(new_train_features, train_targets)
            new_valid_features = valid_features[new_columns]
            predictions = model.predict(new_valid_features)
            current_scores[column_to_add] = metric(valid_targets, predictions)
        column_to_add = max(current_scores, key=current_scores.get)
        print(f"Most important feature is {column_to_add}, adding it increases score to {current_scores[column_to_add]}")
        missing_columns = [col for col in missing_columns if col != column_to_add]


def main():
    # Parse arguments
    parser = parsing.Parser('Fit a linear model baseline for a single dataset')
    parsing.add_information(parser)
    parser.add_group('Dataset', parsing.SCRIPT_DATASET_ARGUMENTS)
    parser.add_group('Fold', parsing.FOLD_ARGUMENTS)
    parser.add_group("Model", MODEL_ARGUMENTS)
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
    model_arguments = parser.get_group('Model')
    target_arguments = parser.get_group('Targets')
    feature_arguments = parser.get_group('Features')
    output_arguments = parser.get_group('Output')
    base_path = dataset_arguments['base_path']
    dataset = dataset_arguments['dataset']
    path_manager = paths.PathManager(base_path)
    nums_train_years = fold_arguments['num_train_years']
    num_valid_years = fold_arguments['num_valid_years']
    lags = fold_arguments['lags']
    model = model_arguments.pop('model')
    importance = model_arguments.pop("importance")
    target_columns = target_arguments["target_names"]
    feature_columns = feature_arguments.get("features", None)
    feature_groups = feature_arguments.get("feature_groups", None)
    use_fakes = feature_arguments["fakes"]
    train_scores = []
    valid_scores = []
    feature_importances = []
    # Prepare folds
    for num_train_years in nums_train_years:
        logger.info(f'Fold {num_train_years}')
        # Read column information, train and valid dataframes
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
        train_targets = train_dataframe[target_columns]
        valid_targets = valid_dataframe[target_columns]
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
        train_features = train_dataframe[feature_columns]
        valid_features = valid_dataframe[feature_columns]
        # Prepare fake, shuffled features if requested
        if use_fakes:
            train_features = add_fake_columns(train_features)
            valid_features = add_fake_columns(valid_features)
        # Define model, perform fit and evaluate
        fold_model = MODELS[model](**model_arguments)
        # forward_selection(fold_model, train_features, train_targets, valid_features, valid_targets)
        # abort
        fold_model.fit(train_features, train_targets)
        train_predictions = fold_model.predict(train_features)
        fold_train_scores = {
            metric_name: metric_function(y_pred=train_predictions, y_true=train_targets)
            for metric_name, metric_function in METRICS.items()
        }
        train_scores.append(fold_train_scores)
        valid_predictions = fold_model.predict(valid_features)
        fold_valid_scores = {
            metric_name: metric_function(y_pred=valid_predictions, y_true=valid_targets)
            for metric_name, metric_function in METRICS.items()
        }
        valid_scores.append(fold_valid_scores)
        if importance:
            try:
                feature_importances.append(dict(zip(train_features.columns, fold_model.feature_importances_)))
            except:
                feature_importances.append(dict(zip(train_features.columns, fold_model.coef_)))
    feature_groups = ["all", ] if feature_groups is None else feature_groups
    train_scores_df = pd.DataFrame(train_scores)
    train_scores_df["dataset"] = dataset
    train_scores_df["target"] = target_columns[0] if dataset == "dataset" else dataset
    train_scores_df["model"] = model
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
    valid_scores_df["model"] = model
    valid_scores_df["features"] = "_".join(feature_groups)
    if output_arguments["print_valid"]:
        print("Validation:")
        print(valid_scores_df.to_string())
    if output_arguments["print_valid_stats"]:
        print("Validation stats:")
        print(valid_scores_df.describe().loc[["mean", "std"]])
    if importance:
        feature_importances_df = pd.DataFrame(feature_importances)
        print("Feature importance:")
        fm = feature_importances_df.mean().sort_values()
        important_columns = []
        for index, value in fm.iteritems():
            print(index, value, feature_importances_df[index].values)
            if value > 1e-3:
                important_columns.append(index)
        print(" ".join(important_columns[::-1]))


if __name__ == '__main__':
    main()
