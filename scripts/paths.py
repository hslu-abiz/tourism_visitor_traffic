# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
from enum import Enum
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Union


_DATA_FOLDER = 'data'
_RESULTS_FOLDER = 'results'
_RAW_FOLDER = 'raw'
_PROCESSED_FOLDER = 'processed'
_DATASET_FOLDER = '{dataset}'
_RAW_FILENAME_TEMPLATE = '{dataset}{version}{what}.csv'
_PROCESSED_FILENAME_TEMPLATE = ('{dataset}_{what}_'
                                '{num_train_years}ty_{num_valid_years}vy{lags}.csv')
_TRAINING_FOLDER = ('{model}_{targets}_{loss}{loss_weights}_{optimizer}_'
                    '{num_train_years}ty_{num_valid_years}vy{lags}')
_CHECKPOINT_FILENAME_TEMPLATE = 'checkpoint.{{epoch:04d}}.hdf5'

_RAW_TEMPLATE = os.path.join(
    _DATA_FOLDER, _RAW_FOLDER, _DATASET_FOLDER, _RAW_FILENAME_TEMPLATE)
_PROCESSED_TEMPLATE = os.path.join(
    _DATA_FOLDER, _PROCESSED_FOLDER, _PROCESSED_FILENAME_TEMPLATE)
_TENSORBOARD_TEMPLATE = os.path.join(
    _RESULTS_FOLDER, _DATASET_FOLDER, _TRAINING_FOLDER)
_CHECKPOINT_TEMPLATE = os.path.join(
    _TENSORBOARD_TEMPLATE, _CHECKPOINT_FILENAME_TEMPLATE)


class Raw(Enum):
    COLUMN_INFORMATION = '_translation'
    DATA = ''


class Processed(Enum):
    COLUMN_INFORMATION = 'translation'
    TRAIN = 'train'
    VALID = 'valid'


def expected_delimiter(what: Union[Raw, Processed], dataset: str) -> str:
    comma = (isinstance(what, Raw) and what == Raw.DATA)
    return ',' if comma else ';'


def version_string(dataset: str) -> str:
    if dataset in ('dataset', 'dataset'):
        return '_features'
    return ''


class PathManager(object):

    column_information = 'translation'
    train = 'train'
    valid = 'valid'

    def __init__(
            self,
            base_path: Path,
            raw_template: str = _RAW_TEMPLATE,
            processed_template: str = _PROCESSED_TEMPLATE,
            tensorboard_template: str = _TENSORBOARD_TEMPLATE,
            checkpoint_template: str = _CHECKPOINT_TEMPLATE,
    ):
        self.raw_template = str(base_path / raw_template)
        self.processed_template = str(base_path / processed_template)
        self.tensorboard_template = str(base_path / tensorboard_template)
        self.checkpoint_template = str(base_path / checkpoint_template)

    def raw_path(self, what: Raw, dataset: str) -> Path:
        raw_path = self.raw_template.format(
            what=what.value, version=version_string(dataset), dataset=dataset,
        )
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        return Path(raw_path)

    def processed_path(
            self,
            what: Processed, dataset: str,
            num_train_years: int, num_valid_years: int, lags: Sequence[int],
    ) -> Path:
        lags_str = ('_lag' + '_'.join([str(lag) for lag in lags])) if lags else ''
        processed_path = self.processed_template.format(
            what=what.value, dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags_str,
        )
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        return Path(processed_path)

    def tensorboard_path(
            self,
            dataset: str, model: str, target_names: Sequence[str],
            loss: str, loss_weights: Optional[Dict[str, float]], optimizer: str,
            num_train_years: int, num_valid_years: int, lags: Sequence[int],
    ) -> Path:
        targets = '_'.join(target_names) if target_names else 'all'
        lags_str = ('_lag' + '_'.join([str(lag) for lag in lags])) if lags else ''
        loss_weights_str = ''
        if loss_weights:
            loss_weights_str = '_' + '_'.join(k + str(v) for k, v in loss_weights.items())
        tensorboard_path = self.tensorboard_template.format(
            dataset=dataset, model=model, targets=targets,
            loss=loss, loss_weights=loss_weights_str, optimizer=optimizer,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags_str,
        )
        os.makedirs(os.path.dirname(tensorboard_path), exist_ok=True)
        return Path(tensorboard_path)

    def checkpoint_path(
            self,
            dataset: str, model: str, target_names: Sequence[str],
            loss: str, loss_weights: Optional[Dict[str, float]], optimizer: str,
            num_train_years: int, num_valid_years: int, lags: Sequence[int],
    ) -> Path:
        targets = '_'.join(target_names) if target_names else 'all'
        lags_str = ('_lag' + '_'.join([str(lag) for lag in lags])) if lags else ''
        loss_weights_str = ''
        if loss_weights:
            loss_weights_str = '_' + '_'.join(k + str(v) for k, v in loss_weights.items())
        checkpoint_path = self.checkpoint_template.format(
            dataset=dataset, model=model, targets=targets,
            loss=loss, loss_weights=loss_weights_str, optimizer=optimizer,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags_str,
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        return Path(checkpoint_path)

    def get_raw_column_information_arguments(self, dataset: str) -> dict:
        return {
            'column_information_path': self.raw_path(
                what=Raw.COLUMN_INFORMATION, dataset=dataset),
            'column_information_delimiter': expected_delimiter(
                Raw.COLUMN_INFORMATION, dataset),
        }

    def get_raw_dataset_arguments(self, dataset: str) -> dict:
        return {
            'dataset_path': self.raw_path(what=Raw.DATA, dataset=dataset),
            'dataset_delimiter': expected_delimiter(Raw.DATA, dataset),
        }

    def get_processed_column_information_arguments(
            self, dataset: str,
            num_train_years: int, num_valid_years: int, lags: Sequence[int],
    ) -> dict:
        column_information_path = self.processed_path(
            what=Processed.COLUMN_INFORMATION, dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        column_information_delimiter = expected_delimiter(
            Processed.COLUMN_INFORMATION, dataset)
        return {
            'column_information_path': column_information_path,
            'column_information_delimiter': column_information_delimiter,
        }

    def get_processed_dataset_arguments(
            self, dataset: str,
            num_train_years: int, num_valid_years: int, lags: Sequence[int],
    ) -> dict:
        dataset_path = self.processed_path(
            what=Processed.TRAIN, dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        dataset_delimiter = expected_delimiter(Processed.TRAIN, dataset)
        return {
            'dataset_path': dataset_path,
            'dataset_delimiter': dataset_delimiter,
        }

    def update_pipeline_arguments(
            self,
            arguments: dict, dataset: str,
            num_train_years: int, num_valid_years: int, lags: Sequence[int],
    ) -> None:
        arguments['train_path'] = self.processed_path(
            what=Processed.TRAIN, dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        arguments['train_delimiter'] = expected_delimiter(
            Processed.TRAIN, dataset)
        arguments['valid_path'] = self.processed_path(
            what=Processed.VALID, dataset=dataset,
            num_train_years=num_train_years, num_valid_years=num_valid_years, lags=lags,
        )
        arguments['valid_delimiter'] = expected_delimiter(
            Processed.VALID, dataset)
