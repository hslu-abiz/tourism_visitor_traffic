# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 11.11.2019.
#  
# Project: tourism_workflow
#
# Description: 
#
import pathlib
from typing import Optional, Sequence, Union

import tensorflow as tf

from inputpipelines.csv_generator import CsvGenerator
from training.training_configuration import FLOAT_TYPE


class ColumnInformationPipeline(object):

    def __init__(
            self,
            time_steps: Optional[int],
            batch_size: Optional[int],
            prefetch: int,
            feature_names: Sequence[str],
            target_names: Sequence[str],
    ):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.feature_names = feature_names
        self.target_names = target_names

    def make_generator(
            self,
            dataset_path: Union[str, pathlib.Path],
            dataset_delimiter: str = ';',
    ) -> CsvGenerator:
        time_steps = self.time_steps
        if time_steps is None:
            time_steps = sum(1 for _ in open(dataset_path)) - 1
        return CsvGenerator(
            filenames=[dataset_path],
            feature_names=self.feature_names,
            target_names=self.target_names,
            delimiter=dataset_delimiter,
            steps=time_steps,
        )

    def make_dataset(
            self,
            dataset_path: Union[str, pathlib.Path],
            dataset_delimiter: str = ';',
    ) -> tf.data.Dataset:
        generator = self.make_generator(dataset_path, dataset_delimiter)
        dataset = tf.data.Dataset.from_generator(
            generator=generator.generate,
            output_types={k: FLOAT_TYPE for k in generator.keys},
            output_shapes=generator.output_shapes,
        )
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(self.prefetch)
        return dataset


class ColumnInformationTrainValidPipeline(ColumnInformationPipeline):

    def __init__(
            self,
            train_path: Union[str, pathlib.Path],
            valid_path: Union[str, pathlib.Path],
            train_delimiter: str = ';',
            valid_delimiter: str = ';',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.train = self.make_dataset(train_path, train_delimiter)
        self.valid = self.make_dataset(valid_path, valid_delimiter)
