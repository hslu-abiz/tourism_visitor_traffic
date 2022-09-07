# Copyright 2020 HSLU. All Rights Reserved.
#
# Created by ialionet on 03.02.2020.
#
import pathlib
from typing import Optional, Union

import tensorflow as tf

from datapreparation.datasets.column_information import ColumnInformation
from training.training_configuration import CUSTOM_OBJECTS
from parsing.parser import Parser
from parsing.common_arguments import BUILD_MODEL_ARGUMENTS, PIPELINE_SHAPE_ARGUMENTS
import models.model as m


def update_pipeline_arguments_from_column_information_and_target(
        pipeline_arguments: dict,
        target_arguments: dict,
        column_information_arguments: Optional[dict] = None,
        column_information: Optional[ColumnInformation] = None
):
    if column_information is None:
        if column_information_arguments is None:
            raise ValueError('One of column_information_arguments'
                             'or column_information is required')
        column_information = ColumnInformation(**column_information_arguments)
    feature_names = column_information.get_all_feature_column_names()
    target_names = target_arguments['target_names']
    if not target_names:
        target_names = column_information.get_all_target_column_names()
    pipeline_arguments['feature_names'] = feature_names
    pipeline_arguments['target_names'] = target_names


def update_model_arguments_from_pipeline(
        model_arguments: dict,
        pipeline_arguments: dict,
):
    batch_size = pipeline_arguments.get('batch_size', None)
    model_arguments.setdefault('batch_size', batch_size)
    num_features = len(pipeline_arguments['feature_names'])
    time_steps = pipeline_arguments.get('time_steps', None)
    model_arguments.setdefault('input_shape', (time_steps, num_features))
    model_arguments.setdefault('outputs', pipeline_arguments['target_names'])


def load_model(
    model_path: Union[pathlib.Path, str]
) -> tf.keras.Model:
    with tf.keras.utils.CustomObjectScope(CUSTOM_OBJECTS):
        try:
            model = tf.keras.models.load_model(model_path)
        except OSError:
            raise OSError(
                'Could not load model '
                'from path {}.'.format(model_path)
            )
    return model


def load_model_from_checkpoint(
    model_path: Union[pathlib.Path, str],
    model_arg_file: Union[pathlib.Path, str],
    pipeline_arguments: dict
) -> tf.keras.Model:
    if model_arg_file is None:
        raise ValueError('Argument \'model_arg_file\' is none.')
    model_arg_file = pathlib.Path(model_arg_file)
    if not model_arg_file.exists():
        raise ValueError(f'Argument \'model_arg_file\' invalid. Given {model_arg_file}')
    # Load arguments from file
    parser = Parser(fromfile_prefix_chars='@')
    parser.add_group('Model', BUILD_MODEL_ARGUMENTS)

    unknown = parser.parse_known([f'@{str(model_arg_file)}'])
    model_args = parser.get_group('Model')

    # Update Model arguments
    update_model_arguments_from_pipeline(
        model_args, pipeline_arguments)

    # Construct model
    model = m.make_model(**model_args)

    shape = model_args["input_shape"]
    #model.build(input_shape=model_args["input_shape"])

    # Load weights
    model_path = pathlib.Path(model_path)
    if model_path.is_dir():
        checkpoint = tf.train.latest_checkpoint(model_path)
    else:
        checkpoint = model_path

    #print(model.summary())
    model.build(shape)
    #model.load_weights(str(checkpoint))

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(str(checkpoint)).assert_existing_objects_matched().expect_partial()

    #shape = model.feature_dropout_layer.input_shape

    return model



def update_pipeline_arguments_from_model(
        pipeline_arguments: dict,
        model: tf.keras.Model,
):
    if len(model.input_shape) != 3:
        raise ValueError(
            'The model input shape is not (batch_size, time_steps, features)'
            'but {}.'.format(str(model.input_shape))
        )
    batch_size, time_steps, features = model.input_shape
    pipeline_arguments.setdefault('batch_size', batch_size)
    pipeline_arguments.setdefault('time_steps', time_steps)



