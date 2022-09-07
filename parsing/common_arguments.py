# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 21.11.19
#
import logging
import pathlib

from parsing.argument_types import date, shape_entry, str_key_float_value
from parsing.parser import Parser
from models import AVAILABLE_MODELS
from training.training_configuration import LOSS_DICT


def add_information(parser: Parser) -> None:
    parser.add_group('Information', group_suffix='')
    parser.add_argument('Information', 'help', '-h',
                        action='help',
                        help='Print this help message and exit.')
    parser.add_argument('Information', 'debug', '-d',
                        action="store_const", dest="log_level",
                        const=logging.DEBUG, default=logging.WARNING,
                        help="Print debugging information.")
    parser.add_argument('Information', 'verbose', '-v',
                        action="store_const", dest="log_level",
                        const=logging.INFO, default=logging.WARNING,
                        help="Print progress information.")
    parser.add_argument('Information', 'quiet', '-q',
                        action="store_const", dest="log_level",
                        const=logging.ERROR, default=logging.WARNING,
                        help="Do not print warning messages.")


DEFAULT_CSV_DELIMITER = ';'


BUILD_MODEL_ARGUMENTS = {
    'model': dict(type=str, default='linear', choices=AVAILABLE_MODELS.keys(), help='Name of the model.'),
    'input_shape': dict(type=shape_entry, nargs='+', help='Input shape for the model.'),
    'seed': dict(type=int, help='Random seed for initialization.'),
    'activation': dict(type=str, help='Activation function for dense layers.'),
    'unit_type': dict(type=str, help='Unit type for recurrent layers.'),
    'architecture': dict(type=int, nargs='+', help='Sequence of number of units per layer.'),
    'dropout': dict(type=float, help='Dropout rate of dense layer inputs, excluding the last linear layer.'),
    'hidden_size': dict(type=int, help='Size of hidden feature layer.'),
    'kernel_dropout': dict(type=float, help='Dropout rate of recursive layer inputs.'),
    'recurrent_dropout': dict(type=float, help='Dropout rate of recursive layer recurrent inputs.'),
    'l1_regularization': dict(type=float, help='L2 regularization coefficient for dense layers.'),
    'l2_regularization': dict(type=float, help='L2 regularization coefficient for dense layers.'),
    'kernel_regularization': dict(type=float, help='L2 regularization coefficient for input weights in recursive layers.'),
    'recurrent_regularization': dict(type=float, help='L2 regularization coefficient for recurrent weights in recursive layers.'),
}


TARGET_ARGUMENTS = {
    'target_names': dict(type=str, nargs='*', default=tuple(), help='Name(s) of target column(s) to consider.'),
}


COLUMN_INFORMATION_ARGUMENTS = {
    'column_information_path': dict(type=pathlib.Path, required=True, help='CSV file with the column information.'),
    'column_information_delimiter': dict(type=str, default=DEFAULT_CSV_DELIMITER, help='Delimiter in the column information CSV file.'),
}


DATASET_ARGUMENTS = {
    'dataset_path': dict(type=pathlib.Path, required=True, help='CSV file with the dataset.'),
    'dataset_delimiter': dict(type=str, default=DEFAULT_CSV_DELIMITER, help='Delimiter in the dataset CSV file.'),
}


FOLD_ARGUMENTS = {
    'num_valid_years': dict(type=int, default=1, help='Number of validation years for all folds.'),
    'num_train_years': dict(type=int, nargs='+', default=(1, ), help='Number of training years to consired for the folds.'),
    'lags': dict(type=int, nargs='*', default=tuple(), help='Lags of targets to be used as features.')
}


LOAD_MODEL_ARGUMENTS = {
    'model_paths': dict(type=str, nargs='+', help='Path to the saved model file(s) to consider.'),
    'model_arg_file': dict(type=str, nargs='+', help='Argument file for the model to instatiate.', default=None)
}


SAVE_MODEL_ARGUMENTS = {
    'model_path': dict(type=str, required=True, help='Path to save the model file.'),
}


SAVED_MODEL_ARGUMENTS = {
    'model_names': dict(type=str, nargs='+', help='String identifiers of the models to consider.'),
    'epochs': dict(type=int, nargs='+', help='Number of epochs to consider for each model.'),
}


OPTIMIZERS = ('SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam')


OPTIMIZER_ARGUMENTS = {
    'optimizer': dict(type=str, required=True, help='Name of the optimization algorithm.'),
    'learning_rate': dict(type=float, help='Learning rate of the optimizer.'),
    'decay': dict(type=float, help='Decay of the learning rate.'),
    'clipvalue': dict(type=float, help='Clipping value for gradients.'),
    'clipnorm': dict(type=float, help='Clipping norm for gradients.'),
    'momentum': dict(type=float, help='Momentum for SGD.'),
    'nesterov': dict(type=bool, help='Use Nesterov momentum for SGD.'),
    'rho': dict(type=float, help='Parameter rho for RMSprop and Adadelta.'),
    'amsgrad': dict(type=bool, help='Use amsgrad for Adam.'),
    'beta_1': dict(type=float, help='Parameter beta_1 for Adam-like optimizers.'),
    'beta_2': dict(type=float, help='Parameter beta_2 for Adam-like optimizers.'),
}


PIPELINE_SHAPE_ARGUMENTS = {
    'time_steps': dict(type=shape_entry, help='Number of time steps to feed to the model, None for the whole sequence.'),
    'batch_size': dict(type=shape_entry, help='Size of a batch of data.'),
    'prefetch': dict(type=int, default=16, help='Number of data batches to prefetch.'),
}


PIPELINE_ARGUMENTS = {
    'train_path': dict(type=pathlib.Path, required=True, help='CSV file with the train dataset.'),
    'train_delimiter': dict(type=str, default=DEFAULT_CSV_DELIMITER, help='Delimiter in the train dataset CSV file.'),
    'valid_path': dict(type=pathlib.Path, required=True, help='CSV file with the valid dataset.'),
    'valid_delimiter': dict(type=str, default=DEFAULT_CSV_DELIMITER, help='Delimiter in the valid dataset CSV file.'),
}
PIPELINE_ARGUMENTS.update(PIPELINE_SHAPE_ARGUMENTS)


PLOTTING_ARGUMENTS = {
    'show': dict(action='store_true', default=False, help='Show the plot in a window.'),
    'save': dict(type=str, default=None, help='Path to save results to, if given.'),
    'start_date': dict(type=date, default=None, help='Start date for the plot, in the format dd.mm.yyyy.'),
    'end_date': dict(type=date, default=None, help='End date for the plot, in the format dd.mm.yyyy.'),
}


PREPROCESS_OUTPUT_ARGUMENTS = {
    'out_train_paths': dict(type=pathlib.Path, nargs='+', help='Output scripts for per-fold train data.'),
    'out_valid_paths': dict(type=pathlib.Path, nargs='+', help='Output scripts for per-fold valid data.'),
    'out_column_information_paths': dict(type=pathlib.Path, nargs='+', help='Output scripts for per-fold column information.'),
}


SCRIPT_DATASET_ARGUMENTS = {
    'base_path': dict(type=pathlib.Path, default=pathlib.Path('.'), help='Root of the project directory structure.'),
    'dataset': dict(type=str, required=True, help='Name of the dataset to consider.'),
}


SCRIPT_DATASETS_ARGUMENTS = {
    'base_path': dict(type=pathlib.Path, default=pathlib.Path('.'), help='Root of the project directory structure.'),
    'dataset': dict(type=str, nargs='+', help='Name of the dataset(s) to consider.'),
}


SCRIPT_TRAINING_ARGUMENTS = {
    'epochs': dict(type=int, default=1, help='Number of epochs for the training.'),
    'loss': dict(type=str, default='MSE', choices=LOSS_DICT.keys(), help='Loss function to be optimized.'),
    'loss_weights': dict(type=str_key_float_value, default=None, help='Weights for the losses of each target in the format target1=weight1,target2=weight2,...'),
}


TRAINING_ARGUMENTS = {
    'checkpoint_path': dict(type=pathlib.Path, required=True, help='Template path for model checkpoint files.'),
    'tensorboard_path': dict(type=pathlib.Path, required=True, help='Directory to save tensorboard files to.'),
}
TRAINING_ARGUMENTS.update(SCRIPT_TRAINING_ARGUMENTS)
