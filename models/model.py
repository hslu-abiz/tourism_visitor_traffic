# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
from models.config_name import config_name
from models.linear_model import LinearModel
from models.dense_model import DenseModel
from models.rnn_model import RNNModel
from models.embedded_rnn import EmbeddedRNN
from models.encoded_rnn_model import EncodedRNNModel


AVAILABLE_MODELS = {
    'linear': LinearModel,
    'dense': DenseModel,
    'rnn': RNNModel,
    'embeddedrnn': EmbeddedRNN,
    'encoded_rnn': EncodedRNNModel,
}

KEYWORD_ABBREVIATIONS = {
    'model': '',
    'input_shape': 'in',
    'batch_size': 'bs',
    'seed': 'sd',
    'activation': '',
    'unit_type': '',
    'architecture': 'arch',
    'hidden_size': 'hs',
    'dropout': 'do',
    'kernel_dropout': 'kd',
    'recurrent_dropout': 'rd',
    'l1_regularization': 'l1',
    'l2_regularization': 'l2',
    'kernel_regularization': 'kr',
    'recurrent_regularization': 'rr',
    'layer_normalization': 'norm',
}

def get_name(**kwargs):
    return config_name(KEYWORD_ABBREVIATIONS, **kwargs)

def make_model(model: str = None, **kwargs):
    try:
        model_class = AVAILABLE_MODELS[model]
    except KeyError:
        raise ValueError(
            '{} is not in the list of available models, '.format(model) +
            'which are {}.'.format(AVAILABLE_MODELS)
        )
    config_with_name = {k: v for k, v in kwargs.items()}
    if 'name' not in config_with_name:
        config_with_name['name'] = get_name(model=model, **kwargs)
    model = model_class(**config_with_name)
    return model
