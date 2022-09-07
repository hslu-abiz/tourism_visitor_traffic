# Copyright 2019 HSLU. All Rights Reserved.
#
# Created by ialionet on 20.11.2019.
#
import tensorflow as tf

from models.config_name import config_name


class Optimizer(tf.keras.optimizers.Optimizer):

    KEYWORD_ABBREVIATIONS = {
        'optimizer': '',
        'learning_rate': 'lr',
        'decay': 'dc',
        'clipvalue': 'cv',
        'clipnorm': 'cn',
        'momentum': 'mm',
        'nesterov': 'nv',
        'rho': 'rh',
        'amsgrad': 'ag',
        'beta1': 'b1',
        'beta2': 'b2',
    }

    @classmethod
    def get_name(cls, **kwargs):
        return config_name(cls.KEYWORD_ABBREVIATIONS, **kwargs)

    def __new__(cls, optimizer: str = 'SGD', **config):
        config_with_name = {k: v for k, v in config.items()}
        if 'name' not in config_with_name:
            config_with_name['name'] = cls.get_name(
                optimizer=optimizer, **config)
        return tf.keras.optimizers.deserialize({
            'class_name': optimizer,
            'config': config_with_name,
        })
