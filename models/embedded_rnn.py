# Copyright 2020 HSLU. All Rights Reserved.
#
# Created by tapfaeff on 08.01.2020.
#  
# Project: tourism_workflow
#
# Description: 
#
import tensorflow as tf
from typing import Optional, Sequence


class EmbeddedRNN(tf.keras.Model):

    def __init__(
            self,
            name: str,
            input_shape: Sequence[Optional[int]],
            hidden_size: int,
            outputs: Sequence[str],
            batch_size: int = 0,
            l1_regularization: float = 0.,
            l2_regularization: float = 0.,
            dropout: float = 0.
    ):
        super(EmbeddedRNN, self).__init__(name=name)
        #self.var_size = int(feature_size / hidden_size)
        #assert feature_size % hidden_size == 0 # Hidden size must be divisible without remainder

        # Attributes
        self.feature_size = input_shape[-1]
        self.hidden_size = hidden_size
        self.outputs = outputs
        self.l2_regularization = l2_regularization
        self.dropout = dropout

        # Model definition
        #     self.build(input_shape=input_shape)
        #
        #
        # def build(self, input_shape):
        #     super(EmbeddedRNN, self).__init__(input_shape=input_shape)
        # if len( input_shape ) == 2:
        #     # Only time, featuresize provided.
        #     shape = [None]
        #     shape.extend(list(input_shape))
        # elif len( input_shape ) == 3:
        #     shape = input_shape
        # else:
        #     raise ValueError(f'Unknown input_shape provided. Value: {input_shape}')
        #
        # print(self.input_shape)
        # self.__dict__["input_shape"] = shape
        # print(self.input_shape)
        self.feature_dropout_layer = tf.keras.layers.Dropout(self.dropout, input_shape=input_shape)
        self.query_layer = tf.keras.layers.Dense(self.hidden_size, input_shape=(self.feature_size,),
                                                 activation="sigmoid",
                                                 kernel_initializer="glorot_uniform",
                                                 bias_initializer="zeros"
                                                 )
        self.value_layer = tf.keras.layers.Dense(self.hidden_size, input_shape=(self.feature_size,), activation=None,
                                                 kernel_initializer="glorot_normal",
                                                 kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization)
                                                 )

        self.attention_layer = tf.keras.layers.Attention(use_scale=True, causal=False)

        self.gru = tf.keras.layers.GRU(self.hidden_size,
                                       activation="tanh",
                                       recurrent_activation="sigmoid",
                                       use_bias=True,
                                       kernel_initializer='glorot_normal',
                                       recurrent_initializer='glorot_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
                                       recurrent_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
                                       return_sequences=True,
                                       return_state=True,
                                       stateful=False,
                                       unroll=False
                                       )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            center=True,
            scale=True,
            trainable=False
        )

        self.fc1 = tf.keras.layers.Dense(self.hidden_size, use_bias=True,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=1.e-7)
        self.fc2 = {output: tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization)) for
                    output in self.outputs}
        self.relu = tf.keras.layers.ReLU()


    def call(self, x, training=None, mask=None):
        x_shape = tf.shape(x)

        x = self.feature_dropout_layer(x, training=training)

        x = tf.reshape(x, (-1, self.feature_size))
        q = self.query_layer(x, training=training)
        q = tf.reshape(q, (-1, self.hidden_size))
        v = self.value_layer(x, training=training)
        v = tf.reshape(v, (-1, self.hidden_size))

        attention = self.attention_layer([q, v], training=training)

        #attention = self.averagepooling1d(attention)
        attention = tf.reshape(attention, (-1, x_shape[1], self.hidden_size))

        # passing the concatenated vector to the GRU
        output, state = self.gru(attention, training=training)

        # Layer normalisation
        output = self.layer_norm(output, training=training)

        # x shape == (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # shape == (batch_size, max_length, hidden_size)
        output = self.fc1(output, training=training)
        output = self.leaky_relu(output, training=training)

        # output shape == (batch_size * max_length)
        output = {k: self.relu( v(output) ) for k, v in self.fc2.items()}

        output = {k: tf.reshape(v, (x_shape[0], x_shape[1], 1))
                  for k, v in output.items()}

        return output

