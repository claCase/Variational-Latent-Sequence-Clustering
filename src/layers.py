# %%
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from tensorflow_probability.python.distributions import (
    RelaxedOneHotCategorical,
    OneHotCategorical,
    MultivariateNormalDiag,
    kl_divergence,
)
from tensorflow.python.keras.layers.recurrent import (
    DropoutRNNCellMixin,
    _config_for_enable_caching_device,
    _caching_device,
)
from tensorflow_probability.python.layers import DistributionLambda


@tf.keras.saving.register_keras_serializable(package="Variational")
class VariationalRecurrenceCell(
    DropoutRNNCellMixin, tf.keras.__internal__.layers.BaseRandomLayer
):
    def __init__(
        self,
        hidden_units,
        dropout,
        activation,
        recurrent_activation,
        rnn_type="gru",
        **kwargs,
    ):
        super(VariationalRecurrenceCell, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        if rnn_type == "gru":
            _cell = layers.GRUCell
        elif rnn_type == "lstm":
            _cell = layers.LSTMCell
        elif rnn_type == "rnn":
            _cell = layers.SimpleRNNCell
        else:
            raise NotImplementedError(
                f"rnn_type {rnn_type} invalid. Choose between rnn, lstm, gru"
            )
        self.cell = _cell(
            units=hidden_units,
            activation=activation,
            dropout=dropout,
            recurrent_dropout=dropout,
            recurrent_activation=recurrent_activation,
        )
        self.z_encoder = layers.Dense(hidden_units * 2, activation)
        self.h_encoder = layers.Dense(hidden_units, activation)
        # h, z_param, z_sample
        self.output_size = [
            tf.TensorShape(
                hidden_units,
            ),
            tf.TensorShape(
                hidden_units * 2,
            ),
            tf.TensorShape(
                hidden_units,
            ),
        ]
        self.state_size = tf.TensorShape(
            hidden_units,
        )

        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)

    @staticmethod
    def scale(scale):
        return 1e-3 + tf.math.softplus(0.05 * scale)

    def gaussian_sample(self, x):
        """
        Sample a random vector from a gaussian distribution. Tensorflow probability already reparametrizes the samples to make the parameters differentiable :)
        Inputs:
            x: Tensor of shape of shape 2*d partitioned in [mean, variance] params
        """
        dim = tf.shape(x)[-1] // 2
        distr = MultivariateNormalDiag(
            loc=x[..., :dim],
            scale_diag=self.scale(x[..., dim:]),
        )
        return tf.squeeze(distr.sample(1), axis=0)

    @tf.function
    def call(self, inputs, states, training):
        h_prime, _ = self.cell(inputs, states, training)
        z_prime = self.z_encoder(h_prime)
        z_prime_sample = self.gaussian_sample(z_prime)
        zh_concat = tf.concat([h_prime, z_prime_sample], axis=-1)
        zh_prime = self.h_encoder(zh_concat)
        return [zh_prime, z_prime, z_prime_sample], zh_prime


class IndipendentRNNCell(DropoutRNNCellMixin, tf.keras.__internal__.layers.BaseRandomLayer):
    def __init__(
        self,
        units,
        activation,
        initializer="glorot_uniform",
        regularizer=None,
        dropout=0, 
        recurrent_dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_size = tf.TensorShape(units)
        self.state_size = tf.TensorShape(units)
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = dropout 
        self.recurrent_dropout = recurrent_dropout 
        self.activation = tf.keras.activations.get(activation)

        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)

    def build(self, input_shape):
        state_size = self.state_size.as_list()[0]
        self._recurrent_kernel = self.add_weight(
            "recurrent_kernel",
            shape=(state_size,),
            initializer=self.initializer,
            regularizer=self.regularizer,
        )
        self._input_kernel = self.add_weight(
            "recurrent_kernel",
            shape=(input_shape[-1],state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
        )
        self._bias =  self.add_weight(
            "recurrent_kernel",
            shape=(state_size,),
            initializer=tf.keras.initializers.Zeros(),
            regularizer=self.regularizer,
        )
    def call(self, inputs, states, training):
        states = states[0]
        x_h = tf.matmul(inputs, self._input_kernel)
        if self.dropout:
            x_h_drop = self.get_dropout_mask_for_cell(x_h, training=training, count=1) 
            x_h = x_h * x_h_drop
        h_h = tf.multiply(states, self._recurrent_kernel)
        if self.recurrent_dropout:
            h_h_drop = self.get_recurrent_dropout_mask_for_cell(h_h, training=training, count=1)
            h_h = h_h * h_h_drop 
        x = self.activation(x_h + h_h + self._bias)
        return x, x
        


if __name__ == "__main__":
    """tf.config.run_functions_eagerly(False)
    i = np.random.normal(size=(100, 50, 10))
    vr = VariationalRecurrenceCell(15, 0.1, "relu", "linear")
    rnn = layers.RNN(vr, return_state=True, return_sequences=True)
    o = rnn(i)
    # print(o[0].shape, o[1].shape)

    gen = GenerativeVariationalMixture(15, 10, 0.1, "relu", "linear", 5)
    o = gen(i)

    inf = InferenceVariationalMixture(15, 10, 0.1, "relu", "linear", 5)
    o = inf(i, o[-1])
    """
