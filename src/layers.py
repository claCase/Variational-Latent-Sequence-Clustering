# %%
import tensorflow as tf
from keras import layers, models
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

# from utils import scale


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
        return [h_prime, z_prime, z_prime_sample], zh_prime


class GenerativeVariationalMixture(models.Model):
    def __init__(
        self,
        hidden_units,
        output_units,
        dropout,
        activation,
        recurrent_activation,
        clusters,
        rnn_type="gru",
        **kwargs,
    ):
        super(GenerativeVariationalMixture, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.dropout = dropout
        self.activation = activation
        self.tnn_type = rnn_type
        self.clusters = clusters

        self.P_y = OneHotCategorical(logits=[1.0] * clusters)
        self.P_zg_y = layers.Dense(hidden_units * 2, activation="linear")
        self.P_zd_h = VariationalRecurrenceCell(
            hidden_units=hidden_units,
            activation="linear",
            recurrent_activation=recurrent_activation,
            dropout=dropout,
        )
        self.P_x_zg_zd_h = layers.Dense(output_units * 2, "linear")
        # self.P_x_zg_zd_h = VariationalRecurrenceCell(units=output_units*2, return_sequences=True, return_state=True, recurrent_activation=activation, recurrent_dropout=dropout_rate)

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

    def call(self, inputs, training):
        B, T, D = inputs.get_shape().as_list()

        # Store outputs
        ta_zd_param = tf.TensorArray(
            dtype=inputs[0].dtype,
            size=T,
            element_shape=(B, self.hidden_units * 2),
        )
        ta_x_param = tf.TensorArray(
            dtype=inputs[0].dtype,
            size=T,
            element_shape=(B, self.output_units * 2),
        )

        ta_h_states = tf.TensorArray(
            dtype=inputs[0].dtype,
            size=T,
            element_shape=(B, self.hidden_units),
        )

        ta_zd_sample = tf.TensorArray(
            dtype=inputs[0].dtype, size=T, element_shape=(B, self.hidden_units)
        )
        ta_x_sample = tf.TensorArray(
            dtype=inputs[0].dtype, size=T, element_shape=(B, self.output_units)
        )

        # Compute priors
        y_sample = self.P_y.sample((B,))
        zg_y_param = self.P_zg_y(y_sample)
        zg_y_sample = self.gaussian_sample(zg_y_param)

        # Initialize recurrent state
        h_state = tf.zeros((B, *self.P_zd_h.state_size))

        for t in range(T):
            [h_state, zd_param, zd_sample], _ = self.P_zd_h(
                inputs[:, t, :], h_state, training
            )
            x_in_concat = tf.concat([zd_sample, zg_y_sample, h_state], axis=-1)
            x_param = self.P_x_zg_zd_h(x_in_concat, training=training)
            x_sample = self.gaussian_sample(x_param)
            ta_zd_param = ta_zd_param.write(t, zd_param)
            ta_zd_sample = ta_zd_sample.write(t, zd_sample)
            ta_x_param = ta_x_param.write(t, x_param)
            ta_x_sample = ta_x_sample.write(t, x_sample)
            ta_h_states = ta_h_states.write(t, h_state)
        return (
            y_sample,
            zg_y_sample,
            tf.transpose(ta_zd_sample.stack(),perm=(1, 0, 2)),
            tf.transpose(ta_x_sample.stack(), perm=(1, 0, 2)),
            zg_y_param,
            tf.transpose(ta_zd_param.stack(), perm=(1, 0, 2)),
            tf.transpose(ta_x_param.stack(), perm=(1, 0, 2)),
            tf.transpose(ta_h_states.stack(), perm=(1, 0, 2)),
        )


class RNNWithConstants(
    DropoutRNNCellMixin, tf.keras.__internal__.layers.BaseRandomLayer
):
    def __init__(
        self,
        units,
        activation,
        recurrent_activation,
        dropout,
        recurrent_dropout,
        rnn_type="gru",
        **kwargs,
    ):
        super(RNNWithConstants, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_activation = recurrent_activation
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
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout,
        )
        self.state_size = units
        self.output_size = units

    def call(self, inputs, states, constants):
        inputs = tf.concat([inputs, constants[0]], axis=-1)
        h, _ = self.cell(inputs, states)
        return h, h


class InferenceVariationalMixture(models.Model):
    def __init__(
        self,
        hidden_units,
        output_units,
        dropout,
        activation,
        recurrent_activation,
        clusters,
        **kwargs,
    ):
        super(InferenceVariationalMixture, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.activation = activation
        self.clusters = clusters
        self.output_units = output_units

        self.y_x = layers.Bidirectional(
            layers.GRU(
                units=clusters,
                return_sequences=False,
                return_state=False,
                activation="linear",
                recurrent_activation=recurrent_activation,
                recurrent_dropout=dropout,
                dropout=dropout,
            )
        )
        self.zg_x_y = layers.RNN(
            RNNWithConstants(
                activation="linear",
                units=hidden_units * 2,
                recurrent_activation=recurrent_activation,
                recurrent_dropout=dropout,
                dropout=dropout,
            ),
            return_sequences=False,
            return_state=True,
        )

        self.zd_x_h = layers.Dense(self.hidden_units * 2, activation="linear")

    def categorial_sample(self, logits, temperature=1.0, sample_shape=(1,)):
        """_summary_
        Samples from a one-hot-categorical via the gumbal-softmax reparametrization trick, samples are already reparametrized by tensorflow probability :)
        Args:
            logits (_type_): _description_
            sample_shape (tuple, optional): _description_. Defaults to (1,).

        Returns:
            _type_: _description_
        """
        cat = RelaxedOneHotCategorical(temperature=temperature, logits=logits)
        return tf.squeeze(cat.sample(sample_shape), axis=0)

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

    def call(self, inputs, h_states, training, temperature=0.5):
        y_x_param = self.y_x(inputs, training=training)
        y_x_sample = self.categorial_sample(y_x_param, temperature)
        zg_param, _ = self.zg_x_y(inputs, constants=y_x_sample, training=training)
        zg_sample = self.gaussian_sample(zg_param)
        xh = tf.concat([inputs, h_states], axis=-1)
        zd_x_h_param = self.zd_x_h(xh)
        zd_x_h_sample = self.gaussian_sample(zd_x_h_param)
        return y_x_sample, zg_sample, zd_x_h_sample, y_x_param, zg_param, zd_x_h_param


if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    i = np.random.normal(size=(100, 50, 10))
    vr = VariationalRecurrenceCell(15, 0.1, "relu", "linear")
    """rnn = layers.RNN(vr, return_state=True, return_sequences=True)
    o = rnn(i)
    # print(o[0].shape, o[1].shape)
    """
    gen = GenerativeVariationalMixture(15, 10, 0.1, "relu", "linear", 5)
    o = gen(i)
    
    inf = InferenceVariationalMixture(15, 10, 0.1, "relu", "linear", 5)
    o = inf(i, o[-1])
