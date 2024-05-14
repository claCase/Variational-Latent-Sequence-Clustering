# %%
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
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
        batch_norm=True, 
        layer_norm=False, 
        gamma=0.009, 
        beta=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        if batch_norm:
            self.bn_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.bn_x = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
        if layer_norm:
            self.ln_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.ln_x = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))

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
        default_caching_device = _caching_device(self)
        state_size = self.state_size.as_list()[0]
        self._recurrent_kernel = self.add_weight(
            "recurrent_kernel",
            shape=(state_size,),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._input_kernel = self.add_weight(
            "recurrent_kernel",
            shape=(input_shape[-1],state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._bias =  self.add_weight(
            "recurrent_kernel",
            shape=(state_size,),
            initializer=tf.keras.initializers.Zeros(),
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
    def call(self, inputs, states, training):
        states = states[0] if isinstance(states, tuple) else states

        x_h = tf.matmul(inputs, self._input_kernel)
        h_h = tf.multiply(states, self._recurrent_kernel)
        if self.batch_norm:
            x_h = self.bn_x(x_h)
            h_h = self.bn_h(h_h)
        
        if self.layer_norm:
            x_h = self.ln_x(x_h)
            h_h = self.ln_h(h_h)

        if self.dropout:
            x_h_drop = self.get_dropout_mask_for_cell(x_h, training=training, count=1) 
            x_h = x_h * x_h_drop
        
        if self.recurrent_dropout:
            h_h_drop = self.get_recurrent_dropout_mask_for_cell(h_h, training=training, count=1)
            h_h = h_h * h_h_drop 
        x = self.activation(x_h + h_h + self._bias)
        return x, x
        

class GRUCell(DropoutRNNCellMixin, tf.keras.__internal__.layers.BaseRandomLayer):
    def __init__(
        self,
        units,
        activation,
        initializer="glorot_uniform",
        regularizer=None,
        dropout=0, 
        recurrent_dropout=0,
        batch_norm=True, 
        layer_norm=False, 
        gamma=0.009, 
        beta=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        if batch_norm:
            self.bn_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.bn_x = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.bn_h_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
        if layer_norm:
            self.ln_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.ln_h_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))

        self.output_size = tf.TensorShape(units)
        self.state_size = tf.TensorShape(units)
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = dropout 
        self.recurrent_dropout = recurrent_dropout 
        self.z_act = tf.keras.activations.get(activation)
        self.r_act = tf.keras.activations.get(activation)
        self.h_act = tf.keras.activations.get("tanh")

        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)

    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        state_size = self.state_size.as_list()[0]
        self._recurrent_kernel = self.add_weight(
            "recurrent_kernel",
            shape=(state_size, 2*state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._input_kernel = self.add_weight(
            "input_kernel",
            shape=(input_shape[-1], 3*state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._gate_kernel = self.add_weight(
            "input_kernel",
            shape=(state_size, state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._bias =  self.add_weight(
            "bias_kernel",
            shape=(state_size*3),
            initializer=tf.keras.initializers.Zeros(),
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )

    def call(self, inputs, states, training):
        states = states[0] if isinstance(states, tuple) else states
        x_ = tf.matmul(inputs, self._input_kernel)
        h_ = tf.matmul(states, self._recurrent_kernel)
        if self.batch_norm:
            x_ = self.bn_x(x_)
            h_ = self.bn_h(h_)
        if self.layer_norm:
            x_ = self.ln_x(x_)
            h_ = self.ln_h(h_)
        if self.dropout:
            x_drop = self.get_dropout_mask_for_cell(x_, training=training, count=1) 
            x_ = x_ * x_drop
        if self.recurrent_dropout:
            h_drop = self.get_recurrent_dropout_mask_for_cell(h_, training=training, count=1)
            h_ = h_ * h_drop 
        x_z, x_r, x_h = tf.split(x_, 3, -1)
        h_z, h_r = tf.split(h_, 2, -1)
        z_b, r_b, h_b = tf.split(self._bias, 3, -1)

        z = self.z_act(x_z + h_z + z_b)
        r = self.r_act(x_r + h_r + r_b)
        h = self.h_act(x_h + tf.matmul(tf.multiply(r, states), self._gate_kernel) + z_b)
        if self.batch_norm:
            h = self.bn_h_h(h)
        if self.layer_norm:
            h = self.ln_h(h)
        x = (1-z)*h + z*h

        return x, x
    
class LSTMCell(DropoutRNNCellMixin, tf.keras.__internal__.layers.BaseRandomLayer):
    def __init__(
        self,
        units,
        activation="sigmoid",
        recurrent_activation="sigmoid",
        initializer="glorot_uniform",
        regularizer=None,
        dropout=0, 
        recurrent_dropout=0,
        batch_norm=True, 
        layer_norm=False, 
        gamma=0.009, 
        beta=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        if batch_norm:
            self.bn_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.bn_x = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.bn_c = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
        if layer_norm:
            self.ln_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.ln_x = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.ln_c = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))

        self.output_size = tf.TensorShape(units)
        self.state_size = tf.TensorShape(units)
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = dropout 
        self.recurrent_dropout = recurrent_dropout 
        self.i_act = tf.keras.activations.get(activation)
        self.f_act = tf.keras.activations.get(activation)
        self.o_act = tf.keras.activations.get(activation)
        self.c_act = tf.keras.activations.get("tanh")
        self.h_act = tf.keras.activations.get(recurrent_activation)

        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)

    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        state_size = self.state_size.as_list()[0]
        self._recurrent_kernel = self.add_weight(
            "recurrent_kernel",
            shape=(state_size, 4*state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._input_kernel = self.add_weight(
            "input_kernel",
            shape=(input_shape[-1], 4*state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._bias =  self.add_weight(
            "bias_kernel",
            shape=(state_size*4),
            initializer=tf.keras.initializers.Zeros(),
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )

    def call(self, inputs, states, training):
        states = states[0] if isinstance(states, tuple) else states
        
        x_ = tf.matmul(inputs, self._input_kernel)
        h_ = tf.matmul(states, self._recurrent_kernel)
        
        if self.batch_norm:
            x_ = self.bn_x(x_)
            h_ = self.bn_h(h_)
        if self.layer_norm:
            x_ = self.ln_x(x_)
            h_ = self.ln_h(h_)
        if self.dropout:
            x_drop = self.get_dropout_mask_for_cell(x_, training=training, count=1) 
            x_ = x_ * x_drop
        if self.recurrent_dropout:
            h_drop = self.get_recurrent_dropout_mask_for_cell(h_, training=training, count=1)
            h_ = h_ * h_drop 

        x_f, x_i, x_o, x_c = tf.split(x_, 4, -1)
        h_f, h_i, h_o, h_c = tf.split(h_, 4, -1)
        b_f, b_i, b_o, b_c = tf.split(self._bias, 4, -1)
        
        
        i = self.f_act(x_i + h_i + b_i)
        f = self.f_act(x_f + h_f + b_f)
        o = self.f_act(x_o + h_o + b_o)
        c = self.f_act(x_c + h_c + b_c)
        c_ = tf.multiply(f,states) + tf.multiply(i, c)
        if self.batch_norm:
            c_ = self.bn_c(c_)
        if self.layer_norm:
            c_ = self.ln_c(c_)
        h = tf.multiply(o, self.h_act(c_))
        
        return h, h
    
class SimpleRNNCell(DropoutRNNCellMixin, tf.keras.__internal__.layers.BaseRandomLayer):
    def __init__(
        self,
        units,
        activation="sigmoid",
        initializer="glorot_uniform",
        regularizer=None,
        dropout=0, 
        recurrent_dropout=0,
        batch_norm=True, 
        layer_norm=False, 
        gamma=0.009, 
        beta=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_norm = batch_norm
        self.layer_norm
        if batch_norm:
            self.bn_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.bn_x = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
        if layer_norm:
            self.ln_h = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))
            self.ln_x = layers.BatchNormalization(gamma_initializer=initializers.Constant(gamma), beta_initializer=initializers.Constant(beta))

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
        default_caching_device = _caching_device(self)
        state_size = self.state_size.as_list()[0]
        self._recurrent_kernel = self.add_weight(
            "recurrent_kernel",
            shape=(state_size,state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._input_kernel = self.add_weight(
            "recurrent_kernel",
            shape=(input_shape[-1],state_size),
            initializer=self.initializer,
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
        self._bias =  self.add_weight(
            "recurrent_kernel",
            shape=(state_size,),
            initializer=tf.keras.initializers.Zeros(),
            regularizer=self.regularizer,
            caching_device=default_caching_device,
        )
    def call(self, inputs, states, training):
        states = states[0] if isinstance(states, tuple) else states

        x_h = tf.matmul(inputs, self._input_kernel)
        h_h = tf.matmul(states, self._recurrent_kernel)
        if self.batch_norm:
            x_h = self.bn_x(x_h)
            h_h = self.bn_h(h_h)
        
        if self.layer_norm:
            x_h = self.ln_x(x_h)
            h_h = self.ln_h(h_h)

        if self.dropout:
            x_h_drop = self.get_dropout_mask_for_cell(x_h, training=training, count=1) 
            x_h = x_h * x_h_drop
        
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
