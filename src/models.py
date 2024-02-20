#  %%
import tensorflow as tf
from keras import models
import numpy as np
import pandas as pd
from tensorflow_probability.python.distributions import (
    RelaxedOneHotCategorical,
    OneHotCategorical,
    MultivariateNormalDiag,
    kl_divergence,
)
import layers
from importlib import reload

reload(layers)


class VariationalMixtureRNN(models.Model):
    """_summary_
    Implementation of Disentangled Sequence Clustering for Human Intention Inference https://arxiv.org/abs/2101.09500
    This is a generative variational clustering model for time series. The nice property of this model is that it partitions
    the latent inferred vector z in two: one is a dynamic partition (z_d), capturing the dynamics of the time series, one is a global
    partition (z_g), capturing the cluster from which the time series is sampled.

    We have two models, one generative model P and one inference model (approximate vatiation inference model) Q, we want to maximize the Evidence Lower Bound (ELBO):
        ELBO(x) = E[log(P(x, z, y)/Q(z, y| x))]
    where the expectation is taken w.r.t the inference model Q(z, y| x).
    These models take x (the input sequence), z (the latent stochastic vector) and y (the cluster assignment) as input and decompose the joint probability as:
        1) P(x, z_g, z_d, y) = p(z_g|y)p(y) Π p(x|z_g, z_d, h_d)p(z_d|h_d) where:
            p(x|z_g, z_d, h_d) is a recurrent generative model that models the output given the latent global and dynamic vector and the hidden deterministc recurent state
            p(z_d|h_d) is a recurrent dynamic latent stochastic vector sampled given the recurrent deterministic state
            P(z_g|y) is the latent global vector given the cluster assignemt
            P(y) is the cluster prior (which for semplicity is the uniform distribution)

        2) Q(z,y|x) = q(z_g|x,y)q(y|x) Π q(z_d|x, h) where
            q(z_g|x,y) is a global latent vector assignment given the input sequence and cluster assignment
            q(y|x) is a cluster classifier given the input sequence
            q(z_d|x, h) is a latent vector inference distribution givens the latent determinitc state and input sequence


    Dimension Description:
        B: Batch dimension
        T: sequence length
        D: input sequence dimension
        C: numbert of clusters
        H: number of hidden units

    Inputs:
        x: Input sequence (B, T, d)
        xh: latent deterministic recurrent (or stochastic via drouput) x -> h: (B, T, D) -> (B, T, D)
        xz_g: Global cluster assignment x -> z_g: (B, T, D) -> C
        xz_d: latent inferred vector x -> z_d: (B, T, D) -> (B, T, D)
    Layers:

    Global Classifier (x) -> (y)
    Args:
        layers (_type_): _description_
    """

    def __init__(
        self,
        output_units,
        hidden_units=15,
        dropout_rate=0.0,
        activation="tanh",
        recurrent_activation="tanh",
        output_activation="linear",
        clusters=5,
        rnn_type="gru",
        temperature=0.5,
        **kwargs
    ):
        super(VariationalMixtureRNN, self).__init__(**kwargs)
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.drouput_rate = dropout_rate
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.output_activation = output_activation
        self.clusters = clusters
        self.temperature = temperature 
        self.zd_loss_weight = 1.0
        self.zg_loss_weight = 1.0
        self.entr_weight = .5
        

        self.generative = layers.GenerativeVariationalMixture(
            hidden_units=hidden_units,
            output_units=output_units,
            dropout=dropout_rate,
            clusters=clusters,
            activation=activation,
            recurrent_activation=recurrent_activation,
            output_activation=output_activation,
            rnn_type=rnn_type,
        )

        self.inference = layers.InferenceVariationalMixture(
            hidden_units=hidden_units,
            output_units=output_units,
            dropout=dropout_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            clusters=clusters,
        )

    def build(self, input_shape):
        super().build(input_shape=input_shape)

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
    
    def kl_z(self, Q, P):
        """
        Computes KL[Q||P]
        """
        Q = MultivariateNormalDiag(
            loc=Q[..., : self.hidden_units],
            scale_diag=self.scale(Q[..., self.hidden_units :]),
        )
        P = MultivariateNormalDiag(
            loc=P[..., : self.hidden_units],
            scale_diag=self.scale(P[..., self.hidden_units :]),
        )
        return kl_divergence(Q, P)

    @staticmethod
    def categorical_entropy(y):
        """
        Computes the cluster kl divergence KL(Q(y|x)||P(y)), since P(y) is uniform this results into computing the entropy of Q(y|x)
        """
        cat = OneHotCategorical(logits=y)
        return cat.entropy()

    def call(self, inputs, training=False, *args, **kwargs):
        generative = self.generative(inputs, training=training)
        h_states = generative["h_states"]
        inference = self.inference(inputs, h_states, training=training, temperature=self.temperature)
        return generative, inference

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            generative, inference = self.call(x, training=True)

            (
                P_y_sample,
                P_zg_sample,
                P_zd_sample,
                P_x_sample,
                P_zg_param,
                P_zd_param,
                P_x_param,
                P_h_states,
            ), (
                Q_y_sample,
                Q_zg_sample,
                Q_zd_sample,
                Q_y_x_param,
                Q_zg_param,
                Q_zd_x_h_param,
            ) = (
                generative.values(),
                inference.values(),
            )
            # E[p(x|z_g, z_d, h_d)] = log-likelihood
            ll = tf.reduce_mean(
                tf.reduce_sum(
                    MultivariateNormalDiag(
                        P_x_param[..., :-1, : self.output_units],
                        self.scale(P_x_param[..., :-1, self.output_units :]),
                    ).log_prob(x[:, 1:, :]),
                    axis=-1,
                )
            )

            # KL(zd_posterior||zd_prior)
            kl_Q_zd_x_h__P_zd_h = self.kl_z(Q_zd_x_h_param, P_zd_param)
            zd_loss = tf.reduce_mean(tf.reduce_sum(kl_Q_zd_x_h__P_zd_h, -1))

            # KL(zg_posterior||zg_prior)
            kl_Q_zg_x_y__P_zg_y = self.kl_z(Q_zg_param, P_zg_param)
            zg_loss = tf.reduce_mean(tf.reduce_sum(kl_Q_zg_x_y__P_zg_y, -1))

            # Entropy of infrence cluster assignment
            entr_y = tf.reduce_mean(
                tf.reduce_sum(self.categorical_entropy(Q_y_x_param), -1)
            )

            tot_loss = -(
                ll
                - self.zd_loss_weight * zd_loss
                - self.zg_loss_weight * zg_loss
                + self.entr_weight * entr_y
            )  # negative loss since optimizer minizies
        grad = tape.gradient(tot_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {"ll": ll, "zd_loss": zd_loss, "zg_loss": zg_loss, "entr_y": entr_y}

    @tf.function
    def sample(self, prev_inputs=None, n_samples=100, T=15):

        if prev_inputs is None:
            y_sample = self.generative.P_y.sample(n_samples)  # samples x clusters
            zg_y_param = self.generative.P_zg_y(y_sample)  # samples x d
            zg_y_sample = self.gaussian_sample(zg_y_param)

            h_state = tf.zeros(tf.TensorShape((n_samples, *self.generative.P_zd_h.state_size)))
            x_sample = tf.random.normal(shape=(n_samples, self.output_units))

        else:
            gen, inf = self(prev_inputs)
            y_sample = inf["y_x_sample"]
            zg_y_param = inf["zg_y_x_param"]
            zg_y_sample = inf["zg_y_x_sample"]
            h_state = gen["h_states"][:, -2, :]
            x_sample = prev_inputs[:, -1, :]

        B = n_samples if prev_inputs is None else prev_inputs.shape[0]
        ta_zd_param = tf.TensorArray(
            dtype=tf.float32,
            size=T,
            element_shape=tf.TensorShape((B, self.hidden_units * 2)),
        )
        ta_x_param = tf.TensorArray(
            dtype=tf.float32,
            size=T,
            element_shape=tf.TensorShape((B, self.output_units * 2)),
        )

        ta_h_states = tf.TensorArray(
            dtype=tf.float32,
            size=T,
            element_shape=tf.TensorShape((B, self.hidden_units)),
        )

        ta_zd_sample = tf.TensorArray(
            dtype=tf.float32,
            size=T,
            element_shape=tf.TensorShape((B, self.hidden_units)),
        )
        ta_x_sample = tf.TensorArray(
            dtype=tf.float32,
            size=T,
            element_shape=tf.TensorShape((B, self.output_units)),
        )

        for t in range(T):
            [h_state, zd_param, zd_sample], _ = self.generative.P_zd_h(
                x_sample, h_state, training=False
            )
            x_in_concat = tf.concat([zd_sample, zg_y_sample, h_state], axis=-1)
            x_param = self.generative.P_x_zg_zd_h(x_in_concat, training=False)
            x_sample = self.gaussian_sample(x_param)
            ta_zd_param = ta_zd_param.write(t, zd_param)
            ta_zd_sample = ta_zd_sample.write(t, zd_sample)
            ta_x_param = ta_x_param.write(t, x_param)
            ta_x_sample = ta_x_sample.write(t, x_sample)
            ta_h_states = ta_h_states.write(t, h_state)
        if prev_inputs is not None:
            return {
                "y_sample": y_sample,
                "zg_y_x_sample": zg_y_sample,
                "zd_sample": tf.concat([gen["zd_sample"], tf.transpose(ta_zd_sample.stack(), perm=(1, 0, 2))], 1),
                "x_sample": tf.concat([gen["x_sample"], tf.transpose(ta_x_sample.stack(), perm=(1, 0, 2))], 1),
                "zg_y_x_param": zg_y_param,
                "zd_param": tf.concat([gen["zd_param"], tf.transpose(ta_zd_param.stack(), perm=(1, 0, 2))], 1),
                "x_param": tf.concat([gen["x_param"], tf.transpose(ta_x_param.stack(), perm=(1, 0, 2))], 1),
                "h_states": tf.concat([gen["h_states"], tf.transpose(ta_h_states.stack(), perm=(1, 0, 2))], 1),
            }
        else:
            return {
                "y_sample": y_sample,
                "zg_y_x_sample": zg_y_sample,
                "zd_sample": tf.transpose(ta_zd_sample.stack(), perm=(1, 0, 2)),
                "x_sample": tf.transpose(ta_x_sample.stack(), perm=(1, 0, 2)),
                "zg_y_x_param": zg_y_param,
                "zd_param": tf.transpose(ta_zd_param.stack(), perm=(1, 0, 2)),
                "x_param": tf.transpose(ta_x_param.stack(), perm=(1, 0, 2)),
                "h_states":tf.transpose(ta_h_states.stack(), perm=(1, 0, 2)),
            }


class VariationalRNN(models.Model):
    def __init__(
        self,
        output_units,
        recurent_units,
        dropout=0.0,
        activation="tanh",
        recurrent_activation="tanh",
        rnn_type="gru",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cell = layers.VariationalRecurrenceCell(
            hidden_units=recurent_units,
            recurrent_activation=recurrent_activation,
            dropout=dropout,
            activation=activation,
            rnn_type=rnn_type,
        )
        self.output_units = output_units
        self.rnn = tf.keras.layers.RNN(
            self.cell, return_sequences=True, return_state=False
        )
        self.dense_output = tf.keras.layers.Dense(output_units * 2, activation="linear")
        self.kl_weight = 1.0

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

    def kl_diverge(self, x):
        dim = tf.shape(x)[-1] // 2
        distr = MultivariateNormalDiag(
            loc=x[..., :dim],
            scale_diag=self.scale(x[..., dim:]),
        )
        prior = MultivariateNormalDiag(
            tf.zeros_like(x[..., :dim]), tf.ones_like(x[..., dim:])
        )
        kl = kl_divergence(distr, prior)
        return kl

    @tf.function
    def call(self, inputs, training=False, *args):
        o = self.rnn(inputs, training=training)
        x_param = self.dense_output(o[-1])
        x_sample = self.gaussian_sample(x_param)
        return {"states": o, "x_param": x_param, "x_sample": x_sample}

    def train_step(self, data):
        with tf.GradientTape() as tape:
            o = self(data, training=True)
            z_param = o["states"][1]
            kl = tf.reduce_mean(tf.reduce_sum(self.kl_diverge(z_param), -1))
            x_param = o["x_param"]
            ll = tf.reduce_mean(
                tf.reduce_sum(
                    MultivariateNormalDiag(
                        x_param[..., :-1, : self.output_units],
                        self.scale(x_param[..., :-1, self.output_units :]),
                    ).log_prob(data[:, 1:]),
                    -1,
                )
            )
            loss = -(ll - self.kl_weight * kl)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {"ll": ll, "kl": kl}


if __name__ == "__main__":
    """i = np.random.normal(size=(100, 50, 15))
    i = tf.cast(i, tf.float32)
    vrnn = VariationalMixtureRNN(15, 10)
    o = vrnn(i)
    with tf.device("GPU:0"):
        epochs = 500
        pbar = tf.keras.utils.Progbar(epochs, )
        for e in range(epochs):
            h = vrnn.train_step(i)
            #print(f"{i}: ll: {h['ll']}  zd: {h['zd_loss']}  zg: {h['zg_loss']}  enty: {h['entr_y']}")
            pbar.update(e,values=h.items(), finalize=False)
        pbar.update(e+1, finalize=True)"""
