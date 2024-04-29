#  %%
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pandas as pd
from tensorflow_probability.python.distributions import (
    RelaxedOneHotCategorical,
    OneHotCategorical,
    MultivariateNormalDiag,
    kl_divergence,
)
from src.utils import scale
from src import layers
from importlib import reload

reload(layers)

tfkl = tf.keras.layers


class DiscreteVariationalMixtureRNN(models.Model):
    """Implementation Discrete Variational Mixture RNN from https://arxiv.org/abs/2101.09500

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

    """

    def __init__(
        self,
        output_units,
        hidden_units=15,
        latent_dynamic_units=10,
        latent_global_units=10,
        activation="tanh",
        recurrent_activation="tanh",
        output_activation="linear",
        clusters=5,
        rnn_type="gru",
        dropout=False,
        recurrent_dropout=False,
        temperature=1,
        use_sample_kl=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.latend_dynamic_units = latent_dynamic_units
        self.latend_global_units = latent_global_units
        self.use_kl_sample = use_sample_kl
        self.clusters = clusters
        self.temperature = temperature
        self.kl_weight = 1.
        if rnn_type == "gru":
            _cell = tfkl.GRUCell
        elif rnn_type == "lstm":
            _cell = tfkl.LSTMCell
        elif rnn_type == "rnn":
            _cell = tfkl.SimpleRNNCell
        else:
            raise NotImplementedError(
                f"rnn_type {rnn_type} invalid. Choose between rnn, lstm, gru"
            )
        self._rnn_cell = _cell(
            units=hidden_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout,
        )
        # Bi-RNN used to infer posterior over sequential time series
        self._birnn = models.Sequential(
            [
                tfkl.Bidirectional(
                    tfkl.RNN(
                        _cell(
                            units=hidden_units,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            dropout=dropout,
                            recurrent_dropout=dropout,
                        ),
                        return_state=False,
                        return_sequences=False,
                    )
                ),
                tfkl.Dense(clusters, activation),
            ]
        )
        self.state_size = hidden_units
        self.output_size = output_units
        self.y_given_h = tfkl.Dense(clusters, activation="linear")
        self.zg_given_h_y_posterior = tfkl.Dense(
            latent_global_units * 2, activation="linear"
        )
        self.zg_given_y_prior = tfkl.Dense(latent_global_units * 2, activation="linear")
        self.zd_given_h_prior = tfkl.Dense(
            latent_dynamic_units * 2, activation="linear"
        )
        self.zd_given_x_h_posterior = tfkl.Dense(
            latent_dynamic_units * 2, activation="linear"
        )
        self.z_encoder = tfkl.Dense(hidden_units, activation=activation)
        self.x_given_zg_zd_h = tfkl.Dense(output_units * 2)
        self.y_prior = OneHotCategorical(logits=[1.0] * clusters)

    def relaxed_softmax(self, logits):
        """Reparametrized softmax sample from Bibbs-Boltzman distribution

        Args:
            logits (tf.Tensor): (B, C) where C is the cluster dmiension

        Returns:
            tfp.Distribution:  RelaxedOneHotCategorical
        """
        return RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)

    @staticmethod
    def categorical_onehot(logits):
        """Argmax of softmax sample, not reparametrized

        Args:
            logits (tf.Tensor): (B, C) where C is the cluster dimension

        Returns:
            tfp.Distribution: OneHotCategorical
        """
        return OneHotCategorical(logits=logits)

    @staticmethod
    def mvn(params):
        """Reparametrixed Multivariate Normal Distribution

        Args:
            params (tf.Tensor): (B, 2*D) mean and variance params
        Returns:
            tfp.Distribution: MultivariateNormalDiag
        """
        mean, variance = tf.split(params, 2, -1)
        variance = scale(variance)
        return MultivariateNormalDiag(mean, variance)

    @staticmethod
    def kl(p, q, q_sample=None):
        """
        Computes KL[Q||P]
        """
        if q_sample is None:
            return kl_divergence(q, p)
        p_ll = p.log_prob(q_sample)
        q_ll = q.log_prob(q_sample)
        return q_ll - p_ll

    #@tf.function
    def call(self, inputs, training=False):
        """Call pass to infer and generate

        Args:
            inputs (tf.Tensor): (B, T, D) B=batch, T=time length, D=input dimension
            training (bool, optional): Is Trainig. Defaults to False.
        """
        inputs_shape = tf.shape(inputs)
        inputs_shape_ = inputs.shape
        B = inputs_shape[0]
        T = inputs_shape[1]
        D = inputs_shape[2]

        B_ = inputs_shape_[0]
        T_ = inputs_shape_[1]
        D_ = inputs_shape_[2]
        # Infer cluster Assignment via the Bi-Directional RNN
        bi_h = self._birnn(inputs)
        # The inferred cluster assignment is also used as conditioning for generation
        if training:
            # If training then take a reparametrized relaxed sample
            y = self.relaxed_softmax(self.y_given_h(bi_h))
            y_sample = tf.squeeze(y.sample(1), 0)
        else:
            # if not training then take hard-max sample
            y = self.categorical_onehot(self.y_given_h(bi_h))
            y_sample = tf.cast(tf.squeeze(y.sample(1), 0), inputs.dtype)

        # Infer global latent variable zg prior/posterior
        zg_prior = self.mvn(self.zg_given_y_prior(y_sample))
        zg_prior_sample = tf.squeeze(zg_prior.sample(1))
        zg_posterior = self.mvn(
            self.zg_given_h_y_posterior(tf.concat([bi_h, y_sample], -1))
        )
        zg_posterior_sample = tf.squeeze(zg_posterior.sample(1), 0)

        # initialize hidden state of generating recurrent nn
        h_state = tf.zeros((B, self.state_size))
        kl_zd = tf.zeros((B,))
        x_ll = tf.zeros((B,))
        x_samples = tf.TensorArray(
            dtype=inputs.dtype,
            size=T_,
            element_shape=tf.TensorShape((B_, self.output_size)),
        )
        '''zd_samples = tf.TensorArray(
            dtype=inputs.dtype,
            size=T_,
            element_shape=tf.TensorShape((B_, self.latend_dynamic_units)),
        )'''
        # Generation
        for t in range(T_):
            x = inputs[:, t]
            # hidden recurrence
            h_state, _ = self._rnn_cell(x, states=h_state, training=training)
            # prior and posterior inference of dynamic latent variable
            zd_prior = self.mvn(self.zd_given_h_prior(h_state))
            zd_prior_sample = tf.squeeze(zd_prior.sample(1))

            zd_posterior = self.mvn(
                self.zd_given_x_h_posterior(tf.concat([x, h_state], -1))
            )
            zd_posterior_sample = tf.squeeze(zd_prior.sample(1))
            #zd_samples = zd_samples.write(t, zd_posterior_sample)
            zd_posterior_sample_encoded = self.z_encoder(zd_posterior_sample)

            if self.use_kl_sample:
                kl_zd += self.kl(zd_prior, zd_posterior, zd_posterior_sample)
            else:
                kl_zd += self.kl(zd_prior, zd_posterior)

            # inference of output distribution
            x_recon = self.mvn(
                self.x_given_zg_zd_h(
                    tf.concat(
                        [zd_posterior_sample_encoded, zg_posterior_sample, h_state], -1
                    )
                )
            )
            x_recon_sample = tf.squeeze(x_recon.sample(1), 0)
            x_samples = x_samples.write(t, x_recon_sample)
            x_ll += x_recon.log_prob(x)

        if self.use_kl_sample:
            kl_zg = self.kl(zg_prior, zg_posterior, zg_posterior_sample)
        else:
            kl_zg = self.kl(zg_prior, zg_posterior)

        y_entr = y.entropy()
        elbo = x_ll - self.kl_weight*(kl_zd + kl_zg) + y_entr

        return {
            "x_samples": x_samples.stack(),
            #"zd_samples": zd_samples.stack(),
            "y_sample": y_sample,
            "zg_posterior_sample": zg_posterior_sample,
            "elbo": elbo,
        }

    #@tf.function
    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            out = self(x)
            loss = -tf.reduce_mean(out["elbo"])
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {"elbo": -loss}


@tf.keras.saving.register_keras_serializable(package="Variational")
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
        temperature=1,
        use_sample_kl=True,
        **kwargs,
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
        self.entr_weight = 0.5
        self.use_sample_kl = use_sample_kl

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
        Sample a random vector from a gaussian distribution. Tensorflow probability already reparametrizes
        the samples to make the parameters differentiable :)
        Inputs:
            x: Tensor of shape of shape 2*d partitioned in [mean, variance] params
        """
        dim = tf.shape(x)[-1] // 2

        distr = MultivariateNormalDiag(
            loc=x[..., :dim],
            scale_diag=self.scale(x[..., dim:]),
        )
        return tf.squeeze(distr.sample(1), axis=0)

    def kl(self, Q, P, Q_sample=None):
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
        if Q_sample is None:
            return kl_divergence(Q, P)
        Pll = P.log_prob(Q_sample)
        Qll = Q.log_prob(Q_sample)
        return Qll - Pll

    @staticmethod
    def categorical_entropy(logits):
        """
        Computes the cluster kl divergence KL(Q(y|x)||P(y)), since P(y) is uniform this results into computing the entropy of Q(y|x)
        """
        cat = OneHotCategorical(logits=logits)
        return cat.entropy()

    def call(self, inputs, training=False, *args, **kwargs):
        generative = self.generative(inputs, training=training)
        h_states = generative["h_states"]
        inference = self.inference(
            inputs, h_states, training=training, temperature=self.temperature
        )
        return generative, inference

    def train_step(self, data):
        # print("train step")
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
            if self.use_sample_kl:
                kl_Q_zd_x_h__P_zd_h = self.kl(Q_zd_x_h_param, P_zd_param, Q_zd_sample)
            else:
                kl_Q_zd_x_h__P_zd_h = self.kl(Q_zd_x_h_param, P_zd_param)
            zd_loss = tf.reduce_mean(tf.reduce_sum(kl_Q_zd_x_h__P_zd_h, -1))

            # KL(zg_posterior||zg_prior)
            if self.use_sample_kl:
                kl_Q_zg_x_y__P_zg_y = self.kl(Q_zg_param, P_zg_param, Q_zg_sample)
            else:
                kl_Q_zg_x_y__P_zg_y = self.kl(Q_zg_param, P_zg_param)
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
        # print("Sampling")
        if prev_inputs is None:
            y_sample = self.generative.P_y.sample(n_samples)  # samples x clusters
            zg_y_param = self.generative.P_zg_y(y_sample)  # samples x d
            zg_y_sample = self.gaussian_sample(zg_y_param)

            h_state = tf.zeros(
                tf.TensorShape((n_samples, *self.generative.P_zd_h.state_size))
            )
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
                "zd_sample": tf.concat(
                    [
                        gen["zd_sample"],
                        tf.transpose(ta_zd_sample.stack(), perm=(1, 0, 2)),
                    ],
                    1,
                ),
                "x_sample": tf.concat(
                    [
                        gen["x_sample"],
                        tf.transpose(ta_x_sample.stack(), perm=(1, 0, 2)),
                    ],
                    1,
                ),
                "zg_y_x_param": zg_y_param,
                "zd_param": tf.concat(
                    [
                        gen["zd_param"],
                        tf.transpose(ta_zd_param.stack(), perm=(1, 0, 2)),
                    ],
                    1,
                ),
                "x_param": tf.concat(
                    [gen["x_param"], tf.transpose(ta_x_param.stack(), perm=(1, 0, 2))],
                    1,
                ),
                "h_states": tf.concat(
                    [
                        gen["h_states"],
                        tf.transpose(ta_h_states.stack(), perm=(1, 0, 2)),
                    ],
                    1,
                ),
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
                "h_states": tf.transpose(ta_h_states.stack(), perm=(1, 0, 2)),
            }


@tf.keras.saving.register_keras_serializable(package="Variational")
class VariationalRNN(models.Model):
    def __init__(
        self,
        output_units,
        recurent_units,
        dropout=0.0,
        activation="tanh",
        recurrent_activation="tanh",
        rnn_type="gru",
        **kwargs,
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
        Sample a random vector from a gaussian distribution. Tensorflow probability already reparametrizes
        the samples to make the parameters differentiable :)
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

    @tf.function
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
