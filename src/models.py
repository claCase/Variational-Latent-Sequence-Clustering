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

tfkl = tf.keras.layers


@tf.keras.saving.register_keras_serializable(package="Variational")
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

    Loss function:
    L = Σ log(p(x|z_g, z_d, h_d)) - KL(q(z_d|x, h)||p(z_d|h_d)) - KL(q(z_g|x,y)||P(z_g|y)) + H(q(y|x))

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
        stateful=False,
        dropout=False,
        recurrent_dropout=False,
        temperature=1.0,
        use_sample_kl=True,
        free_runnning_prob=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.latend_dynamic_units = latent_dynamic_units
        self.latend_global_units = latent_global_units
        self.use_kl_sample = use_sample_kl
        self.clusters = clusters
        self.free_running_prob = tf.cast(
            tf.maximum(1.0, tf.minimum(free_runnning_prob, 0.0)), tf.float32
        )
        self.temperature = temperature
        self.kl_weight = 1.0
        _cell_args = dict(
            units=hidden_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout)
        if rnn_type == "gru":
            _cell = tfkl.GRUCell
        elif rnn_type == "lstm":
            _cell = tfkl.LSTMCell
        elif rnn_type == "rnn":
            _cell = tfkl.SimpleRNNCell
        elif rnn_type == "indrnn" :
            _cell = layers.IndipendentRNNCell
            _cell_args.pop("recurrent_activation")
        else:
            raise NotImplementedError(
                f"rnn_type {rnn_type} invalid. Choose between rnn, lstm, gru"
            )
        self._rnn_cell = _cell(**_cell_args)
        # Bi-RNN used to infer posterior over sequential time series
        self._birnn = models.Sequential(
            [
                tfkl.Bidirectional(
                    tfkl.RNN(
                        _cell(**_cell_args),
                        return_state=False,
                        return_sequences=False,
                        stateful=stateful,
                    )
                ),
                tfkl.Dense(clusters, output_activation),
            ]
        )
        self.stateful = stateful
        self.state_size = hidden_units
        self.output_size = output_units
        self.states = None
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

    def reset_states(self):
        self.states = None
            
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

    @tf.function
    def call(self, inputs, training=False):
        """Call pass to infer and generate

        Args:
            inputs (tf.Tensor): (B, T, D) B=batch, T=time length, D=input dimension
            training (bool, optional): Is Trainig. Defaults to False.
        """
        
        inputs_shape = tf.shape(inputs)
        B = inputs_shape[0]
        T = inputs_shape[1]
        D = inputs_shape[2]
    
        # Infer cluster Assignment via the Bi-Directional RNN
        bi_h = self._birnn(inputs)
        y_logits = self.y_given_h(bi_h)
        # The inferred cluster assignment is also used as conditioning for generation
        if training:
            # If training then take a reparametrized relaxed sample
            y = self.relaxed_softmax(y_logits)
            y_sample = tf.squeeze(y.sample(1), 0)
        else:
            # if not training then take hard-max sample
            y = self.categorical_onehot(y_logits)
            y_sample = tf.cast(tf.squeeze(y.sample(1), 0), inputs.dtype)

        # Infer global latent variable zg prior/posterior
        zg_prior = self.mvn(self.zg_given_y_prior(y_sample))
        zg_prior_sample = tf.squeeze(zg_prior.sample(1))
        zg_posterior = self.mvn(
            self.zg_given_h_y_posterior(tf.concat([bi_h, y_sample], -1))
        )
        zg_posterior_sample = tf.squeeze(zg_posterior.sample(1), 0)

        # initialize hidden state of generating recurrent nn
        if self.stateful:
            if self.states is not None:
                h_state = self.states
            else:
                h_state = tf.zeros((B, self.state_size))
        else:
            h_state = tf.zeros((B, self.state_size))
        kl_zd = tf.zeros((B,))
        x_ll = tf.zeros((B,))
        x_samples = tf.TensorArray(
            dtype=tf.float32,
            size=T,
        )
        zd_samples = tf.TensorArray(
            dtype=tf.float32,
            size=T,
        )
        h_states = tf.TensorArray(
            dtype=tf.float32,
            size=T,
        )
        x_recon_sample = tf.zeros(shape=(B, D))
        # Generation
        for t in tf.range(T - 1):
            teacher_force = tf.math.greater(
                tf.random.uniform((1,), 0.0, 1.0), self.free_running_prob
            )
            if t == 0 or teacher_force:
                x = inputs[:, t]
            else:
                x = x_recon_sample
            # hidden recurrence
            h_state, _ = self._rnn_cell(x, states=h_state, training=training)
            h_states = h_states.write(t, h_state)
            # prior and posterior inference of dynamic latent variable
            zd_prior = self.mvn(self.zd_given_h_prior(h_state))
            zd_prior_sample = tf.squeeze(zd_prior.sample(1), 0)
            zd_posterior = self.mvn(
                self.zd_given_x_h_posterior(tf.concat([x, h_state], -1))
            )
            zd_posterior_sample = tf.squeeze(zd_posterior.sample(1), 0)
            zd_samples = zd_samples.write(t, zd_posterior_sample)
            zd_posterior_sample_encoded = self.z_encoder(zd_posterior_sample)

            # Dynamic latent kl divergence
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
            x_ll += x_recon.log_prob(inputs[:, t + 1])

        # Static latent kl divergence
        if self.use_kl_sample:
            kl_zg = self.kl(zg_prior, zg_posterior, zg_posterior_sample)
        else:
            kl_zg = self.kl(zg_prior, zg_posterior)

        # Entropy of categorical distribution
        y_entr = self.categorical_onehot(y_logits).entropy()
        elbo = x_ll - self.kl_weight * (kl_zd + kl_zg) + y_entr

        if self.stateful:
            self.states = h_state
        return {
            "x_samples": tf.transpose(x_samples.stack(), (1, 0, 2)),
            "y_sample": y_sample,
            "zg_sample": zg_posterior_sample,
            "zd_sample": tf.transpose(zd_samples.stack(), (1, 0, 2)),
            "h_states": tf.transpose(h_states.stack(), (1, 0, 2)),
            "elbo": elbo,
        }

    @tf.function
    def sample(
        self, inputs=None, samples=10, length=50, from_prior=True, use_mean=True
    ):
        inputs_shape = tf.shape(inputs)
        B = inputs_shape[0]
        T = inputs_shape[1]
        D = inputs_shape[2]

        if inputs is not None:
            out = self(inputs, training=False)
            y_sample = out["y_sample"]
            zg_sample = out["zg_sample"]
            h_state = out["h_states"][:, -1]
            x = inputs[:, -2]
        else:
            y_sample = tf.cast(self.y_prior.sample(samples), tf.float32)
            x = tf.random.normal(shape=(samples, 1, self.output_size))
            bi_h = self._birnn(x)
            x = x[:, 0]
            if from_prior:
                zg = self.mvn(self.zg_given_y_prior(y_sample))
            else:
                zg = self.mvn(
                    self.zg_given_h_y_posterior(tf.concat([bi_h, y_sample], -1))
                )
            if use_mean:
                zg_sample = zg.mean()
            else:
                zg_sample = tf.squeeze(zg.sample(1), 0)
            h_state = tf.zeros((B, self.state_size))

        dtype = inputs.dtype if inputs is not None else tf.float32
        x_samples = tf.TensorArray(
            dtype=dtype,
            size=length,
        )
        zd_samples = tf.TensorArray(
            dtype=dtype,
            size=length,
        )
        h_states = tf.TensorArray(
            dtype=dtype,
            size=length,
        )
        # Generation
        for t in range(length):
            # hidden recurrence
            h_state, _ = self._rnn_cell(x, states=h_state, training=False)
            h_states = h_states.write(t, h_state)
            # prior and posterior inference of dynamic latent variable
            if from_prior:
                zd_prior = self.mvn(self.zd_given_h_prior(h_state))
                if use_mean:
                    zd_sample = zd_prior.mean()
                else:
                    zd_sample = tf.squeeze(zd_prior.sample(1), 0)
            else:
                zd_posterior = self.mvn(
                    self.zd_given_x_h_posterior(tf.concat([x, h_state], -1))
                )
                if use_mean:
                    zd_sample = zd_posterior.mean()
                else:
                    zd_sample = tf.squeeze(zd_posterior.sample(1), 0)
            zd_samples = zd_samples.write(t, zd_sample)
            zd_sample_encoded = self.z_encoder(zd_sample)
            # inference of output distribution
            x_recon = self.mvn(
                self.x_given_zg_zd_h(
                    tf.concat([zd_sample_encoded, zg_sample, h_state], -1)
                )
            )
            x_recon_sample = tf.squeeze(x_recon.sample(1), 0)
            x = x_recon_sample
            x_samples = x_samples.write(t, x_recon_sample)

        x_samples = tf.transpose(x_samples.stack(), (1, 0, 2))
        zd_samples = tf.transpose(zd_samples.stack(), (1, 0, 2))
        h_states = tf.transpose(h_states.stack(), (1, 0, 2))
        return {
            "x_samples": (
                x_samples
                if inputs is None
                else tf.concat([out["x_samples"], x_samples], 1)
            ),
            "y_sample": y_sample,
            "zg_sample": zg_sample,
            "zd_sample": (
                zd_samples
                if inputs is None
                else tf.concat([out["zd_sample"], zd_samples], 1)
            ),
            "h_states": (
                h_states
                if inputs is None
                else tf.concat([out["h_states"], h_states], 1)
            ),
        }

    def train_step(self, data):
        #@tf.function
        def step():
            with tf.GradientTape() as tape:
                out = self(data, training=True)
                loss = -tf.reduce_mean(out["elbo"])
            return tape.gradient(loss, self.trainable_variables), loss

        grad, loss = step()
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {"elbo": -loss}


class RecurrentNN(models.Model):
    def __init__(
        self,
        units,
        hidden_units,
        rnn_type="gru",
        out_activation="linear",
        activation="silu",
        recurrent_activation="silu",
        recurrent_dropout=0,
        dropout=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        _cell_args = dict(
            units=hidden_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout)
        if rnn_type == "gru":
            _cell = tfkl.GRUCell
        elif rnn_type == "lstm":
            _cell = tfkl.LSTMCell
        elif rnn_type == "rnn":
            _cell = tfkl.SimpleRNNCell
        elif rnn_type == "indrnn" :
            _cell = layers.IndipendentRNN
            _cell_args.pop("recurrent_activation")
        else:
            raise NotImplementedError(
                f"rnn_type {rnn_type} invalid. Choose between rnn, lstm, gru"
            )
        self.state_size = hidden_units
        self.output_size = units
        self._out = tfkl.Dense(units, activation)
        self._cell = _cell(
                **_cell_args
            )
        self._rnn = tfkl.RNN(
            self._cell,
            return_state=True,
            return_sequences=True,
        )
        
    @tf.function
    def call(self, inputs, training=False):
        h_states, h_state = self._rnn(inputs)
        x = self._out(h_states)
        return {"output": x, "states": h_states}
    
    def train_step(self, data):
        #@tf.function
        def step():
            with tf.GradientTape() as tape:
                out = self(data[:, :-1], training=True)
                loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(data[:, 1:], out["output"]))
            return tape.gradient(loss, self.trainable_variables), loss

        grad, loss = step()
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {"loss": loss}
    
    @tf.function
    def sample(self, inputs, prev_state=None, length=100):
        inputs_shape = tf.shape(inputs)
        B = inputs_shape[0]
        T = inputs_shape[1]
        D = inputs_shape[2]
        if prev_state is None:
            h_state = tf.zeros((B, self.state_size))
        else:
            h_state = prev_state
        
        x_samples = tf.TensorArray(
            dtype=inputs.dtype,
            size=length,
        )
        h_states = tf.TensorArray(
            dtype=inputs.dtype,
            size=length,
        )

        # initialize recurrence
        h_states_prev, h_state = self._rnn(inputs, initial_state=h_state, training=False)
        x = self._out(h_states_prev)
        x_prev = x[:, -2]
        # free running sampling
        for t in range(length):
            _, h_state = self._cell(x_prev, h_state)
            x_prev = self._out(h_state)
            x_samples = x_samples.write(t, x_prev)
            h_states = h_states.write(t, h_state)
        return {"output":tf.concat([x, tf.transpose(x_samples.stack(), (1, 0, 2))], 1), 
                "states":tf.concat([h_states_prev, tf.transpose(h_states.stack(), (1, 0, 2))], 1)}


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
