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
        dropout_rate=0.1,
        activation="tanh",
        recurrent_activation="tanh",
        clusters=5,
        rnn_type="gru",
        **kwargs
    ):
        super(VariationalMixtureRNN, self).__init__(**kwargs)
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.drouput_rate = dropout_rate
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.clusters = clusters

        self.generative = layers.GenerativeVariationalMixture(
            hidden_units=hidden_units,
            output_units=output_units,
            dropout=dropout_rate,
            clusters=clusters,
            activation=activation,
            recurrent_activation=recurrent_activation,
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

    def call(self, inputs, training, *args, **kwargs):
        (
            y_sample,
            zg_y_sample,
            zd_sample,
            ta_x_sample,
            zg_y_param,
            zd_param,
            x_param,
            h_states,
        ) = self.generative(inputs, training=training)
        (
            y_x_sample,
            zg_sample,
            zd_x_h_sample,
            y_x_param,
            zg_param,
            zd_x_h_param,
        ) = self.inference(inputs, h_states, training=training)
        return (
            y_sample,
            zg_y_sample,
            zd_sample,
            ta_x_sample,
            zg_y_param,
            zd_param,
            x_param,
            h_states,
        ), (y_x_sample, zg_sample, zd_x_h_sample, y_x_param, zg_param, zd_x_h_param)


    def train_step(self, data):
        print(tf.shape(data), data.get_shape())
        x = data
        with tf.GradientTape() as tape:
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
            ) = self.call(
                x, training=True
            )

            # E[p(x|z_g, z_d, h_d)] = log-likelihood
            nll = -tf.reduce_mean(
                MultivariateNormalDiag(
                    P_x_param[..., : self.output_units],
                    self.scale(P_x_param[..., self.output_units :]),
                ).log_prob(x)
            )

            # KL(zd_posterior||zd_prior)
            kl_Q_zd_x_h__P_zd_h = self.kl_z(Q_zd_x_h_param, P_zd_param)
            zd_loss = -tf.reduce_mean(kl_Q_zd_x_h__P_zd_h)

            # KL(zg_posterior||zg_prior)
            kl_Q_zg_x_y__P_zg_y = self.kl_z(Q_zg_param, P_zg_param)
            zg_loss = -tf.reduce_mean(kl_Q_zg_x_y__P_zg_y)

            # Entropy of infrence cluster assignment
            entr_y = tf.reduce_mean(self.categorical_entropy(Q_y_x_param))

            tot_loss = nll + zd_loss + zg_loss + entr_y
        grad = tape.gradient(tot_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {"nll": nll, "zd_loss": zd_loss, "zg_loss": zg_loss, "entr_y": entr_y}


if __name__ == "__main__":
    '''i = np.random.normal(size=(100, 50, 15))
    vrnn = VariationalMixtureRNN(15, 10)
    o = vrnn(i)'''
