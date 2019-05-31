from .common_setup import *
from ..vi import VariationalBayes, conditional_different_points, WhitenedVariationalPosterior
from ..misc import safe_cholesky
import tensorflow_probability as tfp
import tensorflow as tf
from .. import float_type

def test_variational_bayes(tf_session):
    with tf_session.graph.as_default():
        N=10
        Nf=2
        Yreal = tf.random.normal(shape=[N, Nf], dtype=float_type)
        Yimag = tf.random.normal(shape=[N, Nf], dtype=float_type)
        freqs = tf.random.normal(shape=[Nf], dtype=float_type)
        X = tf.random.normal(shape=[N,13], dtype=float_type)

        VI = VariationalBayes(Yreal, Yimag, freqs, X)

        solve_op = VI.solve_variational_posterior(iters=1000,learning_rate=1e-5)
        print(tf_session.run(solve_op))


def test_conditional(tf_session):
    with tf_session.graph.as_default():
        S = 1000
        N = 10
        M = 11
        X = tf.cast(tf.linspace(0., 10., N), float_type)[:,None]
        Xstar = tf.cast(tf.linspace(0., 10., M), float_type)[:, None]
        kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic()

        K = kern.matrix(X, X)
        L = safe_cholesky(K)

        K_x_xstar = kern.matrix(X, Xstar)
        K_xstar_xstar = kern.matrix(Xstar, Xstar)

        q_mean = tf.random.normal((N,), dtype=float_type)
        q_scale = tf.random.normal((N,), dtype=float_type)
        q_sqrt = tf.nn.softplus(q_scale)


        post = WhitenedVariationalPosterior(N)
        white_samples = post._build_distribution(q_mean, q_scale)


        dist = conditional_different_points(q_mean, q_sqrt, L, K_xstar_xstar, K_x_xstar)
        samples = dist.sample(S)


        print(tf_session.run(samples))