from .common_setup import *
from ..vi import VariationalBayes
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
