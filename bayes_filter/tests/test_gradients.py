from .common_setup import *
import tensorflow as tf
import numpy as np


def test_softplus(tf_session):
    with tf_session.graph.as_default():
        x = tf.random.normal([5])
        softplus = tf.nn.softplus(x)
        sigmoid = tf.nn.sigmoid(x)

        assert np.all(
            np.isclose(*tf_session.run([tf.gradients(softplus, x)[0], sigmoid])))

def test_diag_mult():
    a = np.random.normal(size=[4,4])
    b = np.random.normal(size=[4])

    assert np.all(a.dot(np.diag(b)) == a*b[None, :])

    assert np.all(np.diag(b).dot(a) == b[:, None] * a)

