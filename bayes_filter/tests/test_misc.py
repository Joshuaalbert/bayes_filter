import os

from .common_setup import *
import numpy as np
import tensorflow as tf

from bayes_filter import jitter
from bayes_filter.misc import random_sample, flatten_batch_dims, load_array_file, timer, diagonal_jitter, \
    log_normal_solve_fwhm


def test_random_sample(tf_session):
    with tf_session.graph.as_default():
        t = tf.ones((6,5,8))
        assert tf_session.run(random_sample(t)).shape == (6,5,8)
        assert tf_session.run(random_sample(t,3)).shape == (3, 5, 8)
        assert tf_session.run(random_sample(t, 9)).shape == (6, 5, 8)


def test_flatten_batch_dims(tf_session):
    with tf_session.graph.as_default():
        t = tf.ones((1,2,3,4))
        f = flatten_batch_dims(t)
        assert tuple(tf_session.run(tf.shape(f))) == (6,4)

        t = tf.ones((1, 2, 3, 4))
        f = flatten_batch_dims(t,num_batch_dims=2)
        assert tuple(tf_session.run(tf.shape(f))) == (2, 3 , 4)


def test_load_array_file(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    lofar_cycle0_array = os.path.join(arrays, 'arrays/lofar.cycle0.hba.antenna.cfg')
    gmrt_array = os.path.join(arrays, 'arrays/gmrtPos.csv')
    lofar_array = load_array_file(lofar_array)
    lofar_cycle0_array = load_array_file(lofar_cycle0_array)
    gmrt_array = load_array_file(gmrt_array)
    assert (len(lofar_array[0]),3) == lofar_array[1].shape
    assert (len(lofar_cycle0_array[0]), 3) == lofar_cycle0_array[1].shape
    assert (len(gmrt_array[0]), 3) == gmrt_array[1].shape


def test_timer(tf_session):
    with tf_session.graph.as_default():
        t0 = timer()
        with tf.control_dependencies([t0]):
            t1 = timer()
        t0, t1 = tf_session.run([t0,t1])
        assert t1 > t0


def test_diagonal_jitter(tf_session):
    with tf_session.graph.as_default():
        j = diagonal_jitter(5)
        assert np.all(tf_session.run(j) == jitter*np.eye(5))


def test_log_normal_solve_fwhm():
    mu, stddev = log_normal_solve_fwhm(np.exp(1), np.exp(2), np.exp(-1))
    assert np.isclose(stddev**2, 0.5 * (0.5) ** 2)
    assert np.isclose(mu, 3./2. + 0.5 * (0.5) ** 2)