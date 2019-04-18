
from .common_setup import *

import numpy as np
import tensorflow as tf

from bayes_filter import float_type
from bayes_filter.parameters import ConstrainedBijector, ScaledLowerBoundedBijector, SphericalToCartesianBijector, \
    Parameter, ScaledPositiveBijector, ScaledBijector


def test_constrained_bijector(tf_session):
    with tf_session.graph.as_default():
        x = tf.constant(-10., dtype=float_type)
        b = ConstrainedBijector(-1., 2.)
        y = b.forward(x)
        assert -1. < tf_session.run(y) < 2.
        x = tf.constant(10., dtype=float_type)
        b = ConstrainedBijector(-1., 2.)
        y = b.forward(x)
        assert -1. < tf_session.run(y) < 2.


def test_positive_lowerbound_bijector(tf_session):
    with tf_session.graph.as_default():
        x = tf.constant(-10., dtype=float_type)
        b = ScaledLowerBoundedBijector(2., 5.)
        y = b.forward(x)
        assert 2. < tf_session.run(y)


def test_spherical_to_cartesian_bijector(tf_session):
    with tf_session.graph.as_default():
        sph = tf.constant([[1.,np.pi/2.,np.pi/2.]],dtype=float_type)
        car = tf.constant([[0., 1., 0.]],dtype=float_type)
        b = SphericalToCartesianBijector()
        assert np.linalg.norm(tf_session.run(b.forward(sph))) == 1.
        assert np.all(np.isclose(tf_session.run(b.forward(sph)), tf_session.run(car)))
        assert np.all(np.isclose(tf_session.run(b.inverse(tf_session.run(car))), tf_session.run(sph)))


def test_parameter(tf_session):
    with tf_session.graph.as_default():
        p = Parameter(constrained_value=10.)
        assert tf_session.run(p.constrained_value) == tf_session.run(p.unconstrained_value)


def test_scaled_positive(tf_session):
    with tf_session.graph.as_default():
        b = ScaledPositiveBijector(10.)
        assert tf_session.run(b.inverse(tf.constant(100.,float_type))) == np.log(100./10.)


def test_scaled(tf_session):
    with tf_session.graph.as_default():
        b = ScaledBijector(10.)
        assert tf_session.run(b.inverse(tf.constant(100.,float_type))) == 100./10.