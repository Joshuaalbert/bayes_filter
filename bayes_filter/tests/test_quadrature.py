from .common_setup import *

import numpy as np
import tensorflow as tf

from bayes_filter import float_type
from bayes_filter.quadrature import dblquad


def test_dblquad(tf_session):
    with tf_session.graph.as_default():
        def func(t1,t2):
            return tf.tile(tf.reshape((-tf.square(t1-t2)), (1,)), [50])
        l = tf.constant(0., float_type)
        u = tf.constant(1., float_type)
        I, info = dblquad(func, l, u,lambda t:l, lambda t:u, (50,),ode_type='adaptive')
        print(tf_session.run([I,info]))

        assert np.all(np.isclose(tf_session.run(I), -1/6.))