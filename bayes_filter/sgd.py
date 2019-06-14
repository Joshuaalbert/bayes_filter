import tensorflow as tf
from .misc import random_sample
from . import float_type
import numpy as np
import tensorflow_probability as tfp
from .callbacks import SummarySendCallback
from collections import namedtuple


def stochastic_gradient_descent(log_prob, initial_state, iters, learning_rate=0.1, parallel_iterations=10):
    """

    :param self:
    :param log_prob:
    :param initial_state:
    :param iters:
    :param stepsizes:
    :param learning_rate:
    :param parallel_iterations:
    :return:
    """

    list_like = isinstance(initial_state, (tuple, list))
    if not list_like:
        initial_state = [initial_state]

    def _body(i, params, loss):
        logp = tf.reduce_mean(log_prob(*params))
        with tf.control_dependencies([tf.print('log_prob', logp)]):
            grad = tf.gradients(logp, params)
            params = [random_sample(p) for p in params]
            params = [p + learning_rate*(g + 0.01*tf.math.abs(g)*tf.random.normal(shape=tf.shape(p), dtype=p.dtype)) for (p, g) in zip(params, grad)]
            return i+1, params, loss.write(i, -logp)

    loss = tf.TensorArray(dtype=float_type, size=iters, infer_shape=False,element_shape=())

    _, params, loss = tf.while_loop(lambda i, *args: i < iters,
                                    _body,
                                    (tf.constant(0, dtype=tf.int32),
                                     initial_state,
                                     loss),
                                    parallel_iterations=parallel_iterations,
                                    back_prop=False)

    return params, loss.stack()


def adam_stochastic_gradient_descent_with_linesearch(
        loss_fn,
        adam_params,
        iters=100,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        stop_patience=3,
        patient_percentage=1e-3,
        parallel_iterations=10,
        search_size=5):
    """

    :param loss_fn:
    :param adam_params:
    :param iters:
    :param learning_rate:
    :param beta1:
    :param beta2:
    :param epsilon:
    :param stop_patience:
    :param parallel_iterations:
    :param search_size:
    :return:
    """

    learning_rate = tf.convert_to_tensor(learning_rate, float_type, 'learning_rate')
    stop_patience = tf.convert_to_tensor(stop_patience, tf.int32, 'stop_patience')
    iters = tf.convert_to_tensor(iters, tf.int32, 'iters')
    beta1 = tf.convert_to_tensor(beta1, float_type, 'beta1')
    beta2 = tf.convert_to_tensor(beta2, float_type, 'beta2')
    epsilon = tf.convert_to_tensor(epsilon, float_type, 'epsilon')

    if not isinstance(adam_params, (tuple, list)):
        raise ValueError('adam params should be list like')

    adam_params = list(adam_params)

    m0 = [tf.zeros_like(v) for v in adam_params]
    v0 = [tf.zeros_like(v) for v in adam_params]

    def _body(t, adam_params, m, v, loss_ta, min_loss, patience):

        loss = tf.reduce_mean(loss_fn(*adam_params))
        loss_better = tf.less_equal(loss, (1. - tf.convert_to_tensor(patient_percentage, float_type)) * min_loss)
        min_loss = tf.minimum(min_loss, loss)
        patience = tf.cond(loss_better, lambda: tf.constant(0, patience.dtype),
                           lambda: patience + tf.constant(1, patience.dtype))

        loss_ta = loss_ta.write(t, loss)

        adam_grads = tf.gradients(loss, adam_params)
        pert_grads = []
        for g_t in adam_grads:
            if g_t is None:
                pert_grads.append(g_t)
                continue
            pert_grads.append(
                g_t + tf.constant(0.01, float_type) * tf.math.abs(g_t) * tf.random.normal(shape=tf.shape(g_t), dtype=g_t.dtype))

        next_adam_params, next_m, next_v = _adam_update(adam_grads, adam_params, m, t, v, loss)
        [n.set_shape(p.shape) for n, p in zip(next_adam_params, adam_params)]

        return t + 1, next_adam_params, next_m, next_v, loss_ta, min_loss, patience

    def _adam_update(adam_grads, adam_params, m, t, v, loss0):
        t_float = tf.cast(t, float_type) + 1.
        lr_t = tf.math.sqrt(1. - tf.math.pow(beta2, t_float)) * \
               tf.math.reciprocal(tf.math.sqrt(1. - tf.math.pow(beta1, t_float)))
        next_m, next_v = [], []
        for (m_t, v_t, g_t) in zip(m, v, adam_grads):
            if g_t is None:
                next_m.append(m_t)
                next_v.append(v_t)
                continue

            m_t = beta1 * m_t + (1. - beta1) * g_t
            v_t = beta2 * v_t + (1. - beta2) * tf.math.square(g_t)
            next_m.append(m_t)
            next_v.append(v_t)

            # p_t = p_t - lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon)

        def search_function(a):
            # TODO: don't need to redo predictive x because nat_params fixed
            test_adam_params = []
            for (m_t, v_t, p_t, g_t) in zip(next_m, next_v, adam_params, adam_grads):
                if g_t is None:
                    test_adam_params.append(p_t)
                    continue
                test_adam_params.append(p_t - a * lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon))
            loss = tf.reduce_mean(loss_fn(*test_adam_params))
            return loss - loss0

        search_space = tf.math.exp(
            tf.cast(tf.linspace(tf.math.log(learning_rate) - 7., tf.math.log(learning_rate), search_size), float_type))
        search_results = tf.map_fn(search_function, search_space, parallel_iterations=search_size)
        argmin = tf.argmin(search_results)

        a = search_space[argmin]
        loss_min = search_results[argmin]

        with tf.control_dependencies([tf.print('Step:', t, 'Optimal', 'Learning rate:', a, 'loss reduction', loss_min,
                                               'from loss:', loss0)]):
            next_adam_params = []
            for (m_t, v_t, p_t, g_t) in zip(next_m, next_v, adam_params, adam_grads):
                if g_t is None:
                    next_adam_params.append(p_t)
                    continue
                next_adam_params.append(p_t - a * lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon))

        return next_adam_params, next_m, next_v

    def _cond(t, adam_params, m, v, loss_ta, min_loss, patience):
        return tf.logical_and(tf.less(patience, stop_patience), tf.less(t, iters))

    loss_ta = tf.TensorArray(dtype=float_type, size=iters, infer_shape=False, element_shape=())

    _, adam_params, m, v, loss_ta, _, _ = tf.while_loop(_cond,
                                                        _body,
                                                        (tf.constant(0, dtype=tf.int32),
                                                         adam_params,
                                                         m0,
                                                         v0,
                                                         loss_ta,
                                                         tf.constant(np.inf, float_type),
                                                         tf.constant(0, tf.int32)),
                                                        parallel_iterations=parallel_iterations,
                                                        back_prop=False,
                                                        return_same_structure=True)

    return adam_params, loss_ta.stack()


###
# forward_gradients_v2: Taken from https://github.com/renmengye/tensorflow-forward-ad
###

def forward_gradients_v2(ys, xs, grad_xs=None, gate_gradients=False):
    """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
    With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
    the vector being pushed forward."""
    if type(ys) == list:
        v = [tf.ones_like(yy) for yy in ys]
    else:
        v = tf.ones_like(ys)  # dummy variable
    g = tf.gradients(ys, xs, grad_ys=v)
    return tf.gradients(g, v, grad_ys=grad_xs)


def natural_adam_stochastic_gradient_descent(loss_fn,
                                             nat_params,
                                             adam_params,
                                             iters,
                                             learning_rate=0.001,
                                             gamma=0.,
                                             beta1=0.9,
                                             beta2=0.999,
                                             epsilon=1e-8,
                                             parallel_iterations=10):
    """
    Inspired by https://arxiv.org/pdf/1803.09151.pdf

    :param self:
    :param log_prob:
    :param initial_state:
    :param iters:
    :param stepsizes:
    :param learning_rate:
    :param parallel_iterations:
    :return:
    """
    #TODO: natural gradients

    learning_rate = tf.convert_to_tensor(learning_rate, float_type, 'learning_rate')
    beta1 = tf.convert_to_tensor(beta1, float_type, 'beta1')
    beta2 = tf.convert_to_tensor(beta2, float_type, 'beta2')
    epsilon = tf.convert_to_tensor(epsilon, float_type, 'epsilon')

    if not isinstance(nat_params, (tuple, list)):
        raise ValueError('nat params should be list like')

    if not isinstance(adam_params, (tuple, list)):
        raise ValueError('adam params should be list like')

    nat_params = list(nat_params)
    adam_params = list(adam_params)

    m0 = [tf.zeros_like(v) for v in adam_params]
    v0 = [tf.zeros_like(v) for v in adam_params]

    def _body(t, nat_params, adam_params, m, v, loss_ta):

        loss = tf.reduce_mean(loss_fn(*nat_params, *adam_params))
        loss_ta = loss_ta.write(t, loss)

        nat_grads = tf.gradients(loss, nat_params)

        next_nat_params = _natgrad_update(nat_grads, nat_params)

        adam_grads = tf.gradients(loss, adam_params)

        next_adam_params, next_m, next_v = _adam_update(adam_grads, adam_params, m, t, v)
        [n.set_shape(p.shape) for n, p in zip(next_adam_params, adam_params)]
        [n.set_shape(p.shape) for n, p in zip(next_nat_params, nat_params)]

        return t+1, next_nat_params, next_adam_params, next_m, next_v, loss_ta

    def _natgrad_update(nat_grads, nat_params):
        q_mean, q_scale = nat_params
        q_sqrt = tf.nn.softplus(q_scale)
        diag_F_q_mean_inv = tf.math.square(q_sqrt)
        diag_F_q_scale_inv = 0.5 * diag_F_q_mean_inv * tf.math.square(tf.math.reciprocal(tf.nn.sigmoid(q_scale)))
        next_nat_params = [q_mean - gamma * diag_F_q_mean_inv * nat_grads[0], q_scale - gamma * diag_F_q_scale_inv * \
                          nat_grads[1]]
        return next_nat_params

    def _adam_update(adam_grads, adam_params, m, t, v):
        t_float = tf.cast(t, float_type) + 1.
        lr_t = learning_rate * \
               tf.math.sqrt(1. - tf.math.pow(beta2, t_float)) * \
               tf.math.reciprocal(tf.math.sqrt(1. - tf.math.pow(beta1, t_float)))
        next_m, next_v, next_adam_params = [], [], []
        for (m_t, v_t, p_t, g_t) in zip(m, v, adam_params, adam_grads):
            if g_t is None:
                next_m.append(m_t)
                next_v.append(v_t)
                next_adam_params.append(p_t)
                continue
            m_t = beta1 * m_t + (1. - beta1) * g_t
            v_t = beta2 * v_t + (1. - beta2) * tf.math.square(g_t)
            p_t = p_t - lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon)
            next_m.append(m_t)
            next_v.append(v_t)
            next_adam_params.append(p_t)
        return next_adam_params, next_m, next_v

    loss_ta = tf.TensorArray(dtype=float_type, size=iters, infer_shape=False,element_shape=())

    _, nat_params, adam_params, m, v, loss_ta = tf.while_loop(lambda i, *args: i < iters,
                                    _body,
                                    (tf.constant(0, dtype=tf.int32),
                                     nat_params,
                                     adam_params,
                                     m0,
                                     v0,
                                     loss_ta),
                                    parallel_iterations=parallel_iterations,
                                    back_prop=False,
                                    return_same_structure=True)

    return nat_params, adam_params, loss_ta.stack()


def natural_adam_stochastic_gradient_descent_with_linesearch(
        loss_fn,
        nat_params,
        adam_params,
        iters=100,
        learning_rate=0.1,
        gamma=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        stop_patience=3,
        parallel_iterations=10):
    """
    Inspired by https://arxiv.org/pdf/1803.09151.pdf
    uses linesearch

    :param log_prob:
    :param initial_state:
    :param iters:
    :param stepsizes:
    :param learning_rate:
    :param parallel_iterations:
    :return:
    """
    #TODO: natural gradients

    learning_rate = tf.convert_to_tensor(learning_rate, float_type, 'learning_rate')
    gamma = tf.convert_to_tensor(gamma, float_type, 'gamma')
    stop_patience = tf.convert_to_tensor(stop_patience, tf.int32, 'stop_patience')
    iters = tf.convert_to_tensor(iters, tf.int32, 'iters')
    beta1 = tf.convert_to_tensor(beta1, float_type, 'beta1')
    beta2 = tf.convert_to_tensor(beta2, float_type, 'beta2')
    epsilon = tf.convert_to_tensor(epsilon, float_type, 'epsilon')

    if not isinstance(nat_params, (tuple, list)):
        raise ValueError('nat params should be list like')

    if not isinstance(adam_params, (tuple, list)):
        raise ValueError('adam params should be list like')

    nat_params = list(nat_params)
    adam_params = list(adam_params)

    m0 = [tf.zeros_like(v) for v in adam_params]
    v0 = [tf.zeros_like(v) for v in adam_params]

    def _body(t, nat_params, adam_params, m, v, loss_ta, min_loss, patience):

        loss = tf.reduce_mean(loss_fn(*nat_params, *adam_params))
        loss_better = tf.less_equal(loss, min_loss)
        min_loss = tf.minimum(min_loss, loss)
        patience = tf.cond(loss_better, lambda: tf.constant(0, patience.dtype),
                           lambda: patience + tf.constant(1, patience.dtype))

        loss_ta = loss_ta.write(t, loss)

        nat_grads = tf.gradients(loss, nat_params)

        next_nat_params = _natgrad_update(nat_grads, nat_params, adam_params,loss, t)

        adam_grads = tf.gradients(loss, adam_params)

        next_adam_params, next_m, next_v = _adam_update(adam_grads, adam_params, nat_params, m, t, v, loss)
        [n.set_shape(p.shape) for n, p in zip(next_adam_params, adam_params)]
        [n.set_shape(p.shape) for n, p in zip(next_nat_params, nat_params)]

        return t+1, next_nat_params, next_adam_params, next_m, next_v, loss_ta, min_loss, patience

    def _natgrad_update(nat_grads, nat_params, adam_params, loss0, t):
        q_mean, q_scale = nat_params
        q_sqrt = tf.nn.softplus(q_scale)
        diag_F_q_mean_inv = tf.math.square(q_sqrt)
        diag_F_q_scale_inv = 0.5 * diag_F_q_mean_inv * tf.math.square(tf.math.reciprocal(tf.nn.sigmoid(q_scale)))

        def search_function(a):
            test_nat_params = [q_mean - a * diag_F_q_mean_inv * nat_grads[0], q_scale - a * diag_F_q_scale_inv * \
                               nat_grads[1]]
            #TODO: don't need to redo L because adams params fixed
            loss = tf.reduce_mean(loss_fn(*test_nat_params, *adam_params))
            return loss - loss0

        search_space = tf.math.exp(tf.cast(tf.linspace(tf.math.log(gamma)-7., tf.math.log(gamma), 5),float_type))
        search_results = tf.map_fn(search_function, search_space, parallel_iterations=5)
        argmin = tf.argmin(search_results)

        a = search_space[argmin]
        loss_min = search_results[argmin]

        with tf.control_dependencies([tf.print('Step:', t, 'Optimal','Gamma:', a, 'loss reduction', loss_min)]):
            next_nat_params = [q_mean - a * diag_F_q_mean_inv * nat_grads[0], q_scale - a * diag_F_q_scale_inv * \
                           nat_grads[1]]

            return next_nat_params

    def _adam_update(adam_grads, adam_params, nat_params, m, t, v, loss0):
        t_float = tf.cast(t, float_type) + 1.
        lr_t = tf.math.sqrt(1. - tf.math.pow(beta2, t_float)) * \
               tf.math.reciprocal(tf.math.sqrt(1. - tf.math.pow(beta1, t_float)))
        next_m, next_v = [], []
        for (m_t, v_t, g_t) in zip(m, v, adam_grads):
            if g_t is None:
                next_m.append(m_t)
                next_v.append(v_t)
                continue
            m_t = beta1 * m_t + (1. - beta1) * g_t
            v_t = beta2 * v_t + (1. - beta2) * tf.math.square(g_t)
            next_m.append(m_t)
            next_v.append(v_t)

            # p_t = p_t - lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon)

        def search_function(a):
            #TODO: don't need to redo predictive x because nat_params fixed
            test_adam_params = []
            for (m_t, v_t, p_t, g_t) in zip(next_m, next_v, adam_params, adam_grads):
                if g_t is None:
                    next_adam_params.append(p_t)
                    continue
                test_adam_params.append(p_t - a * lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon))
            loss = tf.reduce_mean(loss_fn(*nat_params, *test_adam_params))
            return loss - loss0

        search_space = tf.math.exp(tf.cast(tf.linspace(tf.math.log(learning_rate) - 7., tf.math.log(learning_rate), 5), float_type))
        search_results = tf.map_fn(search_function, search_space, parallel_iterations=5)
        argmin = tf.argmin(search_results)

        a = search_space[argmin]
        loss_min = search_results[argmin]

        with tf.control_dependencies([tf.print('Step:', t, 'Optimal', 'Learning rate:', a, 'loss reduction', loss_min)]):
            next_adam_params = []
            for (m_t, v_t, p_t, g_t) in zip(next_m, next_v, adam_params, adam_grads):
                if g_t is None:
                    next_adam_params.append(p_t)
                    continue
                next_adam_params.append(p_t - a * lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon))

        return next_adam_params, next_m, next_v

    def _cond(t, nat_params, adam_params, m, v, loss_ta, min_loss, patience):
        return tf.logical_and(tf.less(patience, stop_patience), tf.less(t, iters))

    loss_ta = tf.TensorArray(dtype=float_type, size=iters, infer_shape=False,element_shape=())

    _, nat_params, adam_params, m, v, loss_ta, _, _ = tf.while_loop(_cond,
                                    _body,
                                    (tf.constant(0, dtype=tf.int32),
                                     nat_params,
                                     adam_params,
                                     m0,
                                     v0,
                                     loss_ta,
                                     tf.constant(np.inf,float_type),
                                     tf.constant(0, tf.int32)),
                                    parallel_iterations=parallel_iterations,
                                    back_prop=False,
                                    return_same_structure=True)

    return nat_params, adam_params, loss_ta.stack()


def natural_adam_stochastic_gradient_descent_with_linesearch_minibatch(
        loss_fn,
        X,
        Y,
        minibatch_size,
        nat_params,
        adam_params,
        iters=100,
        learning_rate=0.1,
        gamma=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        stop_patience=3,
        num_linesearch=5,
        parallel_iterations=10):
    """
    Inspired by https://arxiv.org/pdf/1803.09151.pdf
    uses linesearch and minibatch

    :param loss_fn:
    :param X:
    :param Y:
    :param minibatch_size:
    :param nat_params:
    :param adam_params:
    :param iters:
    :param learning_rate:
    :param gamma:
    :param beta1:
    :param beta2:
    :param epsilon:
    :param stop_patience:
    :param parallel_iterations:
    :return:
    """
    #TODO: natural gradients

    learning_rate = tf.convert_to_tensor(learning_rate, float_type, 'learning_rate')
    gamma = tf.convert_to_tensor(gamma, float_type, 'gamma')
    stop_patience = tf.convert_to_tensor(stop_patience, tf.int32, 'stop_patience')
    iters = tf.convert_to_tensor(iters, tf.int32, 'iters')
    beta1 = tf.convert_to_tensor(beta1, float_type, 'beta1')
    beta2 = tf.convert_to_tensor(beta2, float_type, 'beta2')
    epsilon = tf.convert_to_tensor(epsilon, float_type, 'epsilon')

    if not isinstance(nat_params, (tuple, list)):
        raise ValueError('nat params should be list like')

    if not isinstance(adam_params, (tuple, list)):
        raise ValueError('adam params should be list like')

    nat_params = list(nat_params)
    adam_params = list(adam_params)

    m0 = [tf.zeros_like(v) for v in adam_params]
    v0 = [tf.zeros_like(v) for v in adam_params]

    N = tf.shape(X)[0]

    def _body(t, nat_params, adam_params, m, v, loss_ta, min_loss, patience):
        if minibatch_size is not None:
            minibatch_selection = tf.random.shuffle(tf.range(N))[:minibatch_size]
            _X = tf.gather(X, minibatch_selection,axis=0)
            _Y = (tf.gather(Y[0], minibatch_selection, axis=0), tf.gather(Y[1], minibatch_selection, axis=0))
        else:
            _X = X
            _Y = Y

        loss = tf.reduce_mean(loss_fn(*nat_params, *adam_params, _X, _Y))
        loss_better = tf.less_equal(loss, min_loss)
        min_loss = tf.minimum(min_loss, loss)
        patience = tf.cond(loss_better, lambda: tf.constant(0, patience.dtype),
                           lambda: patience + tf.constant(1, patience.dtype))

        loss_ta = loss_ta.write(t, loss)

        nat_grads = tf.gradients(loss, nat_params)

        next_nat_params = _natgrad_update(nat_grads, nat_params, adam_params,loss, t, _X, _Y)

        adam_grads = tf.gradients(loss, adam_params)

        next_adam_params, next_m, next_v = _adam_update(adam_grads, adam_params, nat_params, m, t, v, loss, _X, _Y)
        [n.set_shape(p.shape) for n, p in zip(next_adam_params, adam_params)]
        [n.set_shape(p.shape) for n, p in zip(next_nat_params, nat_params)]

        return t+1, next_nat_params, next_adam_params, next_m, next_v, loss_ta, min_loss, patience

    def _natgrad_update(nat_grads, nat_params, adam_params, loss0, t, X, Y):
        q_mean, q_scale = nat_params
        q_sqrt = tf.nn.softplus(q_scale)
        diag_F_q_mean_inv = tf.math.square(q_sqrt)
        diag_F_q_scale_inv = 0.5 * diag_F_q_mean_inv * tf.math.square(tf.math.reciprocal(tf.nn.sigmoid(q_scale)))

        def search_function(a):
            test_nat_params = [q_mean - a * diag_F_q_mean_inv * nat_grads[0], q_scale - a * diag_F_q_scale_inv * \
                               nat_grads[1]]
            #TODO: don't need to redo L because adams params fixed
            loss = tf.reduce_mean(loss_fn(*test_nat_params, *adam_params, X, Y))
            return loss - loss0

        search_space = tf.math.exp(tf.cast(tf.linspace(tf.math.log(gamma)-7., tf.math.log(gamma), num_linesearch),float_type))
        search_results = tf.map_fn(search_function, search_space, parallel_iterations=num_linesearch)
        argmin = tf.argmin(search_results)

        a = search_space[argmin]
        loss_min = search_results[argmin]

        with tf.control_dependencies([tf.print('Step:', t, 'Optimal','Gamma:', a, 'loss reduction', loss_min)]):
            next_nat_params = [q_mean - a * diag_F_q_mean_inv * nat_grads[0], q_scale - a * diag_F_q_scale_inv * \
                           nat_grads[1]]

            return next_nat_params

    def _adam_update(adam_grads, adam_params, nat_params, m, t, v, loss0, X, Y):
        t_float = tf.cast(t, float_type) + 1.
        lr_t = tf.math.sqrt(1. - tf.math.pow(beta2, t_float)) * \
               tf.math.reciprocal(tf.math.sqrt(1. - tf.math.pow(beta1, t_float)))
        next_m, next_v = [], []
        for (m_t, v_t, g_t) in zip(m, v, adam_grads):
            if g_t is None:
                next_m.append(m_t)
                next_v.append(v_t)
                continue
            m_t = beta1 * m_t + (1. - beta1) * g_t
            v_t = beta2 * v_t + (1. - beta2) * tf.math.square(g_t)
            next_m.append(m_t)
            next_v.append(v_t)

            # p_t = p_t - lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon)

        def search_function(a):
            #TODO: don't need to redo predictive x because nat_params fixed
            test_adam_params = []
            for (m_t, v_t, p_t, g_t) in zip(next_m, next_v, adam_params, adam_grads):
                if g_t is None:
                    next_adam_params.append(p_t)
                    continue
                test_adam_params.append(p_t - a * lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon))
            loss = tf.reduce_mean(loss_fn(*nat_params, *test_adam_params, X, Y))
            return loss - loss0

        search_space = tf.math.exp(tf.cast(tf.linspace(tf.math.log(learning_rate) - 7., tf.math.log(learning_rate), num_linesearch), float_type))
        search_results = tf.map_fn(search_function, search_space, parallel_iterations=num_linesearch)
        argmin = tf.argmin(search_results)

        a = search_space[argmin]
        loss_min = search_results[argmin]

        with tf.control_dependencies([tf.print('Step:', t, 'Optimal', 'Learning rate:', a, 'loss reduction', loss_min)]):
            next_adam_params = []
            for (m_t, v_t, p_t, g_t) in zip(next_m, next_v, adam_params, adam_grads):
                if g_t is None:
                    next_adam_params.append(p_t)
                    continue
                next_adam_params.append(p_t - a * lr_t * m_t * tf.math.reciprocal(tf.math.sqrt(v_t) + epsilon))

        return next_adam_params, next_m, next_v

    def _cond(t, nat_params, adam_params, m, v, loss_ta, min_loss, patience):
        return tf.logical_and(tf.less(patience, stop_patience), tf.less(t, iters))

    loss_ta = tf.TensorArray(dtype=float_type, size=iters, infer_shape=False,element_shape=())

    t, nat_params, adam_params, m, v, loss_ta, _, _ = tf.while_loop(_cond,
                                    _body,
                                    (tf.constant(0, dtype=tf.int32),
                                     nat_params,
                                     adam_params,
                                     m0,
                                     v0,
                                     loss_ta,
                                     tf.constant(np.inf,float_type),
                                     tf.constant(0, tf.int32)),
                                    parallel_iterations=parallel_iterations,
                                    back_prop=False,
                                    return_same_structure=True)

    return nat_params, adam_params, loss_ta.stack(), t