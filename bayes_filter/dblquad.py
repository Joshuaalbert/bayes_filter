from .odeint import odeint, odeint_fixed
from . import float_type
import tensorflow as tf
from .settings import float_type

def dblquad(func, a, b, lfun, ufun, shape, ode_type='fixed', **options):
    """
    Function that takes two scalars t1 and t2 and returns [shape].

    :param func:
    :param a:
    :param b:
    :param lfun:
    :param ufun:
    :return:
    """

    if ode_type == 'fixed':
        resolution = tf.convert_to_tensor(options.pop('resolution',3.), float_type)

    if ode_type == 'adaptive':
        rtol = tf.convert_to_tensor(options.pop('rtol',1e-3), float_type)
        atol = tf.convert_to_tensor(options.pop('atol', 1e-12), float_type)
        first_step = tf.convert_to_tensor(options.pop('first_step', 1./3.), float_type)

    def _tempfunc(y, t1):
        l = lfun(t1)
        u = ufun(t1)
        T = tf.stack([l, u],axis=0)
        if ode_type == 'fixed':
            res = odeint_fixed(lambda y, t: func(t1, t), y, T, (u-l)/resolution, method='rk4')
        elif ode_type == 'adaptive':
            res = odeint(lambda y, t: func(t1, t), y, T, method='dopri5', rtol=rtol, atol=atol,
                         options={'first_step':first_step})
        return res[1,...]-y

    I0 = tf.zeros(shape,dtype=float_type)
    T = tf.stack([a, b],axis=0)
    if ode_type == 'fixed':
        res = odeint_fixed(_tempfunc, I0, T, (b - a) / resolution, method='rk4')
        return res[1, ...], None
    elif ode_type == 'adaptive':
        res, info_dict = odeint(_tempfunc,I0, T, method='dopri5', rtol=rtol, atol=atol, full_output=True,
                                options={'first_step':first_step})
        return res[1,...], info_dict
