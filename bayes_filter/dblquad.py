from .odeint import odeint, odeint_fixed
from . import float_type
import tensorflow as tf

def dblquad(func, a, b, lfun, ufun, shape, rtol=1e-6, atol=1e-12, **options):
    """
    Function that takes two scalars t1 and t2 and returns [shape].

    :param func:
    :param a:
    :param b:
    :param lfun:
    :param ufun:
    :return:
    """

    def _tempfunc(y, t1):
        l = lfun(t1)
        u = ufun(t1)
        T = tf.stack([l, u],axis=0)
        res = odeint_fixed(lambda y, t: func(t1, t), y, T, (u-l)/3., method='rk4')
        # res = odeint(lambda y, t: func(t1, t), y, T, method='dopri5', rtol=rtol, atol=atol, options=options)
        return res[1,...]-y

    I0 = tf.zeros(shape,dtype=float_type)
    T = tf.stack([a, b],axis=0)
    res = odeint_fixed(_tempfunc, I0, T, (b - a) / 3., method='rk4')
    # res, info_dict = odeint(_tempfunc,I0, T, method='dopri5', rtol=rtol, atol=atol, full_output=True, options=options)
    return res[1,...], None#info_dict
