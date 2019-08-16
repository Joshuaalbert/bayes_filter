import tensorflow as tf
from . import float_type
from gpflow.kernels import Kernel
from gpflow.params import Parameter
from gpflow import params_as_tensors
from gpflow import settings
from gpflow import transforms
import numpy as np
from .model import HGPR
from gpflow.kernels import Matern52, Matern32, RBF, ArcCosine

@tf.custom_gradient
def safe_acos_squared(x):
    safe_x = tf.clip_by_value(x, tf.constant(-1., float_type), tf.constant(1., float_type))
    acos = tf.math.acos(safe_x)
    result = tf.math.square(acos)
    def grad(dy):
        g = -2.*acos/tf.math.sqrt(1. - tf.math.square(safe_x))
        g = tf.where(tf.equal(safe_x, tf.constant(1., float_type)), tf.constant(-2., float_type)*tf.ones_like(g), g)
        g = tf.where(tf.equal(safe_x, tf.constant(-1., float_type)), tf.constant(-100, float_type)*tf.ones_like(g), g)
        with tf.control_dependencies([tf.print(tf.reduce_all(tf.is_finite(g)), tf.reduce_all(tf.is_finite(dy)))]):
            return g*dy
    return result, grad


class ArcCosineEQ(object):

    def __init__(
            self,
            amplitude=None,
            length_scale=None,
            name='ArcCosineRBF'):

        with tf.name_scope(name) as name:
            self.amplitude = amplitude
            self.length_scale = length_scale
            self.name = name

    def __call__(self, x1, x2):
        """
        :param x1: tf.Tensor
            [..., b, d]
        :param x2: tf.Tensor
            [..., c, d]
        :return: tf.Tensor
            [..., b, c]

        """
        with tf.name_scope(self.name, values=[x1, x2]):
            # ..., b, c
            dot = tf.reduce_sum(
                tf.math.multiply(x1[..., :, None, :], x2[..., None, :, :]), axis=-1)
            log_res = safe_acos_squared(dot)
            log_res *= -0.5
            if self.length_scale is not None:
                log_res /= self.length_scale ** 2
            if self.amplitude is not None:
                log_res += 2. * tf.math.log(self.amplitude)
            return tf.math.exp(log_res)


class Piecewise(object):

    def __init__(
            self,
            amplitude=None,
            length_scale=None,
            name='Piecewise'):
        with tf.name_scope(name) as name:
            self.amplitude = amplitude
            self.length_scale = length_scale
            self.name = name

    def __call__(self, x1, x2):
        """
        :param x1: tf.Tensor
            [..., b, d]
        :param x2: tf.Tensor
            [..., c, d]
        :return: tf.Tensor
            [..., b, c]

        """
        with tf.name_scope(self.name, values=[x1, x2]):
            # ..., b, c
            diff = tf.reduce_sum(
                tf.math.squared_difference(x1[..., :, None, :], x2[..., None, :, :]), axis=-1)

            res = tf.where(tf.greater_equal(diff, self.length_scale ** 2), tf.zeros_like(diff), diff)
            if self.amplitude is not None:
                res = self.amplitude ** 2 * res
            return res


def gpflow_kernel(kernel, dims = 3):
    class GPFlowKernel(object):
        def __init__(
                self,
                amplitude=None,
                length_scale=None,
                name='GPFlowWrapper'):

            with tf.name_scope(name) as name:
                if kernel == 'RBF':
                    self.kernel = RBF(dims, lengthscales = length_scale, variance = amplitude**2)
                if kernel == 'M32':
                    self.kernel = Matern32(dims, lengthscales=length_scale, variance=amplitude ** 2)
                if kernel == 'M52':
                    self.kernel = Matern52(dims, lengthscales=length_scale, variance=amplitude ** 2)
                if kernel == 'ArcCosine':
                    self.kernel = ArcCosine(dims, variance=amplitude ** 2)
                self.name = name

        def __call__(self, x1, x2):
            """
            :param x1: tf.Tensor
                [..., b, d]
            :param x2: tf.Tensor
                [..., c, d]
            :return: tf.Tensor
                [..., b, c]

            """
            with tf.name_scope(self.name, values=[x1, x2]):
                # ..., b, c
                return tf.map_fn(lambda x1,x2 :self.kernel.K(x1,x2), [x1,x2], dtype=float_type, parallel_iterations=10)

    return GPFlowKernel

class DirectionalKernel(Kernel):
    def __init__(self, input_dim,
                 amplitude=1., dirscale=0.1,
                 ref_direction=[0., 0., 1.],
                 anisotropic=False,
                 active_dims=None,
                 inner_kernel = ArcCosineEQ,
                 obs_type='DDTEC', name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.amplitude = Parameter(amplitude,
                                   transform=transforms.positiveRescale(amplitude),
                                   dtype=settings.float_type)
        self.dirscale = Parameter(dirscale,
                                  transform=transforms.positiveRescale(dirscale),
                                  dtype=settings.float_type)
        self.inner_kernel = inner_kernel

        self.obs_type = obs_type
        self.ref_direction = Parameter(ref_direction,
                                       dtype=float_type, trainable=False)
        self.anisotropic = anisotropic
        if self.anisotropic:
            # Na, 3, 3
            self.M = Parameter(np.eye(3), dtype=float_type,
                               transform=transforms.LowerTriangular(3, squeeze=True))

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.linalg.diag_part(self.K(X, None))

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):

        if not presliced:
            X1, X2 = self._slice(X1, X2)

        sym = False
        if X2 is None:
            X2 = X1
            sym = True

        k1 = X1
        k2 = X2

        if self.anisotropic:
            # M_ij.k_nj
            k1 = tf.matmul(k1, self.M, transpose_b=True)
            if sym:
                k2 = k1
            else:
                k2 = tf.matmul(k2, self.M, transpose_b=True)

        kern_dir = self.inner_kernel(
            length_scale=self.dirscale)
        res = None
        if self.obs_type == 'TEC' or self.obs_type == 'DTEC':
            res = kern_dir(k1, k2)
        if self.obs_type == 'DDTEC':
            if sym:
                dir_sym = kern_dir(k1, self.ref_direction[None, :])
                res = kern_dir(k1, k2) - dir_sym - tf.transpose(dir_sym, (1,0)) + 1.
            res =  kern_dir(k1, k2) - kern_dir(self.ref_direction[None, :], k2) - kern_dir(k1,self.ref_direction[None,:]) + 1.
        return tf.math.square(self.amplitude) * res

def generate_models(X, Y, Y_var, ref_direction, initial_amplitude=1., initial_dirscale=0.1, reg_param = 1., parallel_iterations=10):
    dir_kernels = [Piecewise, ArcCosineEQ, gpflow_kernel('RBF'), gpflow_kernel('M52'), gpflow_kernel('M32'),
                   gpflow_kernel('ArcCosine')]
    kernels = []
    for d in dir_kernels:
        kernels.append(DirectionalKernel(3,
                                        amplitude=initial_amplitude,
                                        dirscale=initial_dirscale,
                                        ref_direction=ref_direction,
                                        anisotropic=False,
                                        inner_kernel=d,
                                        obs_type='DDTEC'))

    models = [HGPR(X, Y, Y_var, kern, regularisation_param=reg_param, parallel_iterations=parallel_iterations)
              for kern in kernels]

    return models

