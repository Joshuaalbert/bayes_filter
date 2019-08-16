import tensorflow as tf
from . import float_type
from gpflow.kernels import Kernel
from gpflow.params import Parameter
from gpflow import params_as_tensors
from gpflow import settings
from gpflow import transforms
import numpy as np
from .model import HGPR
from .directional_models import ArcCosineEQ, gpflow_kernel, Piecewise

class NonIntegralKernel(Kernel):
    def __init__(self, input_dim,
                 amplitude=1., dirscale=0.1, antscale=10.,
                 ref_direction=[0., 0., 1.],
                 ref_location = [0., 0., 0.],
                 ant_anisotropic=False,
                 dir_anisotropic=False,
                 active_dims=None,
                 dir_kernel = ArcCosineEQ,
                 ant_kernel=Piecewise,
                 obs_type='DDTEC', name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.amplitude = Parameter(amplitude,
                                   transform=transforms.positiveRescale(amplitude),
                                   dtype=settings.float_type)
        self.dirscale = Parameter(dirscale,
                                  transform=transforms.positiveRescale(dirscale),
                                  dtype=settings.float_type)
        self.antscale = Parameter(dirscale,
                                  transform=transforms.positiveRescale(antscale),
                                  dtype=settings.float_type)
        self.dir_kernel = dir_kernel
        self.ant_kernel = ant_kernel

        self.obs_type = obs_type
        self.ref_direction = Parameter(ref_direction,
                                       dtype=float_type, trainable=False)
        self.ref_location = Parameter(ref_location,
                                       dtype=float_type, trainable=False)
        self.ant_anisotropic = ant_anisotropic
        self.dir_anisotropic = dir_anisotropic
        if self.ant_anisotropic:
            # Na, 3, 3
            self.ant_M = Parameter(np.eye(3), dtype=float_type,
                               transform=transforms.LowerTriangular(3, squeeze=True))

        if self.dir_anisotropic:
            # Na, 3, 3
            self.dir_M = Parameter(np.eye(3), dtype=float_type,
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


        k1 = X1[..., 0:3]
        if sym:
            k2 = X2[..., 0:3]
        else:
            k2 = X2[..., 0:3]

        x1 = X1[..., 3:6]
        if sym:
            x2 = X2[..., 3:6]
        else:
            x2 = X2[..., 3:6]

        if self.dir_anisotropic:
            # M_ij.k_nj
            k1 = tf.matmul(k1, self.dir_M, transpose_b=True)
            if sym:
                k2 = k1
            else:
                k2 = tf.matmul(k2, self.dir_M, transpose_b=True)

        if self.ant_anisotropic:
            # M_ij.x_nj
            x1 = tf.matmul(x1, self.ant_M, transpose_b=True)
            if sym:
                x2 = x1
            else:
                x2 = tf.matmul(x2, self.ant_M, transpose_b=True)

        kern_dir = self.dir_kernel(
            length_scale=self.dirscale)
        kern_ant = self.ant_kernel(
            length_scale=self.antscale)

        res = None
        if self.obs_type == 'TEC':
            res = kern_dir(k1, k2)
        if self.obs_type == 'DTEC':
            if sym:
                ant_sym = kern_ant(x1, self.ref_direction[None, :])
                res = kern_ant(x1, x2) - ant_sym - tf.transpose(ant_sym, (1, 0)) + 1.
            res = kern_dir(x1, x2) - kern_dir(self.ref_location[None, :], x2) - kern_ant(x1, self.ref_location[None,:]) + 1.

        if self.obs_type == 'DDTEC':
            if sym:
                dir_sym = kern_dir(k1, self.ref_direction[None, :])
                ant_sym = kern_ant(x1, self.ref_direction[None, :])
                res = (kern_ant(x1, x2) - ant_sym - tf.transpose(ant_sym, (1, 0)) + 1.)*(kern_dir(k1, k2) - dir_sym - tf.transpose(dir_sym, (1,0)) + 1.)
            res =  (kern_dir(x1, x2) - kern_dir(self.ref_location[None, :], x2) - kern_ant(x1, self.ref_location[None,:]) + 1.)*(kern_dir(k1, k2) - kern_dir(self.ref_direction[None, :], k2) - kern_dir(k1,self.ref_direction[None,:]) + 1.)
        return tf.math.square(self.amplitude) * res

def generate_models(X, Y, Y_var, ref_direction, ref_location, initial_amplitude=1., initial_dirscale=0.1, initial_antscale=10., reg_param = 1., parallel_iterations=10):

    dir_kernels = [Piecewise, ArcCosineEQ, gpflow_kernel('RBF'), gpflow_kernel('M52'), gpflow_kernel('M32'), gpflow_kernel('ArcCosine')]
    ant_kernels = [Piecewise, ArcCosineEQ, gpflow_kernel('RBF'), gpflow_kernel('M52'), gpflow_kernel('M32'), gpflow_kernel('ArcCosine')]
    kernels = []
    for a in ant_kernels:
        for d in dir_kernels:
            kernels.append(NonIntegralKernel(6,
                                amplitude=initial_amplitude,
                                dirscale=initial_dirscale,
                                 antscale=initial_antscale,
                                ref_direction=ref_direction,
                                 ref_location=ref_location,
                                ant_anisotropic=False,
                                 dir_anisotropic=False,
                                dir_kernel=d,
                                 ant_kernel = a,
                                obs_type='DDTEC'))

    models = [HGPR(X, Y, Y_var, kern, regularisation_param=reg_param, parallel_iterations=parallel_iterations)
              for kern in kernels]

    return models