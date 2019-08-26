import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.params import Parameter
from gpflow import params_as_tensors
from gpflow import settings
from gpflow import transforms
import numpy as np
from .model import HGPR
from .directional_models import gpflow_kernel

float_type = settings.float_type

from . import KERNEL_SCALE


def _validate_arg_if_not_none(arg, assertion, validate_args):
    if arg is None:
        return arg
    with tf.control_dependencies([assertion(arg)] if validate_args else []):
        result = tf.identity(arg)
    return result


class TomographicKernel(Kernel):
    """
        The DTEC kernel is derived from first principles by assuming a GRF over the electron density, from which DTEC kernel
        can be caluclated as,

        K(ray_i, ray_j) =     I(a_i, k_i, t_i, a_j, k_j, t_j)  + I(a0_i, k_i, t_i, a0_j, k_j, t_j)
                            - I(a0_i, k_i, t_i, a_j, k_j, t_j) - I(a_i, k_i, t_i, a0_j, k_j, t_j)

        where,
                    I(a,b,c,d,e,g) = iint [K_ne(y(a,b,c), y(d,e,f))](s1,s2) ds1 ds2
        """

    def __init__(self,
                 a=250.,
                 b=100.,
                 ref_direction=[0., 0., 1.],
                 ref_location=[0., 0., 0.],
                 anisotropic=False,
                 active_dims=None,
                 fed_kernel: Kernel = None,
                 obs_type='DDTEC',
                 ionosphere_type='flat',
                 resolution=8):
        super().__init__(6, active_dims,
                         name="TrapKernel{}_{}{}".format(resolution,
                                                         'aniso' if anisotropic else "iso",
                                                         fed_kernel.name))

        self.resolution = resolution
        self.fed_kernel = fed_kernel

        self.obs_type = obs_type
        if ionosphere_type == 'curved':
            raise ValueError("Curved not implemented yet.")
        self.ionosphere_type = ionosphere_type
        self.a = Parameter(a, dtype=float_type, transform=transforms.positiveRescale(a))
        self.b = Parameter(b, dtype=float_type, transform=transforms.positiveRescale(b))
        self.ref_direction = Parameter(ref_direction,
                                       dtype=float_type, trainable=False)
        self.ref_location = Parameter(ref_location,
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
    def calculate_ray_endpoints(self, x, k):
        """
        Calculate the s where x+k*(s- + Ds*s) intersects the ionosphere.
        l = x + k*s-
        m = k*Ds

        :param x:
        :param k:
        :return:
        """
        with tf.name_scope('calculate_ray_endpoints', values=[x, k]):
            if self.ionosphere_type == 'flat':
                # N
                sec = tf.math.reciprocal(k[:, 2], name='secphi')

                # N
                bsec = sec * self.b

                # N
                sm = sec * (self.a + self.ref_location[2] - x[:, 2]) - 0.5 * bsec
                # N, 3
                l = x + k * sm[:, None]
                # N, 3
                m = k * bsec[:, None]
                Delta_s = bsec
                return l, m, Delta_s

        raise NotImplementedError("curved not implemented")

    @params_as_tensors
    def K(self, X1, X2, presliced=False):
        if not presliced:
            X1, X2 = self._slice(X1, X2)

        self.sym = False
        if X2 is None:
            X2 = X1
            self.sym = True

        with tf.name_scope('Tomographic_K'):
            coord_list = None
            I_coeff = None
            if self.obs_type in ['TEC', 'DTEC', 'DDTEC']:
                coord_list = [(X1[:, 0:3], X1[:, 3:6], X2[:, 0:3], X2[:, 3:6])]
                I_coeff = [1.]
            if self.obs_type in ['DTEC', 'DDTEC']:
                coord_list_prior = coord_list
                I_coeff_prior = I_coeff
                I_coeff = []
                coord_list = []
                for i in I_coeff_prior:
                    I_coeff.append(i)
                    I_coeff.append(-i)
                    I_coeff.append(-i)
                    I_coeff.append(i)
                for c in coord_list_prior:
                    coord_list.append(c)
                    coord_list.append((c[0], c[1], c[2], self.ref_location[None, :]))
                    coord_list.append((c[0], self.ref_location[None, :], c[2], c[3]))
                    coord_list.append((c[0], self.ref_location[None, :], c[2], self.ref_location[None, :]))
            if self.obs_type in ['DDTEC']:
                coord_list_prior = coord_list
                I_coeff_prior = I_coeff
                I_coeff = []
                coord_list = []
                for i in I_coeff_prior:
                    I_coeff.append(i)
                    I_coeff.append(-i)
                    I_coeff.append(-i)
                    I_coeff.append(i)
                for c in coord_list_prior:
                    coord_list.append(c)
                    coord_list.append((c[0], c[1], self.ref_direction[None, :], c[3]))
                    coord_list.append((self.ref_direction[None, :], c[1], c[2], c[3]))
                    coord_list.append((self.ref_direction[None, :], c[1], self.ref_direction[None, :], c[3]))
            IK = []
            for i, c in zip(I_coeff, coord_list):
                IK.append(i * self.I(*c))

            K = tf.math.square(tf.constant(KERNEL_SCALE, float_type)) * tf.add_n(IK)

            return K

    @params_as_tensors
    def I(self, k1, x1, k2, x2):
        """
        Calculate the ((D)D)TEC kernel based on the FED kernel.

        :param X: float_type, tf.Tensor (N, 7[10[13]])
            Coordinates in order (time, kx, ky, kz, x,y,z, [x0, y0, z0, [kx0, ky0, kz0]])
        :param X2:
            Second coordinates, if None then equal to X
        :return:
        """

        with tf.name_scope('Trapezoid_I'):
            # N, 3
            l1, m1, ds1 = self.calculate_ray_endpoints(x1, k1)
            if self.sym:
                l2, m2, ds2 = l1, m1, ds1
            else:
                # M, 3
                l2, m2, ds2 = self.calculate_ray_endpoints(x2, k2)

            if self.anisotropic:
                # M_ij.k_nj = kni
                l1 = tf.matmul(l1, self.M, transpose_b=True)
                m1 = tf.matmul(m1, self.M, transpose_b=True)
                if self.sym:
                    l2, m2 = l1, m1
                else:
                    l2 = tf.matmul(l2, self.M, transpose_b=True)
                    m2 = tf.matmul(m2, self.M, transpose_b=True)

            N = tf.shape(l1)[0]
            M = tf.shape(l2)[0]

            # res
            s = tf.cast(tf.linspace(0., 1., self.resolution + 1), dtype=float_type)
            # N,M
            ds = (ds1[:, None] * tf.math.reciprocal(tf.cast(self.resolution, float_type))) * (
                    ds2[None, :] * tf.math.reciprocal(tf.cast(self.resolution, float_type)))
            # res^2, res^2
            s1, s2 = tf.meshgrid(s, s, indexing='ij')
            # res^2, N, 3
            ray1 = l1 + m1 * tf.reshape(s1, (-1,))[:, None, None]
            # res^2, M, 3
            ray2 = l2 + m2 * tf.reshape(s2, (-1,))[:, None, None]

            # res^2, N,M
            I = tf.map_fn(lambda ray1, ray2: self.fed_kernel.K(ray1, ray2), (ray1, ray2), dtype=float_type)
            # res, res, N,M
            I = tf.reshape(I, (self.resolution + 1, self.resolution + 1, N, M))

            # N,M
            I = 0.25 * ds * tf.add_n([I[0, 0, :, :],
                                      I[-1, 0, :, :],
                                      I[0, -1, :, :],
                                      I[-1, -1, :, :],
                                      2 * tf.reduce_sum(I[-1, :, :, :], axis=[0]),
                                      2 * tf.reduce_sum(I[0, :, :, :], axis=[0]),
                                      2 * tf.reduce_sum(I[:, -1, :, :], axis=[0]),
                                      2 * tf.reduce_sum(I[:, 0, :, :], axis=[0]),
                                      4 * tf.reduce_sum(I[1:-1, 1:-1, :, :], axis=[0, 1])])
            return I


###
# %

def generate_models(X, Y, Y_var, ref_direction, ref_location, reg_param=1., parallel_iterations=10):
    fed_settings = [('RBF', dict(variance=1., lengthscales=10.)),
                    ('M52', dict(variance=1., lengthscales=10.)),
                    ('M32', dict(variance=1., lengthscales=10.)),
                    ('ArcCosine', dict(variance=1.))]
    kernels = []
    for k in fed_settings:
        kernels.append(TomographicKernel(a=250., b=100.,
                                         ref_direction=ref_direction,
                                         ref_location=ref_location,
                                         anisotropic=False,
                                         fed_kernel=gpflow_kernel(k[0], **k[1]),
                                         obs_type='DDTEC',
                                         resolution=5,
                                         ionosphere_type='flat'))

    models = [HGPR(X, Y, Y_var, kern, regularisation_param=reg_param, parallel_iterations=parallel_iterations,
                   name='HGPR_{}'.format(kern.name))
              for kern in kernels]

    return models
