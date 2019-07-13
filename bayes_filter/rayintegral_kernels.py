import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

import tensorflow as tf
import numpy as np
from . import logging, float_type
from collections import OrderedDict
import itertools



def _validate_arg_if_not_none(arg, assertion, validate_args):
    if arg is None:
        return arg
    with tf.control_dependencies([assertion(arg)] if validate_args else []):
        result = tf.identity(arg)
    return result

class RayKernel(object):
    def __init__(self, a, b, mu=None, ref_location = None, ref_direction = None, obs_type='DTEC', ionosphere_type='flat'):
        if ionosphere_type == 'curved':
            if mu is None:
                raise ValueError("Need a mu for curved.")
        self.a = a
        self.b = b
        self.mu = mu
        self.obs_type = obs_type
        self.ref_location = ref_location
        self.ref_direction = ref_direction
        self.ionosphere_type = ionosphere_type

    def calculate_ray_endpoints(self, x, k):
        """
        Calculate the s where x+k*(s- + Ds*s) intersects the ionosphere.
        l = x + k*s-
        m = k*Ds

        :param x:
        :param k:
        :return:
        """
        if self.ionosphere_type == 'flat':
            # N
            sec = tf.math.reciprocal(k[:, 2], name='secphi')

            # N
            bsec = sec * self.b

            # N
            sm = sec * (self.a + self.ref_location[2] - x[:, 2]) - 0.5 * bsec
            # N
            sp = sm + bsec
            l = x + k*sm[:, None]
            m = k*bsec[:, None]
            # s_lower = tf.zeros_like(k[:, 2])
            # s_upper = tf.ones_like(k[:, 2])
            dl_da = k*sec[:, None]
            dl_db = -0.5*dl_da#-0.5*k*sec[:, None]
            dm_da = tf.zeros_like(m)
            dm_db = dl_da #k*sec[:, None]
            return l, m,(dl_da, dl_db), (dm_da,dm_db)

        raise NotImplementedError("curved not implemented")

    def _replace_ant(self, X, x):
        x_tile = tf.tile(x[None,:], (tf.shape(X)[0], 1))
        return tf.concat([X[:,0:3], x_tile], axis=1)

    def _replace_dir(self, X, x):
        x_tile = tf.tile(x[None,:], (tf.shape(X)[0], 1))
        return tf.concat([x_tile, X[:,3:6]], axis=1)

    def K(self, X1, X2):
        coord_list = None
        I_coeff = None
        if self.obs_type in ['TEC', 'DTEC', 'DDTEC']:
            coord_list = [(X1, X2)]
            I_coeff = [1.]
        if self.obs_type == ['DTEC', 'DDTEC']:
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
                coord_list.append((c[0], self._replace_ant(c[1], self.ref_location)))
                coord_list.append((self._replace_ant(c[0], self.ref_location), c[1]))
                coord_list.append((self._replace_ant(c[0], self.ref_location), self._replace_ant(c[1], self.ref_location)))
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
                coord_list.append((c[0], self._replace_dir(c[1], self.ref_direction)))
                coord_list.append((self._replace_dir(c[0], self.ref_direction), c[1]))
                coord_list.append(
                    (self._replace_dir(c[0], self.ref_direction), self._replace_dir(c[1], self.ref_direction)))
        IK = []
        for i,c in zip(I_coeff, coord_list):
            IK.append(i*self.I(*c))


        K = tf.add_n(IK)


        return K

class IntegrandKernel(object):
    def __init__(self, theta):
        self.theta = theta

    def apply(self, lamda):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [N,M,3]
        :return: list of tf.Tensor
            shapes [N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
        """
        raise NotImplementedError()

class RBF_(IntegrandKernel):
    """
    RBF(lamda) = sigma^2 exp(-0.5*lamda^2/lengthscale^2)
    = exp(2*log(sigma) - 0.5*lamda^2/lengthscale^2)
    """
    def __init__(self, theta):
        self.theta = theta
        self.sigma = theta[0]
        self.lengthscale = theta[1]

    def apply(self, lamda):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [N,M,3]
        :return: list of tf.Tensor
            shapes [N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
        """
        l2 = tf.math.square(self.lengthscale)
        lamda2 = lamda / l2
        #N,M
        chi2 = tf.reduce_sum(tf.math.square(lamda/self.lengthscale), axis=2)
        # N,M
        K = tf.math.exp(2.*tf.math.log(self.sigma) -0.5*chi2)
        # #N,M,3
        # dK = -lamda2*K[:,:,None]
        #N,M,3,3
        d2K = (lamda2[:,:,:,None]*lamda2[:,:,None,:] - tf.eye(3, dtype=float_type)/l2)*K[:,:,None,None]

        def dK(v):
            """
            Return product dK with v
            :param v: tf.Tensor
                [N,M,3]
            :return: tf.Tensor
                [N,M]
            """
            return -tf.reduce_sum(lamda2*v*K[:,:,None], axis=2)

        x = lamda[:, :, 0:1]
        y = lamda[:, :, 1:2]
        z = lamda[:, :, 2:3]

        x2 = tf.math.square(x)
        y2 = tf.math.square(y)
        z2 = tf.math.square(z)

        def d3K(v):
            """
            Return product d3K with v
            :param v: tf.Tensor
                [N,M,3]
            :return: tf.Tensor
                [N,M,3,3]
            """
            #N,M,1
            a = v[:,:,0:1]
            b = v[:,:,1:2]
            c = v[:,:,2:3]


            a1 = l2-x2
            aa1 = a*a1
            a2 = l2-y2
            ba2 = b*a2
            a3 = l2-z2
            xyz = x*y*z
            ax = a*x
            cz = c*z
            by = b*y

            M00 = ax*(3.*l2 - x2) + a1*(by+cz)
            M01 = aa1*y + x*ba2 - c*xyz
            M02 = aa1*z - b*xyz + c*x*a3
            M10 =  M01
            M11 = ax*a2 + by*(3.*l2 - y2) + cz*a2
            M12 = ba2*z - a*xyz
            M20 = M02
            M21 = M12
            M22 = ax*a3 + by*a3 + cz*(3.*l2 - z2)
            #N,M,3,3
            M = tf.concat([tf.stack([M00,M01,M02],axis=3),
                           tf.stack([M10, M11, M12], axis=3),
                           tf.stack([M20, M21, M22], axis=3)],
                          axis=2)*tf.math.reciprocal(tf.math.square(l2*self.lengthscale))
            return M*K[:,:,None,None]

        K_theta = K[:,:,None]*tf.stack([2.*tf.ones_like(K)*self.sigma, chi2/self.lengthscale], axis=2)
        def d2K_theta():
            #N,M,3,3
            d2K_sigma = 2.*tf.math.reciprocal(self.sigma)*d2K

            xyz2 = x2+y2+z2
            M00 = 2.*l2*l2 + x2*xyz2 - l2*(4.*x2 + xyz2)
            M01 = x*y*(-4.*l2 + xyz2)
            M02 = x*z*(-4.*l2 + xyz2)
            M10 = M01
            M11 = 2.*l2*l2 + y2*xyz2 - l2*(4.*y2 + xyz2)
            M12 = y*z*(-4.*l2 + xyz2)
            M20 = M02
            M21 = M12
            M22 = 2*l2*l2 + z2*xyz2 - l2*(4.*z2 + xyz2)
            d2K_lengthscale = tf.concat([tf.stack([M00, M01, M02], axis=3),
                                         tf.stack([M10, M11, M12], axis=3),
                                         tf.stack([M20, M21, M22], axis=3)],
                                        axis=2)*K[:,:,None,None]/l2/l2/l2/self.lengthscale
            #N,M,2,3,3
            return tf.stack([d2K_sigma, d2K_lengthscale], axis=2)

        return K, dK, d2K, d3K, K_theta, d2K_theta()

class RBF(IntegrandKernel):
    """
    RBF(lamda) = sigma^2 exp(-0.5*lamda^2/lengthscale^2)
    = exp(2*log(sigma) - 0.5*lamda^2/lengthscale^2)
    """
    def __init__(self, theta):
        self.theta = theta
        self.sigma = theta[0]
        self.lengthscale = theta[1]

    def apply(self, lamda):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [N,M,3]
        :return: list of tf.Tensor
            shapes [N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
        """
        l2 = tf.math.square(self.lengthscale)
        lamda2 = lamda / l2
        #N,M
        chi2 = tf.reduce_sum(tf.math.square(lamda/self.lengthscale), axis=2)
        # N,M
        K = tf.math.exp(2.*tf.math.log(self.sigma) -0.5*chi2)
        # #N,M,3
        # dK = -lamda2*K[:,:,None]
        #N,M,3,3
        d2K = (lamda2[:,:,:,None]*lamda2[:,:,None,:] - tf.eye(3, dtype=float_type)/l2)*K[:,:,None,None]
        K_theta = K[:, :, None] * tf.stack([2. * tf.ones_like(K) * self.sigma, chi2 / self.lengthscale], axis=2)
        @tf.custom_gradient
        def custom_K(theta):
            def grad(dK):
                return tf.reduce_sum(K_theta*dK[:,:,None], axis=[0,1])
            return K, grad
        return custom_K(self.theta), d2K


def test_rbf_taylor():
    with tf.Session(graph=tf.Graph()) as sess:
        N = 2
        x = tf.constant([[0., 0., t] for t in np.linspace(0., 60., N)], float_type)
        k = tf.constant([[0., np.sin(theta),np.cos(theta)] for theta in np.linspace(-4.*np.pi/180., 4.*np.pi/180., N)], float_type)
        k /= tf.linalg.norm(k, axis=1, keepdims=True)
        #
        X = tf.concat([k,x],axis=1)
        theta = tf.constant([1., 10.],float_type)
        a = tf.constant(200., float_type)
        b = tf.constant(100., float_type)
        mu = None
        ref_location = X[0,3:6]
        ref_direction = X[0,0:3]

        ref_kern = RandomKernel(RBF(theta), 2000, a, b, mu=mu, ref_location=ref_location, ref_direction=ref_direction,
                                obs_type='DTEC', ionosphere_type='flat')
        ref_K = ref_kern.K(X, X)

        ref_g = tf.gradients(ref_K,[theta])[0]
        ref_K = sess.run(ref_K)
        ref_g = sess.run(ref_g)
        F = []
        R = [4,5,6,7,8,9]
        import pylab as plt
        plt.imshow(ref_K)
        plt.colorbar()
        plt.show()
        for res in R:
            test_kern = TrapezoidKernel(RBF(theta), res, a, b, mu=mu, ref_location=ref_location,
                                     ref_direction=ref_direction,
                                     obs_type='DTEC', ionosphere_type='flat')
            from timeit import default_timer
            K = test_kern.K(X, X)
            g = tf.gradients(K,[theta])[0]
            t0 = default_timer()
            K = sess.run(K)
            print(K)
            print((default_timer()-t0))
            t0 = default_timer()
            g = sess.run(g)
            print((default_timer() - t0))
            plt.imshow(K)
            plt.colorbar()
            plt.show()
            print(ref_g,g, ref_g-g)

            f = np.mean(np.abs(ref_K - K))
            F.append(f)

        plt.plot(R, F)

        plt.show()

        # ref_kern = RandomKernel(RBF(theta), 2000, a, b, mu=mu, ref_location = ref_location, ref_direction = ref_direction, obs_type='DTEC', ionosphere_type='flat')
        # g = tf.gradients(tf.reduce_sum(ref_kern.K(X,X)), [theta])[0]
        # ref_g = sess.run(g)
        # F = []
        # R = [2,3,4,5]
        # for res in R:
        #     test_kern = TaylorKernel(RBF(theta), res, a, b, mu=mu, ref_location=ref_location, ref_direction=ref_direction,
        #                  obs_type='DTEC', ionosphere_type='flat')
        #     g = tf.gradients(tf.reduce_sum(test_kern.K(X, X)), [theta])[0]
        #
        #     f = np.mean(np.abs(ref_g - sess.run(g)))
        #     F.append(f)
        # import pylab as plt
        # plt.plot(R,F)
        #
        # plt.show()

class TaylorKernel(RayKernel):
    def __init__(self, integrand_kernel:IntegrandKernel, partitions, *args, **kwargs):
        super(TaylorKernel, self).__init__(*args, **kwargs)
        self.integrand_kernel = integrand_kernel
        if isinstance(partitions, (float,int)):
            partitions = np.linspace(0., 1., int(partitions) + 1)
            partitions = list(np.stack([partitions[:-1], partitions[1:]], axis=1))
        self.regions = list(itertools.product(partitions, partitions))

    def I(self, X1, X2):

        k1, x1 = X1[:,0:3], X1[:, 3:6]
        l1, m1, (dl1_da, dl1_db), (dm1_da,dm1_db) = self.calculate_ray_endpoints(x1, k1)
        if X2 is None:
            l2, m2, (dl2_da, dl2_db), (dm2_da,dm2_db) = l1, m1, (dl1_da, dl1_db), (dm1_da,dm1_db)
        else:
            k2, x2 = X2[:, 0:3], X2[:, 3:6]
            l2, m2, (dl2_da, dl2_db), (dm2_da,dm2_db) = self.calculate_ray_endpoints(x2, k2)
        # N,M,3
        L12 = (l1[:, None, :] - l2[None, :, :])
        # N,M,3
        dL12_da = (dl1_da[:, None, :] - dl2_da[None, :, :])
        # N,M,3
        dL12_db = (dl1_db[:, None, :] - dl2_db[None, :, :])

        IK_subregions = []

        # lamda = L12 + s1*m1 - s2*m2
        # dlamda_da = dL12_da + s1*dm1_da - s2*dm2_da
        # dlamda_db = dL12_db + s1*dm1_db - s2*dm2_db

        for interval1, interval2 in self.regions:
            s1_mean = np.mean(interval1)
            s2_mean = np.mean(interval2)
            D1 = interval1[1] - interval1[0]
            D2 = interval2[1] - interval2[0]
            #N,M,3
            lamda = L12 + (s1_mean * m1[:, None, :] - s2_mean * m2[None, :, :])

            # N, M, 3, 3
            _dot = 1. / 24. * (D1 ** 3 * D2 * m1[:, None, :, None] * m1[None, :, None, :]
                               + D2 ** 3 * D1 * m2[:, None, :,None] * m2[None, :,None, :])

            #[N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
            K, d2K = self.integrand_kernel.apply(lamda)

            #N, M
            IK = D1*D2*K + tf.reduce_sum(d2K* _dot, axis=[2,3])

            IK_subregions.append(IK)

        return tf.add_n(IK_subregions)

# class TaylorKernel(RayKernel):
#     def __init__(self, integrand_kernel:IntegrandKernel, partitions, *args, **kwargs):
#         super(TaylorKernel, self).__init__(*args, **kwargs)
#         self.integrand_kernel = integrand_kernel
#         if isinstance(partitions, (float,int)):
#             partitions = np.linspace(0., 1., int(partitions) + 1)
#             partitions = list(np.stack([partitions[:-1], partitions[1:]], axis=1))
#         self.regions = list(itertools.product(partitions, partitions))
#
#     def I(self, X1, X2):
#
#         k1, x1 = X1[:,0:3], X1[:, 3:6]
#         l1, m1, (dl1_da, dl1_db), (dm1_da,dm1_db) = self.calculate_ray_endpoints(x1, k1)
#         if X2 is None:
#             l2, m2, (dl2_da, dl2_db), (dm2_da,dm2_db) = l1, m1, (dl1_da, dl1_db), (dm1_da,dm1_db)
#         else:
#             k2, x2 = X2[:, 0:3], X2[:, 3:6]
#             l2, m2, (dl2_da, dl2_db), (dm2_da,dm2_db) = self.calculate_ray_endpoints(x2, k2)
#         # N,M,3
#         L12 = (l1[:, None, :] - l2[None, :, :])
#         # N,M,3
#         dL12_da = (dl1_da[:, None, :] - dl2_da[None, :, :])
#         # N,M,3
#         dL12_db = (dl1_db[:, None, :] - dl2_db[None, :, :])
#
#         IK_subregions, IdKda_subregions, IdKdb_subregions, IdKdtheta_subregions = [], [], [], []
#
#         # lamda = L12 + s1*m1 - s2*m2
#         # dlamda_da = dL12_da + s1*dm1_da - s2*dm2_da
#         # dlamda_db = dL12_db + s1*dm1_db - s2*dm2_db
#
#         for interval1, interval2 in self.regions:
#             s1_mean = np.mean(interval1)
#             s2_mean = np.mean(interval2)
#             D1 = interval1[1] - interval1[0]
#             D2 = interval2[1] - interval2[0]
#             #N,M,3
#             lamda = L12 + (s1_mean * m1[:, None, :] - s2_mean * m2[None, :, :])
#             # N,M,3
#             dlamda_da = dL12_da + (s1_mean * dm1_da[:, None, :] - s2_mean * dm2_da[None, :, :])
#             # N,M,3
#             dlamda_db = dL12_db + (s1_mean * dm1_db[:, None, :] - s2_mean * dm2_db[None, :, :])
#
#             # N, M, 3, 3
#             _dot = 1. / 24. * (D1 ** 3 * D2 * m1[:, None, :, None] * m1[None, :, None, :]
#                                + D2 ** 3 * D1 * m2[:, None, :,None] * m2[None, :,None, :])
#
#             #[N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
#             K, dK, d2K, d3K, K_theta, d2K_theta = self.integrand_kernel.apply(lamda)
#
#             #N, M
#             dKda = dK(dlamda_da)
#             #tf.reduce_sum(dK*dlamda_da, axis=2)
#             #N,M,3,3
#             d2dKda = d3K(dlamda_da)
#             #tf.reduce_sum(d3K * dlamda_da[:, :, :, None, None], axis=2)
#
#             # N, M
#             dKdb = dK(dlamda_db)
#             #tf.reduce_sum(dK * dlamda_db, axis=2)
#             # N,M,3,3
#             d2dKdb = d3K(dlamda_db)
#             #tf.reduce_sum(d3K * dlamda_db[:, :, :, None, None], axis=2)
#
#             #N, M
#             IK = D1*D2*K + tf.reduce_sum(d2K* _dot, axis=[2,3])
#             #N,M
#             IdKda = D1*D2*dKda + tf.reduce_sum(d2dKda* _dot, axis=[2,3])
#             #N,M
#             IdKdb = D1 * D2 * dKdb + tf.reduce_sum(d2dKdb* _dot, axis=[2,3])
#             #N,M,T,3,3
#             _dot = tf.tile(_dot[:, :, None, :, :], (1,1,tf.shape(K_theta)[2],1,1))
#             #N,M,T
#             IK_theta = D1 * D2 * K_theta + tf.reduce_sum(d2K_theta* _dot, axis=[3,4])
#             IK_subregions.append(IK)
#             IdKda_subregions.append(IdKda)
#             IdKdb_subregions.append(IdKdb)
#             IdKdtheta_subregions.append(IK_theta)
#         return tf.add_n(IK_subregions), tf.add_n(IdKda_subregions), tf.add_n(IdKdb_subregions), tf.add_n(IdKdtheta_subregions)
#
#     def _replace_ant(self, X, x):
#         x_tile = tf.tile(x[None,:], (tf.shape(X)[0], 1))
#         return tf.concat([X[:,0:3], x_tile], axis=1)
#
#     def _replace_dir(self, X, x):
#         x_tile = tf.tile(x[None,:], (tf.shape(X)[0], 1))
#         return tf.concat([x_tile, X[:,3:6]], axis=1)
#
#     def K(self, X1, X2):
#         coord_list = None
#         I_coeff = None
#         if self.obs_type in ['TEC', 'DTEC', 'DDTEC']:
#             coord_list = [(X1, X2)]
#             I_coeff = [1.]
#         if self.obs_type == ['DTEC', 'DDTEC']:
#             coord_list_prior = coord_list
#             I_coeff_prior = I_coeff
#             I_coeff = []
#             coord_list = []
#             for i in I_coeff_prior:
#                 I_coeff.append(i)
#                 I_coeff.append(-i)
#                 I_coeff.append(-i)
#                 I_coeff.append(i)
#             for c in coord_list_prior:
#                 coord_list.append(c)
#                 coord_list.append((c[0], self._replace_ant(c[1], self.ref_location)))
#                 coord_list.append((self._replace_ant(c[0], self.ref_location), c[1]))
#                 coord_list.append((self._replace_ant(c[0], self.ref_location), self._replace_ant(c[1], self.ref_location)))
#         if self.obs_type in ['DDTEC']:
#             coord_list_prior = coord_list
#             I_coeff_prior = I_coeff
#             I_coeff = []
#             coord_list = []
#             for i in I_coeff_prior:
#                 I_coeff.append(i)
#                 I_coeff.append(-i)
#                 I_coeff.append(-i)
#                 I_coeff.append(i)
#             for c in coord_list_prior:
#                 coord_list.append(c)
#                 coord_list.append((c[0], self._replace_dir(c[1], self.ref_direction)))
#                 coord_list.append((self._replace_dir(c[0], self.ref_direction), c[1]))
#                 coord_list.append(
#                     (self._replace_dir(c[0], self.ref_direction), self._replace_dir(c[1], self.ref_direction)))
#         IK, IdKda, IdKdb, IdKdtheta = [],[],[],[]
#         for i,c in zip(I_coeff, coord_list):
#             IK_subregions, IdKda_subregions, IdKdb_subregions, IdKdtheta_subregions = [i*v for v in self.I(*c)]
#             IK.append(IK_subregions)
#             IdKda.append(IdKda_subregions)
#             IdKdb.append(IdKdb_subregions)
#             IdKdtheta.append(IdKdtheta_subregions)
#
#         K = tf.add_n(IK)
#         dKda = tf.add_n(IdKda)
#         dKdb = tf.add_n(IdKdb)
#         dKdtheta = tf.add_n(IdKdtheta)
#
#         @tf.custom_gradient
#         def custom_K(a, b, theta):
#             def grad(dK):
#                 return tf.reduce_sum(dKda*dK, axis=[0,1]), tf.reduce_sum(dKdb*dK, axis=[0,1]), tf.reduce_sum(dKdtheta*dK[:,:,None], axis=[0,1])
#             return K, grad
#
#         return custom_K(self.a, self.b, self.integrand_kernel.theta)

class RandomKernel(RayKernel):
    def __init__(self, integrand_kernel:IntegrandKernel, resolution, *args, **kwargs):
        super(RandomKernel, self).__init__(*args, **kwargs)
        self.integrand_kernel = integrand_kernel
        self.resolution = resolution

    def I(self, X1, X2):

        k1, x1 = X1[:,0:3], X1[:, 3:6]
        l1, m1, (dl1_da, dl1_db), (dm1_da,dm1_db) = self.calculate_ray_endpoints(x1, k1)
        if X2 is None:
            l2, m2, (dl2_da, dl2_db), (dm2_da,dm2_db) = l1, m1, (dl1_da, dl1_db), (dm1_da,dm1_db)
        else:
            k2, x2 = X2[:, 0:3], X2[:, 3:6]
            l2, m2, (dl2_da, dl2_db), (dm2_da,dm2_db) = self.calculate_ray_endpoints(x2, k2)
        N = tf.shape(k1)[0]
        M = tf.shape(k2)[0]
        # N,M,3
        L12 = (l1[:, None, :] - l2[None, :, :])

        s1 = tf.random.uniform((self.resolution,), dtype=float_type)
        s2 = tf.random.uniform((self.resolution,), dtype=float_type)
        lamda = L12[None, :,:,:] + s1[:, None,None,None] * m1[None,:,None,:] - s2[:,None,None,None] * m2[None,None,:,:]

        # I = tf.scan(lambda accumulated, lamda: accumulated + self.integrand_kernel.apply(lamda)[0], initializer=tf.zeros([N,M], float_type), elems=lamda)
        # return I/self.resolution
        I = tf.map_fn(lambda lamda: self.integrand_kernel.apply(lamda)[0], lamda)
        return tf.reduce_mean(I,axis=0)


class TrapezoidKernel(RayKernel):
    """
    The DTEC kernel is derived from first principles by assuming a GRF over the electron density, from which DTEC kernel
    can be caluclated as,

    K(ray_i, ray_j) =     I(a_i, k_i, t_i, a_j, k_j, t_j)  + I(a0_i, k_i, t_i, a0_j, k_j, t_j)
                        - I(a0_i, k_i, t_i, a_j, k_j, t_j) - I(a_i, k_i, t_i, a0_j, k_j, t_j)

    where,
                I(a,b,c,d,e,g) = iint [K_ne(y(a,b,c), y(d,e,f))](s1,s2) ds1 ds2
    """

    def __init__(self, integrand_kernel:IntegrandKernel, resolution, *args, **kwargs):
        super(TrapezoidKernel, self).__init__(*args, **kwargs)
        self.integrand_kernel = integrand_kernel

        self.resolution = resolution

    def I(self, X1, X2):
        """
        Calculate the ((D)D)TEC kernel based on the FED kernel.

        :param X: float_type, tf.Tensor (N, 7[10[13]])
            Coordinates in order (time, kx, ky, kz, x,y,z, [x0, y0, z0, [kx0, ky0, kz0]])
        :param X2:
            Second coordinates, if None then equal to X
        :return:
        """

        k1, x1 = X1[:, 0:3], X1[:, 3:6]
        l1, m1, (dl1_da, dl1_db), (dm1_da, dm1_db) = self.calculate_ray_endpoints(x1, k1)
        if X2 is None:
            l2, m2, (dl2_da, dl2_db), (dm2_da, dm2_db) = l1, m1, (dl1_da, dl1_db), (dm1_da, dm1_db)
        else:
            k2, x2 = X2[:, 0:3], X2[:, 3:6]
            l2, m2, (dl2_da, dl2_db), (dm2_da, dm2_db) = self.calculate_ray_endpoints(x2, k2)
        N = tf.shape(k1)[0]
        M = tf.shape(k2)[0]
        # N,M,3
        L12 = (l1[:, None, :] - l2[None, :, :])

        s = tf.cast(tf.linspace(0., 1., self.resolution + 1), dtype=float_type)
        ds = tf.math.reciprocal(tf.cast(self.resolution,float_type))

        # res, res, N, M, 3
        lamda = L12[None, None, :, :, :] + s[:, None, None, None, None] * m1[None, None, :, None, :] - s[None,  :, None, None, None] * m2[None,None, None,:,:]
        # res*res, N,M,3
        lamda = tf.reshape(lamda,(-1, N,M,3))
        # I = tf.scan(lambda accumulated, lamda: accumulated + self.integrand_kernel.apply(lamda)[0], initializer=tf.zeros([N,M], float_type), elems=lamda)
        # return I/self.resolution
        # res*res, N,M
        I = tf.map_fn(lambda lamda: self.integrand_kernel.apply(lamda)[0], lamda)
        # res,res, N,M
        I = tf.reshape(I, (self.resolution+1, self.resolution+1, N,M))

        # N,M
        I = 0.25 * tf.math.square(ds) * tf.add_n([I[ 0, 0, :, :],
                                      I[ -1, 0, :, :],
                                      I[ 0, -1, :, :],
                                      I[ -1, -1, :, :],
                                      2 * tf.reduce_sum(I[ -1, :, :, :], axis=[0]),
                                      2 * tf.reduce_sum(I[ 0, :, :, :], axis=[0]),
                                      2 * tf.reduce_sum(I[ :, -1, :, :], axis=[0]),
                                      2 * tf.reduce_sum(I[ :, 0, :, :], axis=[0]),
                                      4 * tf.reduce_sum(I[ 1:-1, 1:-1,: , :], axis=[0,1])])
        return I
