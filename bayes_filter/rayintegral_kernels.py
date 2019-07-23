import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

import tensorflow as tf
import numpy as np
from . import logging, float_type
from collections import OrderedDict
import itertools
from .misc import flatten_batch_dims


def _validate_arg_if_not_none(arg, assertion, validate_args):
    if arg is None:
        return arg
    with tf.control_dependencies([assertion(arg)] if validate_args else []):
        result = tf.identity(arg)
    return result


class RayKernel(object):
    def __init__(self, a, b, mu=None, ref_location=None, ref_direction=None, obs_type='DTEC', ionosphere_type='flat'):
        if ionosphere_type == 'curved':
            if mu is None:
                raise ValueError("Need a mu for curved.")
        self.a = a
        self.b = b
        self.mu = mu
        self.obs_type = obs_type
        self.ref_location = ref_location if ref_location is not None else tf.constant([0., 0., 0.], float_type)
        self.ref_direction = ref_direction if ref_direction is not None else tf.constant([0., 0., 1.], float_type)
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
        with tf.name_scope('calculate_ray_endpoints', values=[x, k]):
            if self.ionosphere_type == 'flat':
                # N
                sec = tf.math.reciprocal(k[:, 2], name='secphi')

                # N
                bsec = sec * self.b

                # N
                sm = sec * (self.a + self.ref_location[2] - x[:, 2]) - 0.5 * bsec
                # N
                l = x + k * sm[:, None]
                m = k * bsec[:, None]
                return l, m

        raise NotImplementedError("curved not implemented")

    def _replace_ant(self, X, x):
        x_tile = tf.tile(x[None, :], (tf.shape(X)[0], 1))
        return tf.concat([X[:, 0:3], x_tile], axis=1)

    def _replace_dir(self, X, x):
        x_tile = tf.tile(x[None, :], (tf.shape(X)[0], 1))
        return tf.concat([x_tile, X[:, 3:6]], axis=1)

    def K(self, X1, X2):
        with tf.name_scope('RayKernel_K'):
            coord_list = None
            I_coeff = None
            if self.obs_type in ['TEC', 'DTEC', 'DDTEC']:
                coord_list = [(X1, X2)]
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
                    coord_list.append((c[0], self._replace_ant(c[1], self.ref_location)))
                    coord_list.append((self._replace_ant(c[0], self.ref_location), c[1]))
                    coord_list.append(
                        (self._replace_ant(c[0], self.ref_location), self._replace_ant(c[1], self.ref_location)))
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
            for i, c in zip(I_coeff, coord_list):
                IK.append(i * self.I(*c))

            K = tf.add_n(IK)

            return K


class IntegrandKernel(object):
    def __init__(self, theta, has_d2K=False):
        self.has_d2K = has_d2K
        self.theta = theta

    def _apply(self, lamda, return_d2K=False):
        """
        Applies the kernel to already differenced coordinates.
        Assumes the integrand kernel is stationary, and therefore is only a function of
        lamda.
        :param lamda: tf.Tensor
            differenced coordinates [b0,...,bB, 3]
        :return: tf.Tensor
            The covariance function applied to lamda [b0, ..., bB]
        """
        raise NotImplementedError()

    def __add__(self, other):
        return SumIntegrand([self, other])

    def __mul__(self, other):
        return ProductIntegrand([self, other])

    def apply(self, lamda, return_d2K=False):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [N,M,3]
        :return: list of tf.Tensor
            shapes [N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
        """
        with tf.name_scope("IntegrandKernel_apply", values=[lamda]):
            if return_d2K:
                if not self.has_d2K:
                    raise ValueError("Does not have d2K")
                return self._apply(lamda, True)
            return self._apply(lamda, False)
            # if self.has_d2K:
            #     return self._apply(lamda)
            # lamda_flat = flatten_batch_dims(lamda, -1)
            # K_flat = self._apply(lamda_flat)
            # if isinstance(K_flat, (tuple, list)):
            #     K_flat = K_flat[0]
            # print(K_flat)
            # J = batch_jacobian(K_flat[:,None], lamda_flat)
            # print(J)
            # H = batch_jacobian(J, lamda_flat)
            # print(H)
            # d2K = tf.reshape(H, tf.concat([tf.shape(lamda)] + [tf.shape(lamda)[-1:]], axis=0))
            # K = tf.reshape(K_flat, tf.shape(lamda)[:-1])
            # return K, d2K


class SumIntegrand(IntegrandKernel):
    def __init__(self, kernels):
        has_d2K = np.all([k.has_d2K for k in kernels])
        self.kernels = kernels
        super(SumIntegrand, self).__init__(None, has_d2K=has_d2K)

    def _apply(self, lamda, return_d2K=False):
        if return_d2K:
            res = [k._apply(lamda, return_d2K=False) for k in self.kernels]
            return np.sum([r[0] for r in res]), np.sum([r[1] for r in res])
        return np.sum([k._apply(lamda, return_d2K=False) for k in self.kernels])


class ProductIntegrand(IntegrandKernel):
    def __init__(self, kernels):
        has_d2K = np.all([k.has_d2K for k in kernels])
        self.kernels = kernels
        super(ProductIntegrand, self).__init__(None, has_d2K=has_d2K)

    def _apply(self, lamda, return_d2K=False):
        if return_d2K:
            res = [k._apply(lamda, return_d2K=False) for k in self.kernels]
            return np.prod([r[0] for r in res]), np.prod([r[1] for r in res])
        return np.prod([k._apply(lamda, return_d2K=False) for k in self.kernels])


class RBF(IntegrandKernel):
    """
    RBF(lamda) = sigma^2 exp(-0.5*lamda^2/lengthscale^2)
    = exp(2*log(sigma) - 0.5*lamda^2/lengthscale^2)
    """

    def __init__(self, theta):
        super(RBF, self).__init__(theta, has_d2K=True)
        self.sigma = theta[0]
        self.lengthscale = theta[1]

    def _apply(self, lamda, return_d2K=False):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [b0, ..., bB,3]
        :return: list of tf.Tensor
            shapes [b0, ..., bB], [b0, ..., bB,3]
        """
        with tf.name_scope('RBF_apply', values=[lamda]):
            l2 = tf.math.square(self.lengthscale)
            lamda2 = lamda / l2
            # N,M
            chi2 = tf.reduce_sum(tf.math.square(lamda / self.lengthscale), axis=-1)
            # N,M
            K = tf.math.exp(2. * tf.math.log(self.sigma) - 0.5 * chi2)
            K_theta = K[..., None] * tf.stack([2. * tf.ones_like(K) * self.sigma, chi2 / self.lengthscale], axis=-1)

            @tf.custom_gradient
            def custom_K(theta):
                def grad(dK):
                    return tf.reduce_sum(K_theta * dK[..., None], axis=list(range(len(K.shape))))

                return K, grad

            if not return_d2K:
                return custom_K(self.theta)
            # N,M,3,3
            d2K = (lamda2[..., None] * lamda2[..., None, :] - tf.eye(3, dtype=float_type) / l2) * K[..., None, None]
            # TODO: jacobian and custom gradient doesn't seem to want to work.
            return K, d2K
            # return custom_K(self.theta), d2K


class M52(IntegrandKernel):
    def __init__(self, theta):
        super(M52, self).__init__(theta, has_d2K=False)
        self.sigma = theta[0]
        self.lengthscale = theta[1]

    def _apply(self, lamda, return_d2K=False):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [N,M,3]
        :return: list of tf.Tensor
            shapes [N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
        """
        if return_d2K:
            raise ValueError("No d2K defined for M52")

        with tf.name_scope('M52_apply', values=[lamda]):
            norm = util.sqrt_with_finite_grads(tf.reduce_sum(tf.math.square(lamda / self.lengthscale), axis=-1))
            series_term = np.sqrt(5) * norm
            log_result = tf.math.log1p(series_term + series_term ** 2 / 3.) - series_term

            if self.sigma is not None:
                log_result += 2. * tf.math.log(self.sigma)
            return tf.math.exp(log_result)


class M32(IntegrandKernel):
    def __init__(self, theta):
        super(M32, self).__init__(theta, has_d2K=False)
        self.sigma = theta[0]
        self.lengthscale = theta[1]

    def _apply(self, lamda, return_d2K=False):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [N,M,3]
        :return: list of tf.Tensor
            shapes [N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
        """
        if return_d2K:
            raise ValueError("No d2K defined for M32")

        with tf.name_scope('M32_apply', values=[lamda]):
            norm = util.sqrt_with_finite_grads(tf.reduce_sum(tf.math.square(lamda / self.lengthscale), axis=-1))
            series_term = np.sqrt(3) * norm
            log_result = tf.math.log1p(series_term) - series_term
            if self.sigma is not None:
                log_result += 2. * tf.math.log(self.sigma)
            return tf.math.exp(log_result)


class M12(IntegrandKernel):
    def __init__(self, theta):
        super(M12, self).__init__(theta, has_d2K=False)
        self.sigma = theta[0]
        self.lengthscale = theta[1]

    def _apply(self, lamda, return_d2K=False):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [N,M,3]
        :return: list of tf.Tensor
            shapes [N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
        """
        if return_d2K:
            raise ValueError("No d2K defined for M12")

        with tf.name_scope('M12_apply', values=[lamda]):
            norm = util.sqrt_with_finite_grads(tf.reduce_sum(tf.math.square(lamda / self.lengthscale), axis=-1))
            log_result = -norm
            if self.sigma is not None:
                log_result += 2. * tf.math.log(self.sigma)
            return tf.math.exp(log_result)


class RQ(IntegrandKernel):
    """
        turbulence goes like r^-2/3 -> scale_mixture_rate = 1/3.
        ```
          k(x, y) = amplitude**2 * (1. + ||x - y|| ** 2 / (2 * scale_mixture_rate * length_scale**2)) ** -scale_mixture_rate
        ```
    """

    def __init__(self, theta):
        super(RQ, self).__init__(theta, has_d2K=False)
        self.sigma = theta[0]
        self.lengthscale = theta[1]
        self.scale_mixture_rate = theta[2]

    def _apply(self, lamda, return_d2K=False):
        """
        Calculate K, dK, d2K, d3K, K_theta, d2K_theta

        :param lamda: tf.Tensor
            Coordinates [N,M,3]
        :return: list of tf.Tensor
            shapes [N, M], [N,M,3], [N,M,3,3], [N,M,3,3,3], [N,M,T], [N,M,T,3,3]
        """
        if return_d2K:
            raise ValueError("No d2K defined for RQ")

        with tf.name_scope('RQ_apply', values=[lamda]):
            norm = tf.reduce_sum(tf.math.square(lamda), axis=-1)

            norm /= 2.

            if self.lengthscale is not None:
                norm /= self.lengthscale ** 2

            if self.scale_mixture_rate is None:
                power = 1.
            else:
                power = self.scale_mixture_rate
                norm /= power

            result = (1. + norm) ** -power

            return self.sigma ** 2 * result


def test_rbf_taylor():
    with tf.Session(graph=tf.Graph()) as sess:
        N = 2
        x = tf.constant([[0., 0., t] for t in np.linspace(0., 60., N)], float_type)
        k = tf.constant(
            [[0., np.sin(theta), np.cos(theta)] for theta in np.linspace(-4. * np.pi / 180., 4. * np.pi / 180., N)],
            float_type)
        k /= tf.linalg.norm(k, axis=1, keepdims=True)
        #
        X = tf.concat([k, x], axis=1)
        theta = tf.constant([1., 10.], float_type)
        a = tf.constant(200., float_type)
        b = tf.constant(100., float_type)
        mu = None
        ref_location = X[0, 3:6]
        ref_direction = X[0, 0:3]

        ref_kern = RandomKernel(M52(theta), 2000, a, b, mu=mu, ref_location=ref_location, ref_direction=ref_direction,
                                obs_type='DTEC', ionosphere_type='flat')
        ref_K = ref_kern.K(X, X)

        ref_g = tf.gradients(ref_K, [theta])[0]
        ref_K = sess.run(ref_K)
        ref_g = sess.run(ref_g)
        F = []
        R = [4, 5, 6, 7, 8, 9]
        import pylab as plt
        plt.imshow(ref_K)
        plt.colorbar()
        plt.show()
        for res in R:
            test_kern = TrapezoidKernel(M52(theta), res, a, b, mu=mu, ref_location=ref_location,
                                        ref_direction=ref_direction,
                                        obs_type='DTEC', ionosphere_type='flat')
            from timeit import default_timer
            K = test_kern.K(X, X)
            g = tf.gradients(K, [theta])[0]
            t0 = default_timer()
            K = sess.run(K)
            print(K)
            print((default_timer() - t0))
            t0 = default_timer()
            g = sess.run(g)
            print((default_timer() - t0))
            plt.imshow(K)
            plt.colorbar()
            plt.show()
            print(ref_g, g, ref_g - g)

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
    def __init__(self, integrand_kernel: IntegrandKernel, partitions, a, b, mu=None, ref_location=None,
                 ref_direction=None, obs_type='DTEC', ionosphere_type='flat'):
        super(TaylorKernel, self).__init__(a, b, mu=mu, ref_location=ref_location, ref_direction=ref_direction,
                                           obs_type=obs_type, ionosphere_type=ionosphere_type)
        self.integrand_kernel = integrand_kernel
        if isinstance(partitions, (float, int)):
            partitions = np.linspace(0., 1., int(partitions) + 1)
            partitions = list(np.stack([partitions[:-1], partitions[1:]], axis=1))
        self.regions = list(itertools.product(partitions, partitions))

    def I(self, X1, X2):
        with tf.name_scope("TaylorKernel_I"):
            k1, x1 = X1[:, 0:3], X1[:, 3:6]
            l1, m1 = self.calculate_ray_endpoints(x1, k1)
            if X2 is None:
                l2, m2 = l1, m1
            else:
                k2, x2 = X2[:, 0:3], X2[:, 3:6]
                l2, m2 = self.calculate_ray_endpoints(x2, k2)
            # N,M,3
            L12 = (l1[:, None, :] - l2[None, :, :])

            IK_subregions = []

            # lamda = L12 + s1*m1 - s2*m2
            # dlamda_da = dL12_da + s1*dm1_da - s2*dm2_da
            # dlamda_db = dL12_db + s1*dm1_db - s2*dm2_db

            for interval1, interval2 in self.regions:
                s1_mean = np.mean(interval1)
                s2_mean = np.mean(interval2)
                D1 = interval1[1] - interval1[0]
                D2 = interval2[1] - interval2[0]
                # N,M,3
                lamda = L12 + (s1_mean * m1[:, None, :] - s2_mean * m2[None, :, :])

                # N, M, 3, 3
                _dot = 1. / 24. * (D1 ** 3 * D2 * m1[:, None, :, None] * m1[None, :, None, :]
                                   + D2 ** 3 * D1 * m2[:, None, :, None] * m2[None, :, None, :])

                # [N, M]
                K, d2K = self.integrand_kernel.apply(lamda, return_d2K=True)

                # N, M
                IK = D1 * D2 * K + tf.reduce_sum(d2K * _dot, axis=[2, 3])

                IK_subregions.append(IK)

            return tf.add_n(IK_subregions)


class RandomKernel(RayKernel):
    def __init__(self, integrand_kernel: IntegrandKernel, resolution, a, b, mu=None, ref_location=None,
                 ref_direction=None, obs_type='DTEC', ionosphere_type='flat'):
        super(RandomKernel, self).__init__(a, b, mu=mu, ref_location=ref_location, ref_direction=ref_direction,
                                           obs_type=obs_type, ionosphere_type=ionosphere_type)
        self.integrand_kernel = integrand_kernel
        self.resolution = resolution

    def I(self, X1, X2):
        with tf.name_scope("RandomKernel_I"):
            k1, x1 = X1[:, 0:3], X1[:, 3:6]
            l1, m1 = self.calculate_ray_endpoints(x1, k1)
            if X2 is None:
                l2, m2 = l1, m1
            else:
                k2, x2 = X2[:, 0:3], X2[:, 3:6]
                l2, m2 = self.calculate_ray_endpoints(x2, k2)
            # N,M,3
            L12 = (l1[:, None, :] - l2[None, :, :])

            s1 = tf.random.uniform((self.resolution,), dtype=float_type)
            s2 = tf.random.uniform((self.resolution,), dtype=float_type)
            lamda = L12[None, :, :, :] + s1[:, None, None, None] * m1[None, :, None, :] - s2[:, None, None, None] * m2[
                                                                                                                    None,
                                                                                                                    None,
                                                                                                                    :,
                                                                                                                    :]

            # I = tf.scan(lambda accumulated, lamda: accumulated + self.integrand_kernel.apply(lamda)[0], initializer=tf.zeros([N,M], float_type), elems=lamda)
            # return I/self.resolution
            I = tf.map_fn(lambda lamda: self.integrand_kernel.apply(lamda, return_d2K=False), lamda)
            return tf.reduce_mean(I, axis=0)


class TrapezoidKernel(RayKernel):
    """
    The DTEC kernel is derived from first principles by assuming a GRF over the electron density, from which DTEC kernel
    can be caluclated as,

    K(ray_i, ray_j) =     I(a_i, k_i, t_i, a_j, k_j, t_j)  + I(a0_i, k_i, t_i, a0_j, k_j, t_j)
                        - I(a0_i, k_i, t_i, a_j, k_j, t_j) - I(a_i, k_i, t_i, a0_j, k_j, t_j)

    where,
                I(a,b,c,d,e,g) = iint [K_ne(y(a,b,c), y(d,e,f))](s1,s2) ds1 ds2
    """

    def __init__(self, integrand_kernel: IntegrandKernel, resolution, a, b, mu=None, ref_location=None,
                 ref_direction=None, obs_type='DTEC', ionosphere_type='flat', use_map_fn=True):
        super(TrapezoidKernel, self).__init__(a, b, mu=mu, ref_location=ref_location, ref_direction=ref_direction,
                                              obs_type=obs_type, ionosphere_type=ionosphere_type)
        self.integrand_kernel = integrand_kernel
        self.resolution = resolution
        self.use_map_fn = use_map_fn

    def I(self, X1, X2):
        """
        Calculate the ((D)D)TEC kernel based on the FED kernel.

        :param X: float_type, tf.Tensor (N, 7[10[13]])
            Coordinates in order (time, kx, ky, kz, x,y,z, [x0, y0, z0, [kx0, ky0, kz0]])
        :param X2:
            Second coordinates, if None then equal to X
        :return:
        """

        with tf.name_scope('TrapezoidKernel_I'):

            k1, x1 = X1[:, 0:3], X1[:, 3:6]
            l1, m1 = self.calculate_ray_endpoints(x1, k1)
            if X2 is None:
                l2, m2 = l1, m1
            else:
                k2, x2 = X2[:, 0:3], X2[:, 3:6]
                l2, m2 = self.calculate_ray_endpoints(x2, k2)
            N = tf.shape(k1)[0]
            M = tf.shape(k2)[0]
            # N,M,3
            L12 = (l1[:, None, :] - l2[None, :, :])

            s = tf.cast(tf.linspace(0., 1., self.resolution + 1), dtype=float_type)
            ds = tf.math.reciprocal(tf.cast(self.resolution, float_type))

            # res, res, N, M, 3
            lamda = L12 + s[:, None, None, None, None] * m1[:, None, :] - s[None, :, None, None, None] * m2[None, :, :]

            if self.use_map_fn:
                # res*res, N,M,3
                lamda = tf.reshape(lamda, (-1, N, M, 3))
                # res*res, N,M
                I = tf.map_fn(lambda lamda: self.integrand_kernel.apply(lamda, return_d2K=False), lamda)
                # res,res, N,M
                I = tf.reshape(I, (self.resolution + 1, self.resolution + 1, N, M))
            else:
                I = self.integrand_kernel.apply(lamda, return_d2K=False)

            # N,M
            I = 0.25 * tf.math.square(ds) * tf.add_n([I[0, 0, :, :],
                                                      I[-1, 0, :, :],
                                                      I[0, -1, :, :],
                                                      I[-1, -1, :, :],
                                                      2 * tf.reduce_sum(I[-1, :, :, :], axis=[0]),
                                                      2 * tf.reduce_sum(I[0, :, :, :], axis=[0]),
                                                      2 * tf.reduce_sum(I[:, -1, :, :], axis=[0]),
                                                      2 * tf.reduce_sum(I[:, 0, :, :], axis=[0]),
                                                      4 * tf.reduce_sum(I[1:-1, 1:-1, :, :], axis=[0, 1])])
            return I


class Histogram(tfp.positive_semidefinite_kernels.PositiveSemidefiniteKernel):
    def __init__(self, heights, edgescales=None, lengthscales=None, feature_ndims=1, validate_args=False,
                 name='Histogram'):
        """Construct an Histogram kernel instance.
        Gives the histogram kernel on the isotropic distance from a point.

        Args:
        heights: floating point `Tensor` heights of spectum histogram.
            Must be broadcastable with `edgescales` and inputs to
            `apply` and `matrix` methods.
        edgescales: floating point `Tensor` that controls how wide the
            spectrum bins are. These are lengthscales, and edges are actually 1/``edgescales``.
            Must be broadcastable with
            `heights` and inputs to `apply` and `matrix` methods.
        lengthscales: floating point `Tensor` that controls how wide the
            spectrum bins are. The edges are actually 1/``lengthscales``.
            Must be broadcastable with
            `heights` and inputs to `apply` and `matrix` methods.
        feature_ndims: Python `int` number of rightmost dims to include in the
            squared difference norm in the exponential.
        validate_args: If `True`, parameters are checked for validity despite
            possibly degrading runtime performance
        name: Python `str` name prefixed to Ops created by this class.
        """
        with tf.name_scope(name, values=[heights, edgescales]) as name:
            dtype = dtype_util.common_dtype([heights, edgescales], float_type)
            if heights is not None:
                heights = tf.convert_to_tensor(
                    heights, name='heights', dtype=dtype)
            self._heights = _validate_arg_if_not_none(
                heights, tf.assert_positive, validate_args)
            if lengthscales is not None:
                lengthscales = tf.convert_to_tensor(
                    lengthscales, dtype=dtype, name='lengthscales')
                lengthscales = tf.nn.top_k(lengthscales, k=tf.shape(lengthscales)[-1], sorted=True)[0]
                edgescales = tf.reciprocal(lengthscales)

            if edgescales is not None:
                edgescales = tf.convert_to_tensor(
                    edgescales, dtype=dtype, name='edgescales')
                edgescales = tf.reverse(tf.nn.top_k(edgescales, k=tf.shape(edgescales)[-1], sorted=True)[0], axis=[-1])
                lengthscales = tf.reciprocal(edgescales)

            self._edgescales = _validate_arg_if_not_none(
                edgescales, tf.assert_positive, validate_args)
            self._lengthscales = _validate_arg_if_not_none(
                lengthscales, tf.assert_positive, validate_args)
            tf.assert_same_float_dtype([self._heights, self._edgescales, self._lengthscales])
        super(Histogram, self).__init__(
            feature_ndims, dtype=dtype, name=name)

    # def plot_spectrum(self,sess,ax=None):
    #     h,e = sess.run([self.heights, self.edgescales])
    #     n = h.shape[-1]
    #     if ax is None:
    #         fig, ax = plt.subplots(1,1)
    #     for i in range(n):
    #         ax.bar(0.5*(e[i+1]+e[i]),h[i],e[i+1]-e[i])
    #
    # def plot_kernel(self,sess,ax=None):
    #     x0 = tf.constant([[0.]], dtype=self.lengthscales.dtype)
    #     x = tf.cast(tf.linspace(x0[0,0],tf.reduce_max(self.lengthscales)*2.,100)[:,None],self.lengthscales.dtype)
    #     K_line = self.matrix(x, x0)
    #     K_line,x = sess.run([K_line,x])
    #     if ax is None:
    #         fig, ax = plt.subplots(1,1)
    #     ax.plot(x[:,0], K_line[:, 0])

    @property
    def heights(self):
        """Heights parameter."""
        return self._heights

    @property
    def edgescales(self):
        """Edgescales parameter."""
        return self._edgescales

    @property
    def lengthscales(self):
        """Edgescales parameter."""
        return self._lengthscales

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            scalar_shape if self.heights is None else self.heights.shape,
            scalar_shape if self.edgescales is None else self.edgescales.shape)

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            [] if self.heights is None else tf.shape(self.heights),
            [] if self.edgescales is None else tf.shape(self.edgescales))

    def _apply(self, x1, x2, param_expansion_ndims=0):
        # Use util.sqrt_with_finite_grads to avoid NaN gradients when `x1 == x2`.norm = util.sqrt_with_finite_grads(
        # x1 = B,Np,D -> B,Np,1,D
        # x2 = B,N,D -> B,1,N,D
        # B, Np,N
        with tf.control_dependencies([tf.assert_equal(tf.shape(self.heights)[-1] + 1, tf.shape(self.edgescales)[-1])]):
            norm = util.sqrt_with_finite_grads(util.sum_rightmost_ndims_preserving_shape(
                tf.squared_difference(x1, x2), self.feature_ndims))
        # B(1),1,Np,N
        norm = tf.expand_dims(norm, -(param_expansion_ndims + 1))

        # B(1), H+1, 1, 1
        edgescales = util.pad_shape_right_with_ones(
            self.edgescales, ndims=param_expansion_ndims)
        norm *= edgescales
        norm *= 2 * np.pi

        zeros = tf.zeros(tf.shape(self.heights)[:-1], dtype=self.heights.dtype)[..., None]
        # B(1),1+H+1
        heights = tf.concat([zeros, self.heights, zeros], axis=-1)
        # B(1), H+1
        dheights = heights[..., :-1] - heights[..., 1:]
        # B(1), H+1, 1, 1
        dheights = util.pad_shape_right_with_ones(
            dheights, ndims=param_expansion_ndims)
        # B(1), H+1, 1, 1
        dheights *= edgescales

        def _sinc(x):
            return tf.sin(x) * tf.reciprocal(x)

        # B(1), H+1, N, Np
        sincs = tf.where(tf.less(norm, tf.constant(1e-15, dtype=norm.dtype)), tf.ones_like(norm), _sinc(norm))
        # B(1), H+1, N, Np
        result = dheights * sincs
        # B(1), N,Np
        return tf.reduce_sum(result, axis=-3)