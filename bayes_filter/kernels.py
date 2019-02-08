import tensorflow as tf
import tensorflow_probability as tfp
from .settings import float_type
from .parameters import Parameter, ScaledPositiveBijector, SphericalToCartesianBijector
import numpy as np

class DTECFrozenFlow(object):
    allowed_obs_type = ['TEC', 'DTEC', 'DDTEC']
    def __init__(self, variance=1.0, lengthscales=10.0,
                 velocity=[0., 0., 0.], a=250., b=50., resolution=10,
                 fed_kernel='RBF', obs_type='TEC'):
        if obs_type not in DTECFrozenFlow.allowed_obs_type:
            raise ValueError("{} is an invalid obs_type. Must be in {}.".format(obs_type, DTECFrozenFlow.allowed_obs_type))
        self.obs_type = obs_type

        if not isinstance(variance, Parameter):
            variance = Parameter(bijector=ScaledPositiveBijector(variance), constrained_value=variance)
        if not isinstance(lengthscales, Parameter):
            lengthscales = Parameter(bijector=ScaledPositiveBijector(np.max(lengthscales)), constrained_value=lengthscales)
        if not isinstance(a, Parameter):
            a = Parameter(bijector=ScaledPositiveBijector(a), constrained_value=a)
        if not isinstance(b, Parameter):
            b = Parameter(bijector=ScaledPositiveBijector(b), constrained_value=b)
        if not isinstance(velocity, Parameter):
            velocity = Parameter(bijector=SphericalToCartesianBijector(), constrained_value=velocity)
        self.variance = variance
        self.lengthscales = lengthscales
        self.a = a
        self.b = b
        self.velocity = velocity

        if fed_kernel in ["RBF", "SE"]:
            fed_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                amplitude=self.variance.constrained_value, length_scale=self.lengthscales.constrained_value, feature_ndims=1)
        if fed_kernel in ["M32"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                amplitude=self.variance.constrained_value, length_scale=self.lengthscales.constrained_value, feature_ndims=1)
            if fed_kernel in ["M52"]:
                fed_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                    amplitude=self.variance.constrained_value, length_scale=self.lengthscales.constrained_value,
                    feature_ndims=1)
        if fed_kernel in ["M12","OU"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                amplitude=self.variance.constrained_value, length_scale=self.lengthscales.constrained_value, feature_ndims=1)
        self.fed_kernel = fed_kernel

        self.resolution = tf.convert_to_tensor(resolution,tf.int32)

    def K(self, X, X_dims, X2=None, X2_dims=None):


        # for DDTEC: I[i,alpha, beta, i, alpha, beta]
        #           + I[i0,alpha, beta, i0, alpha, beta]
        #           - 2 sym(I[i0,alpha, beta, i, alpha, beta])
        #           + I[i,alpha0, beta, i, alpha0, beta]
        #           + I[i0,alpha0, beta, i0, alpha0, beta]
        #           - 2 sym(I[i0,alpha0, beta, i, alpha0, beta])
        #           - 2 I[i,alpha, beta, i, alpha0, beta]
        #           - 2 I[i0,alpha, beta, i0, alpha0, beta]
        #           + 2 I[i,alpha, beta, i0, alpha0, beta]
        #           + 2 I[i0,alpha, beta, i, alpha0, beta]

        if X2 is None:
            X2 = X
            X2_dims = X_dims

        # N
        times = X[:, 0]
        # N,3
        directions = X[:, slice(1, 4, 1)]
        # N,3
        antennas = X[:, slice(4, 7, 1)]

        # Np
        times2 = X2[:, 0]
        # Np,3
        directions2 = X2[:, slice(1, 4, 1)]
        # Np,3
        antennas2 = X2[:, slice(4, 7, 1)]

        # N
        sec1 = tf.reciprocal(directions[:, 2], name='sec1')
        # Np
        sec2 = tf.reciprocal(directions2[:, 2], name='sec2')

        # N
        ds1 = sec1 * self.b.constrained_value / tf.cast(self.resolution - 1, float_type)
        # Np
        ds2 = sec2 * self.b.constrained_value / tf.cast(self.resolution - 1, float_type)
        # N,Np
        ds1ds2 = ds1[:, None] * ds2[None, :]

        ###
        # 1 1 terms
        # N
        s1m = sec1 * (self.a.constrained_value - (antennas[:, 2] - self.velocity.constrained_value[2] * times)) - 0.5 * sec1 * self.b.constrained_value
        # N
        s1p = sec1 * (self.a.constrained_value - (antennas[:, 2] - self.velocity.constrained_value[2] * times)) + 0.5 * sec1 * self.b.constrained_value

        # Np
        s2m = sec2 * (self.a.constrained_value - (antennas2[:, 2] - self.velocity.constrained_value[2] * times2)) - 0.5 * sec2 * self.b.constrained_value
        # Np
        s2p = sec2 * (self.a.constrained_value - (antennas2[:, 2] - self.velocity.constrained_value[2] * times2)) + 0.5 * sec2 * self.b.constrained_value

        # res, N
        s1 = s1m[None, :] + ((s1p - s1m)[None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[:, None], float_type)
        # res, Np
        s2 = s2m[None, :] + ((s2p - s2m)[None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[:, None], float_type)

        # I00

        # res, N, 3
        y1 = antennas[None, :, :] - self.velocity.constrained_value[None, None, :] * times[None, :, None] + directions[None, :, :] * s1[:,:,None]
        shape1 = tf.shape(y1)
        # res1 N, 3
        y1 = tf.reshape(y1, (-1, 3))
        # res, Np, 3
        y2 = antennas2[None, :, :] - self.velocity.constrained_value[None, None, :] * times2[None, :, None] + directions2[None, :, :] * s2[:, :, None]
        shape2 = tf.shape(y2)
        # res2 Np, 3
        y2 = tf.reshape(y2, (-1, 3))

        # res1 N, res2 Np
        K = self.fed_kernel.matrix(y1, y2)
        shape = tf.concat([shape1[:2], shape2[:2]], axis=0)
        # res1, N, res2, Np
        K = tf.reshape(K, shape)

        # N,Np
        I = 0.25 * ds1ds2 * tf.add_n([K[0, :, 0, :],
                                        K[-1, :, 0, :],
                                        K[0, :, -1, :],
                                        K[-1, :, -1, :],
                                        2 * tf.reduce_sum(K[-1, :, :, :], axis=[1]),
                                        2 * tf.reduce_sum(K[0, :, :, :], axis=[1]),
                                        2 * tf.reduce_sum(K[:, :, -1, :], axis=[0]),
                                        2 * tf.reduce_sum(K[:, :, 0, :], axis=[0]),
                                        4 * tf.reduce_sum(K[1:-1, :, 1:-1, :], axis=[0, 2])])

        if self.obs_type == 'TEC':
            # for TEC: I[i,alpha, beta, i', alpha', beta']
            return I

        if self.obs_type == 'DTEC':
            # for DTEC: I[i,alpha, beta, i', alpha', beta']
            #           + I[i0,alpha, beta, i0, alpha', beta']
            #           - I[i0,alpha, beta, i', alpha', beta']
            #           - I[i,alpha, beta, i0, alpha', beta']
            I = tf.reshape(I, tf.concat([X_dims, X2_dims],axis=0))
            #Nt, Nd, Na, Nt', Nd', Na'
            I00 = I[:,:,1:,:,:,1:]
            shape1 = tf.cast(tf.reduce_prod(tf.shape(I00)[:3]),tf.int32)
            shape2 = tf.cast(tf.reduce_prod(tf.shape(I00)[3:]), tf.int32)
            shape = tf.concat([[shape1], [shape2]], axis=0)
            #Nt, Nd, 1, Nt', Nd', 1
            I11 = I[:,:,0:1,:,:,0:1]
            # Nt, Nd, 1, Nt', Nd', Na'
            I10 = I[:, :, 0:1, :, :, 1:]
            # Nt, Nd, Na, Nt', Nd', 1
            I01 = I[:, :, 1:, :, :, 0:1]
            return tf.reshape(I00 + I11 -I01 -I10, shape)

        if self.obs_type == 'DDTEC':
            # for DDTEC:  E[(DTEC[i,alpha,beta] - DTEC[i,alpha0,beta]) * (DTEC[i',alpha',beta'] - DTEC[i',alpha0,beta'])]
            # DTEC[i,alpha,beta]DTEC[i',alpha',beta'] - DTEC[i,alpha0,beta]DTEC[i',alpha',beta'] - DTEC[i,alpha,beta]DTEC[i',alpha0,beta'] + DTEC[i,alpha0,beta]DTEC[i',alpha0,beta']
            # DTEC[i,alpha,beta]DTEC[i',alpha',beta']:
            #             I[i,alpha, beta, i', alpha', beta']
            #           + I[i0,alpha, beta, i0, alpha', beta']
            #           - I[i0,alpha, beta, i', alpha', beta']
            #           - I[i,alpha, beta, i0, alpha', beta']
            # DTEC[i,alpha0,beta]DTEC[i',alpha0,beta']:
            #           + I[i,alpha0, beta, i', alpha0, beta']
            #           + I[i0,alpha0, beta, i0, alpha0, beta']
            #           - I[i0,alpha0, beta, i', alpha0, beta']
            #           - I[i,alpha0, beta, i0, alpha0, beta']
            # DTEC[i,alpha0,beta]DTEC[i',alpha',beta']:
            #           + I[i, alpha0, beta, i', alpha', beta']
            #           + I[i0,alpha0, beta, i0, alpha', beta']
            #           - I[i0,alpha0, beta, i', alpha', beta']
            #           - I[i,alpha0, beta, i0, alpha', beta']
            # DTEC[i,alpha,beta]DTEC[i',alpha0,beta']:
            #           + I[i, alpha, beta, i', alpha0, beta']
            #           + I[i0,alpha, beta, i0, alpha0, beta']
            #           - I[i0,alpha, beta, i', alpha0, beta']
            #           - I[i,alpha, beta, i0, alpha0, beta']
            I = tf.reshape(I, tf.concat([X_dims, X2_dims], axis=0))
            # Nt, Nd, Na, Nt', Nd', Na'
            I00 = I[:, 1:, 1:, :, 1:, 1:]
            shape1 = tf.cast(tf.reduce_prod(tf.shape(I00)[:3]), tf.int32)
            shape2 = tf.cast(tf.reduce_prod(tf.shape(I00)[3:]), tf.int32)
            shape = tf.concat([[shape1], [shape2]], axis=0)
            # Nt, Nd, 1, Nt', Nd', 1
            I11 = I[:, 1:, 0:1, :, 1:, 0:1]
            # Nt, Nd, 1, Nt', Nd', Na'
            I10 = I[:, 1:, 0:1, :, 1:, 1:]
            # Nt, Nd, Na, Nt', Nd', 1
            I01 = I[:, 1:, 1:, :, 1:, 0:1]
            I00a = I[:, 0:1, 1:, :, 0:1, 1:]
            # Nt, Nd, 1, Nt', Nd', 1
            I11a = I[:, 0:1, 0:1, :, 0:1, 0:1]
            # Nt, Nd, 1, Nt', Nd', Na'
            I10a = I[:, 0:1, 0:1, :, 0:1, 1:]
            # Nt, Nd, Na, Nt', Nd', 1
            I01a = I[:, 0:1, 1:, :, 0:1, 0:1]
            I00b = I[:, 0:1, 1:, :, 1:, 1:]
            # Nt, Nd, 1, Nt', Nd', 1
            I11b = I[:, 0:1, 0:1, :, 1:, 0:1]
            # Nt, Nd, 1, Nt', Nd', Na'
            I10b = I[:, 0:1, 0:1, :, 1:, 1:]
            # Nt, Nd, Na, Nt', Nd', 1
            I01b = I[:, 0:1, 1:, :, 1:, 0:1]
            I00c = I[:, 1:, 1:, :, 0:1, 1:]
            # Nt, Nd, 1, Nt', Nd', 1
            I11c = I[:, 1:, 0:1, :, 0:1, 0:1]
            # Nt, Nd, 1, Nt', Nd', Na'
            I10c = I[:, 1:, 0:1, :, 0:1, 1:]
            # Nt, Nd, Na, Nt', Nd', 1
            I01c = I[:, 1:, 1:, :, 0:1, 0:1]
            return tf.reshape(I00 + I11 - I01 - I10 + I00a + I11a - I01a - I10a - I00b - I11b + I01b + I10b - I00c - I11c + I01c + I10c, shape)




