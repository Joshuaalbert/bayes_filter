import tensorflow as tf
import tensorflow_probability as tfp
from .settings import float_type
from .parameters import Parameter, ScaledPositiveBijector, SphericalToCartesianBijector


class DTECFrozenFlow(object):
    allowed_obs_type = ['TEC', 'DTEC', 'DDTEC']
    def __init__(self, variance=0.1, lengthscales=10.0,
                 a=250., b=50., resolution=10, velocity=[0.,0.,0.],
                 fed_kernel='RBF', obs_type='TEC', squeeze=True):
        if obs_type not in DTECFrozenFlow.allowed_obs_type:
            raise ValueError("{} is an invalid obs_type. Must be in {}.".format(obs_type, DTECFrozenFlow.allowed_obs_type))
        self.obs_type = obs_type
        self.squeeze = squeeze

        if not isinstance(variance, Parameter):
            variance = Parameter(bijector=ScaledPositiveBijector(0.1), constrained_value=variance, shape = (-1,))
        if not isinstance(lengthscales, Parameter):
            lengthscales = Parameter(bijector=ScaledPositiveBijector(10.), constrained_value=lengthscales, shape = (-1,))
        if not isinstance(a, Parameter):
            a = Parameter(bijector=ScaledPositiveBijector(250.), constrained_value=a, shape = (-1,))
        if not isinstance(b, Parameter):
            b = Parameter(bijector=ScaledPositiveBijector(100.), constrained_value=b, shape = (-1,))
        if not isinstance(velocity, Parameter):
            velocity = Parameter(bijector=SphericalToCartesianBijector(), constrained_value=velocity, shape= (-1, 3))

        self.variance = variance
        self.lengthscales = lengthscales
        self.a = a
        self.b = b
        self.velocity = velocity

        if fed_kernel in ["RBF", "SE"]:
            fed_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
        if fed_kernel in ["M32"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
        if fed_kernel in ["M52"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
        if fed_kernel in ["M12","OU"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
        self.fed_kernel = fed_kernel

        self.resolution = tf.convert_to_tensor(resolution,tf.int32)

    def K(self, X, X_dims, X2=None, X2_dims=None):

        if X2 is None:
            X2 = X
            X2_dims = X_dims

        # N, 1
        times = X[:, slice(0,1,1)]
        # N,3
        directions = X[:, slice(1, 4, 1)]
        # N,3
        antennas = X[:, slice(4, 7, 1)]

        # Np, 1
        times2 = X2[:, slice(0,1,1)]
        # Np,3
        directions2 = X2[:, slice(1, 4, 1)]
        # Np,3
        antennas2 = X2[:, slice(4, 7, 1)]

        # N
        sec1 = tf.reciprocal(directions[:, 2], name='sec1')
        # Np
        sec2 = tf.reciprocal(directions2[:, 2], name='sec2')

        # num_chains, N
        bsec1 = sec1 * self.b.constrained_value[:, None]
        # num_chains, Np
        bsec2 = sec2 * self.b.constrained_value[:, None]

        #num_chains, N
        ds1 = bsec1 / tf.cast(self.resolution - 1, float_type)
        # num_chains, Np
        ds2 = bsec2 / tf.cast(self.resolution - 1, float_type)
        # num_chains, N,Np
        ds1ds2 = ds1[:,:, None] * ds2[:,None, :]

        # num_chains, N
        s1m = sec1[None, :] * (self.a.constrained_value[:, None] - antennas[:, 2] + self.velocity.constrained_value[:,None, 2] * times[None, :, 0]) - 0.5 * bsec1
        # num_chains, N
        s1p = s1m + bsec1
        #sec1 * (self.a.constrained_value - antennas[:, 2]) + 0.5 * sec1 * self.b.constrained_value

        # num_chains, Np
        s2m = sec2[None, :] * (self.a.constrained_value[:, None] - antennas2[:, 2] + self.velocity.constrained_value[:,None, 2] * times2[None, :, 0]) - 0.5 * bsec2
        # num_chains, Np
        s2p = s2m + bsec2
        #sec2 * (self.a.constrained_value - antennas2[:, 2]) + 0.5 * sec2 * self.b.constrained_value

        # num_chains, res, N
        s1 = s1m[:, None, :] + ((s1p - s1m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None], float_type)
        # num_chains, res, Np
        s2 = s2m[:, None, :] + ((s2p - s2m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None], float_type)

        # I00

        # num_chains, res1, N, 3
        y1 = antennas[None, None, :, :] + directions[None, None, :, :] * s1[:,:,:,None] - self.velocity.constrained_value[:, None, None, :] * times[None, None, :, :]
        shape1 = tf.shape(y1)
        # num_chains, res1 N, 3
        y1 = tf.reshape(y1, tf.concat([tf.shape(y1)[0:1], [-1, 3]],axis=0))
        # num_chains, res2, Np, 3
        y2 = antennas2[None, None, :, :] + directions2[None, None, :, :] * s2[:, :, :, None] - self.velocity.constrained_value[:, None, None, :] * times2[None, None, :, :]
        shape2 = tf.shape(y2)
        # num_chains, res2 Np, 3
        y2 = tf.reshape(y2, tf.concat([tf.shape(y2)[0:1], [-1, 3]],axis=0))

        # num_chains, res1 N, res2 Np
        K = self.fed_kernel.matrix(y1, y2)
        shape = tf.concat([tf.shape(K)[0:1], shape1[1:3], shape2[1:3]], axis=0)
        # num_chains, res1, N, res2, Np
        K = tf.reshape(K, shape)

        # num_chains, N,Np
        I = 0.25 * ds1ds2 * tf.add_n([K[:, 0, :, 0, :],
                                        K[:, -1, :, 0, :],
                                        K[:, 0, :, -1, :],
                                        K[:, -1, :, -1, :],
                                        2 * tf.reduce_sum(K[:, -1, :, :, :], axis=[2]),
                                        2 * tf.reduce_sum(K[:, 0, :, :, :], axis=[2]),
                                        2 * tf.reduce_sum(K[:, :, :, -1, :], axis=[1]),
                                        2 * tf.reduce_sum(K[:, :, :, 0, :], axis=[1]),
                                        4 * tf.reduce_sum(K[:, 1:-1, :, 1:-1, :], axis=[1, 3])])

        if self.obs_type == 'TEC':
            # for TEC: I[i,alpha, beta, i', alpha', beta']
            result = I

        if self.obs_type == 'DTEC':
            # for DTEC: I[i,alpha, beta, i', alpha', beta']
            #           + I[i0,alpha, beta, i0, alpha', beta']
            #           - I[i0,alpha, beta, i', alpha', beta']
            #           - I[i,alpha, beta, i0, alpha', beta']
            I = tf.reshape(I, tf.concat([tf.shape(I)[0:1], X_dims, X2_dims],axis=0))
            #num_chains, Nt, Nd, Na, Nt', Nd', Na'
            I00 = I[:, :,:,1:,:,:,1:]
            shape1 = tf.cast(tf.reduce_prod(tf.shape(I00)[1:4]), tf.int32)
            shape2 = tf.cast(tf.reduce_prod(tf.shape(I00)[4:7]), tf.int32)
            shape = tf.concat([tf.shape(I)[0:1], [shape1], [shape2]], axis=0)
            #num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11 = I[:, :,:,0:1,:,:,0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10 = I[:, :, :, 0:1, :, :, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01 = I[:, :, :, 1:, :, :, 0:1]
            result = tf.reshape(I00 + I11 -I01 -I10, shape)

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
            I = tf.reshape(I, tf.concat([tf.shape(I)[0:1], X_dims, X2_dims],axis=0))
            I00 = I[:, :, 1:, 1:, :, 1:, 1:]
            shape1 = tf.cast(tf.reduce_prod(tf.shape(I00)[1:4]), tf.int32)
            shape2 = tf.cast(tf.reduce_prod(tf.shape(I00)[4:]), tf.int32)
            shape = tf.concat([tf.shape(I)[0:1], [shape1], [shape2]], axis=0)
            # num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11 = I[:, :, 1:, 0:1, :, 1:, 0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10 = I[:, :, 1:, 0:1, :, 1:, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01 = I[:, :, 1:, 1:, :, 1:, 0:1]
            I00a = I[:, :, 0:1, 1:, :, 0:1, 1:]
            # num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11a = I[:, :, 0:1, 0:1, :, 0:1, 0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10a = I[:, :, 0:1, 0:1, :, 0:1, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01a = I[:, :, 0:1, 1:, :, 0:1, 0:1]
            I00b = I[:, :, 0:1, 1:, :, 1:, 1:]
            # num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11b = I[:, :, 0:1, 0:1, :, 1:, 0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10b = I[:, :, 0:1, 0:1, :, 1:, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01b = I[:, :, 0:1, 1:, :, 1:, 0:1]
            I00c = I[:, :, 1:, 1:, :, 0:1, 1:]
            # num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11c = I[:, :, 1:, 0:1, :, 0:1, 0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10c = I[:, :, 1:, 0:1, :, 0:1, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01c = I[:, :, 1:, 1:, :, 0:1, 0:1]
            result = tf.reshape(I00 + I11 - I01 - I10 + I00a + I11a - I01a - I10a - I00b - I11b + I01b + I10b - I00c - I11c + I01c + I10c, shape)

        if self.squeeze:
            return tf.squeeze(result)
        else:
            return result

class DTECIsotropicTime(object):
    """
    The DTEC kernel is derived from first principles by assuming a GRF over the electron density, from which DTEC kernel
    can be caluclated as,

    K(ray_i, ray_j) =     I(a_i, k_i, t_i, a_j, k_j, t_j)  + I(a0_i, k_i, t_i, a0_j, k_j, t_j)
                        - I(a0_i, k_i, t_i, a_j, k_j, t_j) - I(a_i, k_i, t_i, a0_j, k_j, t_j)

    where,
                I(a,b,c,d,e,g) = iint [K_ne(y(a,b,c), y(d,e,f))](s1,s2) ds1 ds2
    """
    allowed_obs_type = ['TEC', 'DTEC', 'DDTEC']
    def __init__(self, variance=0.1, lengthscales=10.0,
                 a=250., b=50., resolution=10, timescale=30.,
                 fed_kernel='RBF', obs_type='TEC', squeeze=True):
        if obs_type not in DTECIsotropicTime.allowed_obs_type:
            raise ValueError("{} is an invalid obs_type. Must be in {}.".format(obs_type, DTECIsotropicTime.allowed_obs_type))
        self.obs_type = obs_type
        self.squeeze = squeeze

        if not isinstance(variance, Parameter):
            variance = Parameter(bijector=ScaledPositiveBijector(0.1), constrained_value=variance, shape = (-1,))
        if not isinstance(lengthscales, Parameter):
            lengthscales = Parameter(bijector=ScaledPositiveBijector(10.), constrained_value=lengthscales, shape = (-1,))
        if not isinstance(timescale, Parameter):
            timescale = Parameter(bijector=ScaledPositiveBijector(50.), constrained_value=timescale, shape = (-1,))
        if not isinstance(a, Parameter):
            a = Parameter(bijector=ScaledPositiveBijector(250.), constrained_value=a, shape = (-1,))
        if not isinstance(b, Parameter):
            b = Parameter(bijector=ScaledPositiveBijector(100.), constrained_value=b, shape = (-1,))

        self.variance = variance
        self.lengthscales = lengthscales
        self.timescale = timescale
        self.a = a
        self.b = b

        if fed_kernel in ["RBF", "SE"]:
            fed_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
            time_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                length_scale=self.timescale.constrained_value)
        if fed_kernel in ["M32"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
            time_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                length_scale=self.timescale.constrained_value)
        if fed_kernel in ["M52"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
            time_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                length_scale=self.timescale.constrained_value)
        if fed_kernel in ["M12","OU"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
            time_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                length_scale=self.timescale.constrained_value)
        self.fed_kernel = fed_kernel
        self.time_kernel = time_kernel

        self.resolution = tf.convert_to_tensor(resolution,tf.int32)

    def K(self, X, X_dims, X2=None, X2_dims=None):

        if X2 is None:
            X2 = X
            X2_dims = X_dims

        #TODO symmetrize when possible

        # def _correct_dims(X,dims):
        #     """Correct for partial batch dims."""
        #     N = tf.shape(X)[0]
        #     N_slice = tf.reduce_prod(dims[1:])
        #     return tf.concat([[tf.floordiv(N,N_slice)], dims[1:]],axis=0)
        #
        # X_dims, X2_dims = _correct_dims(X, X_dims), _correct_dims(X2, X2_dims)

        # N, 1
        times = X[:, slice(0,1,1)]
        # N,3
        directions = X[:, slice(1, 4, 1)]
        # N,3
        antennas = X[:, slice(4, 7, 1)]

        # Np, 1
        times2 = X2[:, slice(0,1,1)]
        # Np,3
        directions2 = X2[:, slice(1, 4, 1)]
        # Np,3
        antennas2 = X2[:, slice(4, 7, 1)]

        # N
        sec1 = tf.reciprocal(directions[:, 2], name='sec1')
        # Np
        sec2 = tf.reciprocal(directions2[:, 2], name='sec2')

        # num_chains, N
        bsec1 = sec1 * self.b.constrained_value[:, None]
        # num_chains, Np
        bsec2 = sec2 * self.b.constrained_value[:, None]

        #num_chains, N
        ds1 = bsec1 / tf.cast(self.resolution - 1, float_type)
        # num_chains, Np
        ds2 = bsec2 / tf.cast(self.resolution - 1, float_type)
        # num_chains, N,Np
        ds1ds2 = ds1[:,:, None] * ds2[:,None, :]

        # num_chains, N
        s1m = sec1[None, :] * (self.a.constrained_value[:, None] - antennas[:, 2]) - 0.5 * bsec1
        # num_chains, N
        s1p = s1m + bsec1
        #sec1 * (self.a.constrained_value - antennas[:, 2]) + 0.5 * sec1 * self.b.constrained_value

        # num_chains, Np
        s2m = sec2[None, :] * (self.a.constrained_value[:, None] - antennas2[:, 2]) - 0.5 * bsec2
        # num_chains, Np
        s2p = s2m + bsec2
        #sec2 * (self.a.constrained_value - antennas2[:, 2]) + 0.5 * sec2 * self.b.constrained_value

        # num_chains, res, N
        s1 = s1m[:, None, :] + ((s1p - s1m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None], float_type)
        # num_chains, res, Np
        s2 = s2m[:, None, :] + ((s2p - s2m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None], float_type)

        # I00

        # num_chains, res1, N, 3
        y1 = antennas[None, None, :, :] + directions[None, None, :, :] * s1[:,:,:,None]
        shape1 = tf.shape(y1)
        # num_chains, res1 N, 3
        y1 = tf.reshape(y1, tf.concat([tf.shape(y1)[0:1], [-1, 3]],axis=0))
        # num_chains, res2, Np, 3
        y2 = antennas2[None, None, :, :] + directions2[None, None, :, :] * s2[:, :, :, None]
        shape2 = tf.shape(y2)
        # num_chains, res2 Np, 3
        y2 = tf.reshape(y2, tf.concat([tf.shape(y2)[0:1], [-1, 3]],axis=0))

        # num_chains, res1 N, res2 Np
        K_space = self.fed_kernel.matrix(y1, y2)
        shape = tf.concat([tf.shape(K_space)[0:1], shape1[1:3], shape2[1:3]], axis=0)
        # num_chains, N, Np
        K_time = self.time_kernel.matrix(times[None, :, :],times2[None, :, :])

        # num_chains, res1, N, res2, Np
        K = tf.reshape(K_space, shape) * K_time[:, None, :, None, :]
        # with tf.control_dependencies([tf.print(tf.shape(K))]):
        # num_chains, N,Np
        I = 0.25 * ds1ds2 * tf.add_n([K[:, 0, :, 0, :],
                                        K[:, -1, :, 0, :],
                                        K[:, 0, :, -1, :],
                                        K[:, -1, :, -1, :],
                                        2 * tf.reduce_sum(K[:, -1, :, :, :], axis=[2]),
                                        2 * tf.reduce_sum(K[:, 0, :, :, :], axis=[2]),
                                        2 * tf.reduce_sum(K[:, :, :, -1, :], axis=[1]),
                                        2 * tf.reduce_sum(K[:, :, :, 0, :], axis=[1]),
                                        4 * tf.reduce_sum(K[:, 1:-1, :, 1:-1, :], axis=[1, 3])])

        if self.obs_type == 'TEC':
            # for TEC: I[i,alpha, beta, i', alpha', beta']
            result = I

        if self.obs_type == 'DTEC':
            # for DTEC: I[i,alpha, beta, i', alpha', beta']
            #           + I[i0,alpha, beta, i0, alpha', beta']
            #           - I[i0,alpha, beta, i', alpha', beta']
            #           - I[i,alpha, beta, i0, alpha', beta']
            I = tf.reshape(I, tf.concat([tf.shape(I)[0:1], X_dims, X2_dims],axis=0))
            #num_chains, Nt, Nd, Na, Nt', Nd', Na'
            I00 = I[:, :,:,1:,:,:,1:]
            shape1 = tf.cast(tf.reduce_prod(tf.shape(I00)[1:4]),tf.int32)
            shape2 = tf.cast(tf.reduce_prod(tf.shape(I00)[4:7]), tf.int32)
            shape = tf.concat([tf.shape(I)[0:1], [shape1], [shape2]], axis=0)
            #num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11 = I[:, :,:,0:1,:,:,0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10 = I[:, :, :, 0:1, :, :, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01 = I[:, :, :, 1:, :, :, 0:1]
            result = tf.reshape(I00 + I11 -I01 -I10, shape)

        if self.obs_type == 'DDTEC':
            #abc abc ->
            # for DDTEC:  E[(DTEC[i,alpha,beta] - DTEC[i,alpha0,beta]) * (DTEC[i',alpha',beta'] - DTEC[i',alpha0,beta'])]
            # DTEC[i,alpha,beta]DTEC[i',alpha',beta'] - DTEC[i,alpha0,beta]DTEC[i',alpha',beta'] - DTEC[i,alpha,beta]DTEC[i',alpha0,beta'] + DTEC[i,alpha0,beta]DTEC[i',alpha0,beta']
            # DTEC[i,alpha,beta]DTEC[i',alpha',beta']:
            #           + I[i,alpha, beta, i', alpha', beta']
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
            I = tf.reshape(I, tf.concat([tf.shape(I)[0:1], X_dims, X2_dims],axis=0))
            I00 = I[:, :, 1:, 1:, :, 1:, 1:]
            shape1 = tf.cast(tf.reduce_prod(tf.shape(I00)[1:4]), tf.int32)
            shape2 = tf.cast(tf.reduce_prod(tf.shape(I00)[4:]), tf.int32)
            shape = tf.concat([tf.shape(I)[0:1], [shape1], [shape2]], axis=0)
            # num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11 = I[:, :, 1:, 0:1, :, 1:, 0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10 = I[:, :, 1:, 0:1, :, 1:, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01 = I[:, :, 1:, 1:, :, 1:, 0:1]
            I00a = I[:, :, 0:1, 1:, :, 0:1, 1:]
            # num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11a = I[:, :, 0:1, 0:1, :, 0:1, 0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10a = I[:, :, 0:1, 0:1, :, 0:1, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01a = I[:, :, 0:1, 1:, :, 0:1, 0:1]
            I00b = I[:, :, 0:1, 1:, :, 1:, 1:]
            # num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11b = I[:, :, 0:1, 0:1, :, 1:, 0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10b = I[:, :, 0:1, 0:1, :, 1:, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01b = I[:, :, 0:1, 1:, :, 1:, 0:1]
            I00c = I[:, :, 1:, 1:, :, 0:1, 1:]
            # num_chains, Nt, Nd, 1, Nt', Nd', 1
            I11c = I[:, :, 1:, 0:1, :, 0:1, 0:1]
            # num_chains, Nt, Nd, 1, Nt', Nd', Na'
            I10c = I[:, :, 1:, 0:1, :, 0:1, 1:]
            # num_chains, Nt, Nd, Na, Nt', Nd', 1
            I01c = I[:, :, 1:, 1:, :, 0:1, 0:1]
            result = tf.reshape(I00 + I11 - I01 - I10 + I00a + I11a - I01a - I10a - I00b - I11b + I01b + I10b - I00c - I11c + I01c + I10c, shape)

        if self.squeeze:
            return tf.squeeze(result)
        else:
            return result


class DTECIsotropicTimeLong(object):
    allowed_obs_type = ['DTEC']
    def __init__(self, variance=0.1, lengthscales=10.0,
                 a=250., b=50., resolution=10, timescale=30.,
                 fed_kernel='RBF', obs_type='TEC', squeeze=True):
        if obs_type not in DTECIsotropicTimeLong.allowed_obs_type:
            raise ValueError("{} is an invalid obs_type. Must be in {}.".format(obs_type, DTECIsotropicTimeLong.allowed_obs_type))
        self.obs_type = obs_type
        self.squeeze = squeeze

        if not isinstance(variance, Parameter):
            variance = Parameter(bijector=ScaledPositiveBijector(0.1), constrained_value=variance, shape = (-1,))
        if not isinstance(lengthscales, Parameter):
            lengthscales = Parameter(bijector=ScaledPositiveBijector(10.), constrained_value=lengthscales, shape = (-1,))
        if not isinstance(timescale, Parameter):
            timescale = Parameter(bijector=ScaledPositiveBijector(50.), constrained_value=timescale, shape = (-1,))
        if not isinstance(a, Parameter):
            a = Parameter(bijector=ScaledPositiveBijector(250.), constrained_value=a, shape = (-1,))
        if not isinstance(b, Parameter):
            b = Parameter(bijector=ScaledPositiveBijector(100.), constrained_value=b, shape = (-1,))

        self.variance = variance
        self.lengthscales = lengthscales
        self.timescale = timescale
        self.a = a
        self.b = b

        if fed_kernel in ["RBF", "SE"]:
            fed_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
            time_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                length_scale=self.timescale.constrained_value)
        if fed_kernel in ["M32"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
            time_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                length_scale=self.timescale.constrained_value)
        if fed_kernel in ["M52"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
            time_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                length_scale=self.timescale.constrained_value)
        if fed_kernel in ["M12","OU"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                amplitude=self.variance.constrained_value,
                length_scale=self.lengthscales.constrained_value)
            time_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                length_scale=self.timescale.constrained_value)
        self.fed_kernel = fed_kernel
        self.time_kernel = time_kernel

        self.resolution = tf.convert_to_tensor(resolution,tf.int32)

    def K(self, X, X_dims, X2=None, X2_dims=None):

        if X2 is None:
            X2 = X
            X2_dims = X_dims

        # def _correct_dims(X,dims):
        #     """Correct for partial batch dims."""
        #     N = tf.shape(X)[0]
        #     N_slice = tf.reduce_prod(dims[1:])
        #     return tf.concat([[tf.floordiv(N,N_slice)], dims[1:]],axis=0)
        #
        # X_dims, X2_dims = _correct_dims(X, X_dims), _correct_dims(X2, X2_dims)


        # N, 1
        times = X[:, slice(0,1,1)]
        # N,3
        directions = X[:, slice(1, 4, 1)]
        # N,3
        antennas = X[:, slice(4, 7, 1)]

        # Np, 1
        times2 = X2[:, slice(0,1,1)]
        # Np,3
        directions2 = X2[:, slice(1, 4, 1)]
        # Np,3
        antennas2 = X2[:, slice(4, 7, 1)]

        # N
        sec1 = tf.reciprocal(directions[:, 2], name='sec1')
        # Np
        sec2 = tf.reciprocal(directions2[:, 2], name='sec2')

        # num_chains, N
        bsec1 = sec1 * self.b.constrained_value[:, None]
        # num_chains, Np
        bsec2 = sec2 * self.b.constrained_value[:, None]

        #num_chains, N
        ds1 = bsec1 / tf.cast(self.resolution - 1, float_type)
        # num_chains, Np
        ds2 = bsec2 / tf.cast(self.resolution - 1, float_type)
        # num_chains, N,Np
        ds1ds2 = ds1[:,:, None] * ds2[:,None, :]

        # num_chains, N
        s1m = sec1[None, :] * (self.a.constrained_value[:, None] - antennas[:, 2]) - 0.5 * bsec1
        # num_chains, N
        s1p = s1m + bsec1
        #sec1 * (self.a.constrained_value - antennas[:, 2]) + 0.5 * sec1 * self.b.constrained_value

        # num_chains, Np
        s2m = sec2[None, :] * (self.a.constrained_value[:, None] - antennas2[:, 2]) - 0.5 * bsec2
        # num_chains, Np
        s2p = s2m + bsec2
        #sec2 * (self.a.constrained_value - antennas2[:, 2]) + 0.5 * sec2 * self.b.constrained_value

        # num_chains, res, N
        s1 = s1m[:, None, :] + ((s1p - s1m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None], float_type)
        # num_chains, res, Np
        s2 = s2m[:, None, :] + ((s2p - s2m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None], float_type)

        # I00

        # num_chains, res1, N, 3
        y1 = antennas[None, None, :, :] + directions[None, None, :, :] * s1[:,:,:,None]
        shape1 = tf.shape(y1)
        # num_chains, res1 N, 3
        y1 = tf.reshape(y1, tf.concat([tf.shape(y1)[0:1], [-1, 3]],axis=0))
        # num_chains, res2, Np, 3
        y2 = antennas2[None, None, :, :] + directions2[None, None, :, :] * s2[:, :, :, None]
        shape2 = tf.shape(y2)
        # num_chains, res2 Np, 3
        y2 = tf.reshape(y2, tf.concat([tf.shape(y2)[0:1], [-1, 3]],axis=0))

        # num_chains, res1 N, res2 Np
        K_space = self.fed_kernel.matrix(y1, y2)
        shape = tf.concat([tf.shape(K_space)[0:1], shape1[1:3], shape2[1:3]], axis=0)
        # num_chains, N, Np
        K_time = self.time_kernel.matrix(times[None, :, :],times2[None, :, :])

        # num_chains, res1, N, res2, Np
        K = tf.reshape(K_space, shape) * K_time[:, None, :, None, :]
        # with tf.control_dependencies([tf.print(tf.shape(K))]):
        # num_chains, N,Np
        I00 = 0.25 * ds1ds2 * tf.add_n([K[:, 0, :, 0, :],
                                        K[:, -1, :, 0, :],
                                        K[:, 0, :, -1, :],
                                        K[:, -1, :, -1, :],
                                        2 * tf.reduce_sum(K[:, -1, :, :, :], axis=[2]),
                                        2 * tf.reduce_sum(K[:, 0, :, :, :], axis=[2]),
                                        2 * tf.reduce_sum(K[:, :, :, -1, :], axis=[1]),
                                        2 * tf.reduce_sum(K[:, :, :, 0, :], axis=[1]),
                                        4 * tf.reduce_sum(K[:, 1:-1, :, 1:-1, :], axis=[1, 3])])

        # num_chains, N
        s1m = sec1[None, :] * (self.a.constrained_value[:, None] - 0.*antennas[:, 2]) - 0.5 * bsec1
        # num_chains, N
        s1p = s1m + bsec1
        # sec1 * (self.a.constrained_value - antennas[:, 2]) + 0.5 * sec1 * self.b.constrained_value

        # num_chains, Np
        s2m = sec2[None, :] * (self.a.constrained_value[:, None] - 0.*antennas2[:, 2]) - 0.5 * bsec2
        # num_chains, Np
        s2p = s2m + bsec2
        # sec2 * (self.a.constrained_value - antennas2[:, 2]) + 0.5 * sec2 * self.b.constrained_value

        # num_chains, res, N
        s1 = s1m[:, None, :] + ((s1p - s1m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None],
                                                                   float_type)
        # num_chains, res, Np
        s2 = s2m[:, None, :] + ((s2p - s2m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None],
                                                                   float_type)

        # I00

        # num_chains, res1, N, 3
        y1 = 0.*antennas[None, None, :, :] + directions[None, None, :, :] * s1[:, :, :, None]
        shape1 = tf.shape(y1)
        # num_chains, res1 N, 3
        y1 = tf.reshape(y1, tf.concat([tf.shape(y1)[0:1], [-1, 3]], axis=0))
        # num_chains, res2, Np, 3
        y2 = 0.*antennas2[None, None, :, :] + directions2[None, None, :, :] * s2[:, :, :, None]
        shape2 = tf.shape(y2)
        # num_chains, res2 Np, 3
        y2 = tf.reshape(y2, tf.concat([tf.shape(y2)[0:1], [-1, 3]], axis=0))

        # num_chains, res1 N, res2 Np
        K_space = self.fed_kernel.matrix(y1, y2)
        shape = tf.concat([tf.shape(K_space)[0:1], shape1[1:3], shape2[1:3]], axis=0)
        # num_chains, N, Np
        K_time = self.time_kernel.matrix(times[None, :, :], times2[None, :, :])

        # num_chains, res1, N, res2, Np
        K = tf.reshape(K_space, shape) * K_time[:, None, :, None, :]
        # with tf.control_dependencies([tf.print(tf.shape(K))]):
        # num_chains, N,Np
        I11 = 0.25 * ds1ds2 * tf.add_n([K[:, 0, :, 0, :],
                                      K[:, -1, :, 0, :],
                                      K[:, 0, :, -1, :],
                                      K[:, -1, :, -1, :],
                                      2 * tf.reduce_sum(K[:, -1, :, :, :], axis=[2]),
                                      2 * tf.reduce_sum(K[:, 0, :, :, :], axis=[2]),
                                      2 * tf.reduce_sum(K[:, :, :, -1, :], axis=[1]),
                                      2 * tf.reduce_sum(K[:, :, :, 0, :], axis=[1]),
                                      4 * tf.reduce_sum(K[:, 1:-1, :, 1:-1, :], axis=[1, 3])])

        # num_chains, N
        s1m = sec1[None, :] * (self.a.constrained_value[:, None] - 0.*antennas[:, 2]) - 0.5 * bsec1
        # num_chains, N
        s1p = s1m + bsec1
        # sec1 * (self.a.constrained_value - antennas[:, 2]) + 0.5 * sec1 * self.b.constrained_value

        # num_chains, Np
        s2m = sec2[None, :] * (self.a.constrained_value[:, None] - antennas2[:, 2]) - 0.5 * bsec2
        # num_chains, Np
        s2p = s2m + bsec2
        # sec2 * (self.a.constrained_value - antennas2[:, 2]) + 0.5 * sec2 * self.b.constrained_value

        # num_chains, res, N
        s1 = s1m[:, None, :] + ((s1p - s1m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None],
                                                                   float_type)
        # num_chains, res, Np
        s2 = s2m[:, None, :] + ((s2p - s2m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None],
                                                                   float_type)

        # I00

        # num_chains, res1, N, 3
        y1 = 0.*antennas[None, None, :, :] + directions[None, None, :, :] * s1[:, :, :, None]
        shape1 = tf.shape(y1)
        # num_chains, res1 N, 3
        y1 = tf.reshape(y1, tf.concat([tf.shape(y1)[0:1], [-1, 3]], axis=0))
        # num_chains, res2, Np, 3
        y2 = antennas2[None, None, :, :] + directions2[None, None, :, :] * s2[:, :, :, None]
        shape2 = tf.shape(y2)
        # num_chains, res2 Np, 3
        y2 = tf.reshape(y2, tf.concat([tf.shape(y2)[0:1], [-1, 3]], axis=0))

        # num_chains, res1 N, res2 Np
        K_space = self.fed_kernel.matrix(y1, y2)
        shape = tf.concat([tf.shape(K_space)[0:1], shape1[1:3], shape2[1:3]], axis=0)
        # num_chains, N, Np
        K_time = self.time_kernel.matrix(times[None, :, :], times2[None, :, :])

        # num_chains, res1, N, res2, Np
        K = tf.reshape(K_space, shape) * K_time[:, None, :, None, :]
        # with tf.control_dependencies([tf.print(tf.shape(K))]):
        # num_chains, N,Np
        I01 = 0.25 * ds1ds2 * tf.add_n([K[:, 0, :, 0, :],
                                      K[:, -1, :, 0, :],
                                      K[:, 0, :, -1, :],
                                      K[:, -1, :, -1, :],
                                      2 * tf.reduce_sum(K[:, -1, :, :, :], axis=[2]),
                                      2 * tf.reduce_sum(K[:, 0, :, :, :], axis=[2]),
                                      2 * tf.reduce_sum(K[:, :, :, -1, :], axis=[1]),
                                      2 * tf.reduce_sum(K[:, :, :, 0, :], axis=[1]),
                                      4 * tf.reduce_sum(K[:, 1:-1, :, 1:-1, :], axis=[1, 3])])

        # num_chains, N
        s1m = sec1[None, :] * (self.a.constrained_value[:, None] - antennas[:, 2]) - 0.5 * bsec1
        # num_chains, N
        s1p = s1m + bsec1
        # sec1 * (self.a.constrained_value - antennas[:, 2]) + 0.5 * sec1 * self.b.constrained_value

        # num_chains, Np
        s2m = sec2[None, :] * (self.a.constrained_value[:, None] - 0.*antennas2[:, 2]) - 0.5 * bsec2
        # num_chains, Np
        s2p = s2m + bsec2
        # sec2 * (self.a.constrained_value - antennas2[:, 2]) + 0.5 * sec2 * self.b.constrained_value

        # num_chains, res, N
        s1 = s1m[:, None, :] + ((s1p - s1m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None],
                                                                   float_type)
        # num_chains, res, Np
        s2 = s2m[:, None, :] + ((s2p - s2m)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None],
                                                                   float_type)

        # I00

        # num_chains, res1, N, 3
        y1 = antennas[None, None, :, :] + directions[None, None, :, :] * s1[:, :, :, None]
        shape1 = tf.shape(y1)
        # num_chains, res1 N, 3
        y1 = tf.reshape(y1, tf.concat([tf.shape(y1)[0:1], [-1, 3]], axis=0))
        # num_chains, res2, Np, 3
        y2 = 0.*antennas2[None, None, :, :] + directions2[None, None, :, :] * s2[:, :, :, None]
        shape2 = tf.shape(y2)
        # num_chains, res2 Np, 3
        y2 = tf.reshape(y2, tf.concat([tf.shape(y2)[0:1], [-1, 3]], axis=0))

        # num_chains, res1 N, res2 Np
        K_space = self.fed_kernel.matrix(y1, y2)
        shape = tf.concat([tf.shape(K_space)[0:1], shape1[1:3], shape2[1:3]], axis=0)
        # num_chains, N, Np
        K_time = self.time_kernel.matrix(times[None, :, :], times2[None, :, :])

        # num_chains, res1, N, res2, Np
        K = tf.reshape(K_space, shape) * K_time[:, None, :, None, :]
        # with tf.control_dependencies([tf.print(tf.shape(K))]):
        # num_chains, N,Np
        I10 = 0.25 * ds1ds2 * tf.add_n([K[:, 0, :, 0, :],
                                      K[:, -1, :, 0, :],
                                      K[:, 0, :, -1, :],
                                      K[:, -1, :, -1, :],
                                      2 * tf.reduce_sum(K[:, -1, :, :, :], axis=[2]),
                                      2 * tf.reduce_sum(K[:, 0, :, :, :], axis=[2]),
                                      2 * tf.reduce_sum(K[:, :, :, -1, :], axis=[1]),
                                      2 * tf.reduce_sum(K[:, :, :, 0, :], axis=[1]),
                                      4 * tf.reduce_sum(K[:, 1:-1, :, 1:-1, :], axis=[1, 3])])

        result = I00 + I11 - I01 - I10

        if self.squeeze:
            return tf.squeeze(result)
        else:
            return result


class DTECIsotropicTimeGeneral(object):
    """
    The DTEC kernel is derived from first principles by assuming a GRF over the electron density, from which DTEC kernel
    can be caluclated as,

    K(ray_i, ray_j) =     I(a_i, k_i, t_i, a_j, k_j, t_j)  + I(a0_i, k_i, t_i, a0_j, k_j, t_j)
                        - I(a0_i, k_i, t_i, a_j, k_j, t_j) - I(a_i, k_i, t_i, a0_j, k_j, t_j)

    where,
                I(a,b,c,d,e,g) = iint [K_ne(y(a,b,c), y(d,e,f))](s1,s2) ds1 ds2
    """
    allowed_obs_type = ['TEC', 'DTEC', 'DDTEC']

    def __init__(self, variance=0.1, lengthscales=10.0,
                 a=250., b=50.,  timescale=30., resolution=10,
                 fed_kernel='RBF', obs_type='TEC', squeeze=True):
        if obs_type not in DTECIsotropicTimeGeneral.allowed_obs_type:
            raise ValueError(
                "{} is an invalid obs_type. Must be in {}.".format(obs_type, DTECIsotropicTimeGeneral.allowed_obs_type))
        self.obs_type = obs_type
        self.squeeze = squeeze

        if not isinstance(variance, Parameter):
            variance = Parameter(bijector=ScaledPositiveBijector(0.1), constrained_value=variance, shape=(-1,1))
        if not isinstance(lengthscales, Parameter):
            lengthscales = Parameter(bijector=ScaledPositiveBijector(10.), constrained_value=lengthscales, shape=(-1,1))
        if not isinstance(timescale, Parameter):
            timescale = Parameter(bijector=ScaledPositiveBijector(50.), constrained_value=timescale, shape=(-1,1))
        if not isinstance(a, Parameter):
            a = Parameter(bijector=ScaledPositiveBijector(250.), constrained_value=a, shape=(-1,1))
        if not isinstance(b, Parameter):
            b = Parameter(bijector=ScaledPositiveBijector(100.), constrained_value=b, shape=(-1,1))

        self.variance = variance
        self.lengthscales = lengthscales
        self.timescale = timescale
        self.a = a
        self.b = b

        if fed_kernel in ["RBF", "SE"]:
            fed_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                amplitude=self.variance.constrained_value[:,0],
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M32"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                amplitude=self.variance.constrained_value[:,0],
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M52"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                amplitude=self.variance.constrained_value[:,0],
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M12", "OU"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                amplitude=self.variance.constrained_value[:,0],
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                length_scale=self.timescale.constrained_value[:,0])
        self.fed_kernel = fed_kernel
        self.time_kernel = time_kernel

        self.resolution = tf.convert_to_tensor(resolution, tf.int32)

    def _calculate_rays(self, X):
        """
        Given 6D coordinates, calculate the rays and ds tensors.

        :param X: float_type tf.Tensor (N, 6)
            Coordinates <time, kx, ky, kz, x, y, z>
        :return: float_type tf.Tensor (num_chains, res, N, 3), float_type tf.Tensor (num_chains, N)
            Rays and different ray lengths
        """
        # N, 1
        times = X[:, slice(0, 1, 1)]
        # N,3
        directions = X[:, slice(1, 4, 1)]
        # N,3
        antennas = X[:, slice(4, 7, 1)]

        # N
        sec = tf.reciprocal(directions[:, 2], name='secphi')

        # num_chains, N
        bsec = sec * self.b.constrained_value

        # num_chains, N
        ds = bsec / tf.cast(self.resolution - 1, float_type)
        # num_chains, N
        sm = sec[None, :] * (self.a.constrained_value - antennas[:, 2]) - 0.5 * bsec
        # num_chains, N
        sp = sm + bsec
        # sec * (self.a.constrained_value - antennas[:, 2]) + 0.5 * sec * self.b.constrained_value
        # num_chains, res, N
        s = sm[:, None, :] + ((sp - sm)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None],
                                                                   float_type)
        # num_chains, res1, N, 3
        y = antennas[None, None, :, :] + directions[None, None, :, :] * s[:, :, :, None]

        return y, ds


    def K(self, X, X2=None):
        """
        Calculate the ((D)D)TEC kernel based on the FED kernel.

        :param X: float_type, tf.Tensor (N, 7[10[13]])
            Coordinates in order (time, kx, ky, kz, x,y,z, [x0, y0, z0, [kx0, ky0, kz0]])
        :param X2:
            Second coordinates, if None then equal to X
        :return:
        """

        def bring_together(X, slices):
            out = []
            for s in slices:
                out.append(X[:,s])
            if len(out) == 1:
                return out[0]
            return tf.concat(out, axis=1)

        # difference with frozen flow is that the time for all rays is independent of ray index

        #N, 1
        times = X[:,0:1]
        N = tf.shape(X)[0]
        if self.obs_type == 'DTEC':
            X = tf.concat([bring_together(X,[slice(0,7,1)]),#i,alpha
                           bring_together(X,[slice(0,4,1), slice(7,10,1)])],#i0,alpha
                          axis=0)
        if self.obs_type == 'DDTEC':
            X = tf.concat([bring_together(X, [slice(0, 7, 1)]),  #i,alpha
                           bring_together(X, [slice(0, 4, 1), slice(7, 10, 1)]),  #i0, alpha
                           bring_together(X, [slice(0, 1, 1), slice(10, 13, 1), slice(4, 7, 1)]),  # i,alpha0
                           bring_together(X, [slice(0, 1, 1), slice(10, 13, 1), slice(7, 10, 1)]),  # i0,alpha0
                           ],
                          axis=0)
        # num_chains, res1, N', 3
        y1, ds1 = self._calculate_rays(X)
        shape1 = tf.shape(y1)
        # num_chains, res1 N', 3
        y1 = tf.reshape(y1, tf.concat([shape1[0:1], [-1, 3]], axis=0))

        if X2 is None:
            sym = True
            Np = N
            times2 = times
            y2, ds2 = y1, ds1
            shape2 = shape1
        else:
            sym = False
            # Np, 1
            times2 = X2[:, 0:1]
            Np = tf.shape(X2)[0]
            if self.obs_type == 'DTEC':
                X2 = tf.concat([bring_together(X2, [slice(0, 7, 1)]),  # i,alpha
                               bring_together(X2, [slice(0, 4, 1), slice(7, 10, 1)])],  # i0,alpha
                              axis=0)
            if self.obs_type == 'DDTEC':
                X2 = tf.concat([bring_together(X2, [slice(0, 7, 1)]),  # i,alpha
                               bring_together(X2, [slice(0, 4, 1), slice(7, 10, 1)]),  # i0, alpha
                               bring_together(X2, [slice(0, 1, 1), slice(10, 13, 1), slice(7, 10, 1)]),  # i0,alpha0
                               bring_together(X2, [slice(0, 1, 1), slice(10, 13, 1), slice(4, 7, 1)])  # i,alpha0
                               ],
                              axis=0)
            # num_chains, res2, Np', 3
            y2, ds2 = self._calculate_rays(X2)
            shape2 = tf.shape(y2)
            # num_chains, res2 Np', 3
            y2 = tf.reshape(y2, tf.concat([shape2[0:1], [-1, 3]], axis=0))

        # num_chains, N', Np'
        ds1ds2 = ds1[:, :, None] * ds2[:, None, :]

        # TODO symmetrize when possible

        # num_chains, res1 N', res2 Np'
        K_space = self.fed_kernel.matrix(y1, y2)
        shape = tf.concat([tf.shape(K_space)[0:1], shape1[1:3], shape2[1:3]], axis=0)

        # num_chains, N, Np
        K_time = self.time_kernel.matrix(times[None, :, :], times2[None, :, :])

        # num_chains, res1, N', res2, Np'
        K = tf.reshape(K_space, shape)
        # with tf.control_dependencies([tf.print(tf.shape(K))]):
        # num_chains, N,Np
        I = 0.25 * ds1ds2 * tf.add_n([K[:, 0, :, 0, :],
                                      K[:, -1, :, 0, :],
                                      K[:, 0, :, -1, :],
                                      K[:, -1, :, -1, :],
                                      2 * tf.reduce_sum(K[:, -1, :, :, :], axis=[2]),
                                      2 * tf.reduce_sum(K[:, 0, :, :, :], axis=[2]),
                                      2 * tf.reduce_sum(K[:, :, :, -1, :], axis=[1]),
                                      2 * tf.reduce_sum(K[:, :, :, 0, :], axis=[1]),
                                      4 * tf.reduce_sum(K[:, 1:-1, :, 1:-1, :], axis=[1, 3])])
        if self.obs_type == 'TEC':
            result = K_time * I
            # if sym:
            #     result = 0.5 * (tf.transpose(result, (0,2,1)) + result)
            if self.squeeze:
                return tf.squeeze(result)
            else:
                return result
        if self.obs_type == 'DTEC':
            n = 2
        if self.obs_type == 'DDTEC':
            n = 4
        out = []
        for i in range(n):
            for j in range(n):
                out.append((1. if (i + j) % 2 == 0 else -1.) * I[:, i * N:(i + 1) * N, j * Np:(j + 1) * Np])
        with tf.control_dependencies([tf.print(tf.shape(I),tf.shape(K))]):
            result = K_time * tf.add_n(out)

        # if sym:
        #     result = 0.5 * (tf.transpose(result, (0, 2, 1)) + result)
        if self.squeeze:
            return tf.squeeze(result)
        else:
            return result
