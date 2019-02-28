import tensorflow as tf
import tensorflow_probability as tfp
from .settings import float_type
from .parameters import Parameter, ScaledPositiveBijector, SphericalToCartesianBijector
from .dblquad import dblquad


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
                 a=250., b=50.,  timescale=30.,
                 fed_kernel='RBF', obs_type='TEC', squeeze=True, kernel_params={}):
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
        self.resolution = tf.convert_to_tensor(kernel_params.pop('resolution', 3), tf.int32)

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
        result = K_time * tf.add_n(out)

        # if sym:
        #     result = 0.5 * (tf.transpose(result, (0, 2, 1)) + result)
        if self.squeeze:
            return tf.squeeze(result)
        else:
            return result


class DTECIsotropicTimeGeneralODE(object):
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
                 a=250., b=50.,  timescale=30.,
                 fed_kernel='RBF', obs_type='TEC', squeeze=True, ode_type='fixed', kernel_params={}):
        if obs_type not in DTECIsotropicTimeGeneral.allowed_obs_type:
            raise ValueError(
                "{} is an invalid obs_type. Must be in {}.".format(obs_type, DTECIsotropicTimeGeneral.allowed_obs_type))
        self.obs_type = obs_type
        self.squeeze = squeeze
        self.kernel_params = kernel_params
        self.ode_type = ode_type

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

    def _calculate_ray_params(self, X):
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
        sm = sec[None, :] * (self.a.constrained_value - antennas[:, 2]) - 0.5 * bsec
        ymin = antennas[None, :, :] + directions[None, :, :] * sm[:, :, None]
        dy = directions[None, :, :] * bsec[:, :, None]

        return ymin, dy


    def K(self, X, X2=None, full_output=False):
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

        # num_chains, N', 3
        ymin1, dy1 = self._calculate_ray_params(X)

        if X2 is None:
            sym = True
            X2 = X
            Np = N
            times2 = times
            ymin2, dy2 = ymin1, dy1
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
                # num_chains, Np', 3
                ymin2, dy2 = self._calculate_ray_params(X2)

        # num_chains, N, Np
        K_time = self.time_kernel.matrix(times[None, :, :], times2[None, :, :])


        def K_func(t1,t2):
            #num_chains, N', 3
            y1 = ymin1 + t1*dy1
            #num_chains, Np', 3
            y2 = ymin2 + t2*dy2
            # num_chains, N', Np'
            K_space = self.fed_kernel.matrix(y1, y2)
            return K_space

        shape = tf.concat([tf.shape(K_time)[0:1], tf.shape(X)[0:1], tf.shape(X2)[0:1]],axis=0)

        l = tf.constant(0., float_type)
        u = tf.constant(1., float_type)
        # num_chains, N,Np
        I,info = dblquad(K_func, l, u, lambda t: l, lambda t: u, shape, ode_type=self.ode_type, **self.kernel_params)

        if self.obs_type == 'TEC':
            n = 1
            result = K_time * I
        if self.obs_type == 'DTEC':
            n = 2
        if self.obs_type == 'DDTEC':
            n = 4
        if n > 1:
            out = []
            for i in range(n):
                for j in range(n):
                    out.append((1. if (i + j) % 2 == 0 else -1.) * I[:, i * N:(i + 1) * N, j * Np:(j + 1) * Np])
            result = K_time * tf.add_n(out)
        if sym:
            result = 0.5*(tf.transpose(result,(0,2,1)) + result)
        if self.squeeze:
            if full_output:
                return tf.squeeze(result), info
            return tf.squeeze(result)
        else:
            if full_output:
                return result, info
            return result


class DTECFrozenflow(object):
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
                 a=250., b=50.,  timescale=30., velocity = [0.,0.,0.],
                 fed_kernel='RBF', obs_type='TEC', squeeze=True, kernel_params={}):
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
        if not isinstance(velocity, Parameter):
            velocity = Parameter(bijector=tfp.bijectors.Identity(), constrained_value=velocity, shape=(-1, 3))

        self.variance = variance
        self.lengthscales = lengthscales
        self.timescale = timescale
        self.a = a
        self.b = b
        self.velocity = velocity

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
        self.resolution = tf.convert_to_tensor(kernel_params.pop('resolution', 3), tf.int32)

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
        sm = sec[None, :] * (self.a.constrained_value - antennas[:, 2] + self.velocity.constrained_value[:, 2:3]*times[:,0]) - 0.5 * bsec
        # num_chains, N
        sp = sm + bsec
        # sec * (self.a.constrained_value - antennas[:, 2]) + 0.5 * sec * self.b.constrained_value
        # num_chains, res, N
        s = sm[:, None, :] + ((sp - sm)[:, None, :]) * tf.cast(tf.linspace(0., 1., self.resolution)[None, :, None],
                                                                   float_type)
        # num_chains, res1, N, 3
        y = antennas[None, None, :, :] + directions[None, None, :, :] * s[:, :, :, None] - self.velocity.constrained_value[:,None, None,:]*times[None,None, :, :]

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
            result = I
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
        result = tf.add_n(out)

        # if sym:
        #     result = 0.5 * (tf.transpose(result, (0, 2, 1)) + result)
        if self.squeeze:
            return tf.squeeze(result)
        else:
            return result
