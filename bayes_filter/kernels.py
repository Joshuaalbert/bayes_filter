import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from .settings import float_type
from .parameters import Parameter, ScaledPositiveBijector, SphericalToCartesianBijector
from .dblquad import dblquad

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util


def _validate_arg_if_not_none(arg, assertion, validate_args):
  if arg is None:
    return arg
  with tf.control_dependencies([assertion(arg)] if validate_args else []):
    result = tf.identity(arg)
  return result

class Histogram(tfp.positive_semidefinite_kernels.PositiveSemidefiniteKernel):
    def __init__(self,heights,edgescales=None, lengthscales=None,feature_ndims=1, validate_args=False,name='Histogram'):
        """Construct an Histogram kernel instance.

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
                    lengthscales, dtype=dtype,name='lengthscales')
                lengthscales = tf.nn.top_k(lengthscales, k = tf.shape(lengthscales)[-1], sorted=True)[0]
                edgescales = tf.reciprocal(lengthscales)

            if edgescales is not None:
                edgescales = tf.convert_to_tensor(
                    edgescales, dtype=dtype, name='edgescales')
                edgescales = tf.reverse(tf.nn.top_k(edgescales,k=tf.shape(edgescales)[-1],sorted=True)[0], axis=[-1])
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
        #x1 = B,Np,D -> B,Np,1,D
        #x2 = B,N,D -> B,1,N,D
        #B, Np,N
        with tf.control_dependencies([tf.assert_equal(tf.shape(self.heights)[-1]+1, tf.shape(self.edgescales)[-1])]):
            norm = util.sqrt_with_finite_grads(util.sum_rightmost_ndims_preserving_shape(
                tf.squared_difference(x1, x2), self.feature_ndims))
        #B(1),1,Np,N
        norm = tf.expand_dims(norm,-(param_expansion_ndims + 1))

        #B(1), H+1, 1, 1
        edgescales = util.pad_shape_right_with_ones(
            self.edgescales, ndims=param_expansion_ndims)
        norm *= edgescales
        norm *= 2*np.pi

        zeros = tf.zeros(tf.shape(self.heights)[:-1],dtype=self.heights.dtype)[...,None]
        # B(1),1+H+1
        heights = tf.concat([zeros, self.heights, zeros],axis=-1)
        # B(1), H+1
        dheights = heights[..., :-1] - heights[..., 1:]
        #B(1), H+1, 1, 1
        dheights = util.pad_shape_right_with_ones(
            dheights, ndims=param_expansion_ndims)
        #B(1), H+1, 1, 1
        dheights *= edgescales
        def _sinc(x):
            return tf.sin(x)*tf.reciprocal(x)
        #B(1), H+1, N, Np
        sincs = tf.where(tf.less(norm, tf.constant(1e-15,dtype=norm.dtype)), tf.ones_like(norm), _sinc(norm))
        #B(1), H+1, N, Np
        result = dheights * sincs
        #B(1), N,Np
        return tf.reduce_sum(result,axis=-3)


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

    def __init__(self, variance=1., lengthscales=10.0,
                 a=250., b=50.,  timescale=30., ref_location=[0.,0.,0.],
                 fed_kernel='RBF', obs_type='TEC', squeeze=True, kernel_params={}):
        if obs_type not in DTECIsotropicTimeGeneral.allowed_obs_type:
            raise ValueError(
                "{} is an invalid obs_type. Must be in {}.".format(obs_type, DTECIsotropicTimeGeneral.allowed_obs_type))
        self.obs_type = obs_type
        self.squeeze = squeeze

        if not isinstance(variance, Parameter):
            #1e3*1e-16*1e9 = 1e-4
            variance = Parameter(bijector=ScaledPositiveBijector(1.), constrained_value=variance, shape=(-1,1))
        if not isinstance(lengthscales, Parameter):
            lengthscales = Parameter(bijector=ScaledPositiveBijector(10.), constrained_value=lengthscales, shape=(-1,1))
        if not isinstance(timescale, Parameter):
            timescale = Parameter(bijector=ScaledPositiveBijector(50.), constrained_value=timescale, shape=(-1,1))
        if not isinstance(a, Parameter):
            a = Parameter(bijector=ScaledPositiveBijector(250.), constrained_value=a, shape=(-1,1))
        if not isinstance(b, Parameter):
            b = Parameter(bijector=ScaledPositiveBijector(100.), constrained_value=b, shape=(-1,1))

        self.ref_location = tf.convert_to_tensor(ref_location,float_type)

        self.variance = variance
        self.lengthscales = lengthscales
        self.timescale = timescale
        self.a = a
        self.b = b

        if fed_kernel in ["RBF", "SE"]:
            fed_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                amplitude=None,
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M32"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                amplitude=None,
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M52"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                amplitude=None,
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M12", "OU"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                amplitude=None,
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
        sm = sec[None, :] * (self.a.constrained_value  + self.ref_location[2] - antennas[:, 2]) - 0.5 * bsec
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
            result = 0.5 * (tf.transpose(result, (0, 2, 1)) + result)
        #num_chains, N, Np
        result = self.variance.constrained_value[:,0,None,None]*result
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
                 a=250., b=50.,  timescale=30., ref_location=[0.,0.,0.],
                 fed_kernel='RBF', obs_type='TEC', squeeze=True, ode_type='fixed', kernel_params={}):
        if obs_type not in DTECIsotropicTimeGeneral.allowed_obs_type:
            raise ValueError(
                "{} is an invalid obs_type. Must be in {}.".format(obs_type, DTECIsotropicTimeGeneral.allowed_obs_type))
        self.obs_type = obs_type
        self.squeeze = squeeze
        self.kernel_params = kernel_params
        self.ode_type = ode_type

        if not isinstance(variance, Parameter):
            variance = Parameter(bijector=ScaledPositiveBijector(1.), constrained_value=variance, shape=(-1,1))
        if not isinstance(lengthscales, Parameter):
            lengthscales = Parameter(bijector=ScaledPositiveBijector(10.), constrained_value=lengthscales, shape=(-1,1))
        if not isinstance(timescale, Parameter):
            timescale = Parameter(bijector=ScaledPositiveBijector(50.), constrained_value=timescale, shape=(-1,1))
        if not isinstance(a, Parameter):
            a = Parameter(bijector=ScaledPositiveBijector(250.), constrained_value=a, shape=(-1,1))
        if not isinstance(b, Parameter):
            b = Parameter(bijector=ScaledPositiveBijector(100.), constrained_value=b, shape=(-1,1))

        self.ref_location = tf.convert_to_tensor(ref_location, float_type)
        self.variance = variance
        self.lengthscales = lengthscales
        self.timescale = timescale
        self.a = a
        self.b = b

        if fed_kernel in ["RBF", "SE"]:
            fed_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                amplitude=None,
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M32"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                amplitude=None,
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.MaternThreeHalves(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M52"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                amplitude=None,
                length_scale=self.lengthscales.constrained_value[:,0])
            time_kernel = tfp.positive_semidefinite_kernels.MaternFiveHalves(
                length_scale=self.timescale.constrained_value[:,0])
        if fed_kernel in ["M12", "OU"]:
            fed_kernel = tfp.positive_semidefinite_kernels.MaternOneHalf(
                amplitude=None,
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
        :return: float_type tf.Tensor (num_chains, res, N, 3), float_type tf.Tensor (num_chains, N), float_type tf.Tensor (num_chains, N)
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
        sm = sec[None, :] * (self.a.constrained_value + self.ref_location[2]  - antennas[:, 2]) - 0.5 * bsec
        ymin = antennas[None, :, :] + directions[None, :, :] * sm[:, :, None]
        dy = directions[None, :, :] * bsec[:, :, None]

        return ymin, dy, bsec


    def K(self, X, X2=None, full_output=False):
        """
        Calculate the ((D)D)TEC kernel based on the FED kernel.
        int_s1-^s1+ int_s2-^s2+ K_fed(x1+k1*s1, x2+k2*s2) ds1 ds2
        = int_s1-^s1+ int_s2-^s2+ K_fed(x1+k1*s1- + k1*(s1+ - s1-)*t1, x2+k2*s2- + k2*(s2+ - s2-)*t2) (s1+ - s1-)*dt1 (s2+ - s2-)*dt2
        x2+k2*s2 = x2 + k2*s2- + k2*(s2+ - s2-)*t2
        -> s2 = s2- + (s2+ - s2-)*t2
        -> ds2 = (s2+ - s2-)*dt2

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
        ymin1, dy1, bsec1 = self._calculate_ray_params(X)

        if X2 is None:
            sym = True
            X2 = X
            Np = N
            times2 = times
            ymin2, dy2, bsec2 = ymin1, dy1, bsec1
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
                               ], axis=0)
            ymin2, dy2, bsec2 = self._calculate_ray_params(X2)

        jac = bsec1[:,:,None]*bsec2[:,None,:]

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
        I *= jac

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
        # num_chains, N, Np
        result = self.variance.constrained_value[:, 0, None, None] * result
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
