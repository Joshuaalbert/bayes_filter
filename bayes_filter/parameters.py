import tensorflow_probability as tfp
import tensorflow as tf
from .settings import float_type

class ConstrainedBijector(tfp.bijectors.Chain):
    def __init__(self,a,b,validate_args=False):
        """
            Create a bijector that constrains the value to (a,b)

            :param a: float scalar
                Lower limit
            :param b: float scalar
                Uppoer limit
            :return: tfp.bijectors.Bijector
                The chained affine and sigmoid that achieves the constraint.
            """
        self.a = tf.convert_to_tensor(a, float_type)
        self.b = tf.convert_to_tensor(b, float_type)
        super(ConstrainedBijector, self).__init__(
            [tfp.bijectors.AffineScalar(shift=self.a, scale=(self.b-self.a)),
             tfp.bijectors.Sigmoid()],validate_args=validate_args, name='constrained')


class SphericalToCartesianBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='spherical'):
        """
        A bijector that converts spherical to Cartesian coordinates.

        :param validate_args: bool
        :param name: optional
        """
        super(SphericalToCartesianBijector, self).__init__(validate_args=validate_args,forward_min_event_ndims=0,
                                                inverse_min_event_ndims=0, is_constant_jacobian=False,dtype=float_type,
                                                name=name)

    def _forward(self,sph):
        """
        Convert spherical to cartesian along last dimension.

        :param sph: float_type, Tensor, [b0,...,bB, 3]
            Spherical coordinates (radial, azimuthal, polar)
        :return: float_type, Tensor, [b0, ..., bB, 3]
            Cartesian coordinates
        """
        r = sph[...,0:1]
        theta = sph[...,1]
        phi = sph[...,2]
        sinphi = tf.sin(phi)
        x = tf.cos(theta) * sinphi
        y = tf.sin(theta) * sinphi
        z = tf.cos(phi)
        xyz = r * tf.stack([x,y,z],axis=-1)
        return xyz

    def _inverse(self,car):
        """
        Convert cartesian to sphereical along last dimension

        :param car: float_type, Tensor, [b0,...,bB, 3]
            The cartesian coordinates
        :return: float_type, Tensor, [b0, ... bB, 3]
            The sphereical coordinates
        """
        x = car[...,0]
        y = car[...,1]
        z = car[...,2]
        r = tf.sqrt(tf.square(x) + tf.square(y) + tf.square(z))
        theta = tf.atan2(y, x)
        phi = tf.acos(z/r)
        sph = tf.stack([r,theta,phi],axis=-1)
        return sph

    def _forward_log_det_jacobian(self,sph):
        """
        The forward jacobian of the transformation.
        log |d y(x)/ d x|
        :param sph: float_type, Tensor, [b0, ...,bB, 3]
            The x of y(x)
        :return: float_type, Tensor , [b0,...,bB]
            The log det Jacobian of y(x)
        """
        r = sph[..., 0]
        theta = sph[..., 1]
        phi = sph[..., 2]
        return 2*tf.log(r) + tf.log(tf.sin(phi))

    def _inverse_log_det_jacobian(self,car):
        """
        The inverse jacobian of the transformation.
        log |d x(y)/ d y|
        :param car: float_type, Tensor, [b0, ...,bB, 3]
            The y of y(x)
        :return: float_type, Tensor , [b0,...,bB]
            The log det Jacobian of x(y)
        """
        return -self._forward_log_det_jacobian(self._inverse(car))


class ScaledPositiveBijector(tfp.bijectors.Chain):
    def __init__(self, scale=1., validate_args=False):
        self.scale = tf.convert_to_tensor(scale, float_type)
        super(ScaledPositiveBijector, self).__init__(
            [tfp.bijectors.AffineScalar(scale=self.scale),tfp.bijectors.Softplus()], validate_args=validate_args,
            name='scaled_positive_bijector')

class ScaledLowerBoundedBijector(tfp.bijectors.Chain):
    def __init__(self, lower_bound=0.,scale=1.,  validate_args=False):
        self.scale = tf.convert_to_tensor(scale, float_type)
        self.lower_bound = tf.convert_to_tensor(lower_bound, float_type)
        super(ScaledLowerBoundedBijector, self).__init__(
            [tfp.bijectors.AffineScalar(shift=self.lower_bound if self.lower_bound != 0. else None, scale=self.scale),
             tfp.bijectors.Softplus()], validate_args=validate_args,
            name='scaled_positive_bijector')


class ScaledBijector(tfp.bijectors.Chain):
    def __init__(self, scale=1., validate_args=False):
        self.scale = tf.convert_to_tensor(scale, float_type)
        super(ScaledBijector, self).__init__(
            [tfp.bijectors.AffineScalar(scale=self.scale)], validate_args=validate_args,
            name='scaled_bijector')


class Parameter(object):
    def __init__(self, unconstrained_value=None, constrained_value=None, bijector=None, distribution=None,
                 dtype=float_type, shape = None):
        """
        Builds a parameter with a bijector and distribution. If the unconstrained value is X, then the constrained
        value is Y = bijector.forward(X), and `distribution` models p(Y) then unconstrained_prior models p(X).

        :param unconstrained_value: tf.Tensor, optional
            If not None, then gives the unconstrained parameter value.
        :param constrained_value: tf.Tensor
            If not None, then gives the constrained parameter value
        :param bijector: tfp.bijectors.Bijector
            If None then Identity, else gives relation Y=g(X)
        :param distribution: tfp.distributions.Distribution
            Gives the distribution of p(Y)
        :param dtype: tf.dtype
            Gives the dtype of the parameter
        :raise ValueError:
            if both unconstrained_value and constrained_value given.
        """

        if bijector is None:
            bijector = tfp.bijectors.Identity()
        self.bijector = bijector
        if unconstrained_value is not None and constrained_value is not None:
            raise ValueError("Only one of constrained_value and unconstrained_value may be given.")
        # if unconstrained_value is None and constrained_value is None:
        #     raise ValueError("At least one of contrained_value and unconstrained_value must be given.")
        self.unconstrained_value = None
        self.constrained_value = None
        if unconstrained_value is not None:
            self.unconstrained_value = tf.convert_to_tensor(unconstrained_value, dtype)
            self.constrained_value = self.bijector.forward(self.unconstrained_value)
        if constrained_value is not None:
            self.constrained_value = tf.convert_to_tensor(constrained_value, dtype)
            self.unconstrained_value = self.bijector.inverse(self.constrained_value)
        if shape is not None:
            self.constrained_value = tf.reshape(self.constrained_value, shape)
            self.unconstrained_value = tf.reshape(self.unconstrained_value, shape)
        if distribution is None:
            distribution = tfp.distributions.Uniform(
                low = tf.constant(0.,dtype=float_type), high = tf.constant(1.,dtype=float_type))
        self.unconstrained_prior = tfp.distributions.TransformedDistribution(
            distribution=distribution, bijector=tfp.bijectors.Invert(self.bijector))
        self.constrained_prior = distribution



