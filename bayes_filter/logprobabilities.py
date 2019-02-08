import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .kernels import DTECFrozenFlow
from .parameters import Parameter, ScaledPositiveBijector, SphericalToCartesianBijector
from collections import namedtuple
from .settings import jitter, float_type
from .misc import diagonal_jitter

class BaseTarget(tfp.distributions.Distribution):
    def __init__(self,*args, bijectors=None, distributions=None, **kwargs):
        for b in bijectors:
            if not isinstance(b,tfp.bijectors.Bijector):
                raise ValueError("{} is not a tfp.bijectors.Bijector".format(type(b)))
        if distributions is None:
            distributions = [None for _ in bijectors]
        if len(bijectors) != len(distributions):
            raise ValueError("length of bijectors and distribution not equal {} {}".format(len(bijectors),len(distributions)))
        for d in distributions:
            if d is None:
                continue
            if not isinstance(b, tfp.distributions.Distribution):
                raise ValueError("{} is not a tfp.distributions.Distribution".format(type(d)))
        self.parameters = [Parameter(bijector=b, distribution=d) for (b,d) in zip(self.bijectors, self.distributions)]

    @property
    def bijectors(self):
        return [p.bijector for p in self.parameters]

    def unconstrained_states(self, *states):
        return [b.inverse(s) for (b,s) in zip(states, self.bijectors)]

    def constrained_states(self, *unconstrained_states):
        return [b.forward(s) for (b,s) in zip(unconstrained_states, self.bijectors)]

    def logp(self,*unconstrained_states):
        states = self.constrained_states(*unconstrained_states)
        return self._logp(*states)

class Target(BaseTarget):
    def __init__(self,*args, **kwargs):
        super(Target, self).__init__(*args, **kwargs)

    def _logp(self, *states):
        raise NotImplementedError("Subclass this.")

class DTECToGains(Target):
    DTECToGainsParams = namedtuple('DTECToGainsBijectors',
                                      ['Y_sigma', 'variance', 'lengthscales', 'a', 'b', 'velocity', 'L'])

    def __init__(self, X, X_dims, Y_real, Y_imag, freqs, bijector_params = DTECToGainsParams(Y_sigma=1., variance=1.0, lengthscales=10.0,
                 velocity=None, a=250., b=50., L=None), obs_type='DDTEC'):

        kern = DTECFrozenFlow(
            variance=bijector_params.variance,
            lengthscales=bijector_params.lengthscales,
            velocity=bijector_params.velocity,
            a=bijector_params.a,
            b=bijector_params.b,
            resolution=3,
            fed_kernel='RBF',
            obs_type=obs_type)

        K = kern.K(X,
                   X_dims,
                   X2 = None,
                   X2_dims = None)
        L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))

        bijectors = DTECToGains.DTECToGainsParams(
            ScaledPositiveBijector(bijector_params.Y_sigma), ScaledPositiveBijector(bijector_params.variance),
            ScaledPositiveBijector(bijector_params.lengthscales), ScaledPositiveBijector(bijector_params.a),
            ScaledPositiveBijector(bijector_params.b), SphericalToCartesianBijector(),
            tfp.distributions.TransformedDistribution(),
            tfp.bijectors.Affine(scale_tril=bijector_params.L))

        super(DTECToGains, self).__init__(bijectors=bijectors, distributions=None)
        self.X = X
        self.X_dims = X_dims
        self.Y_real = Y_real
        self.Y_imag = Y_imag
        self.freqs = freqs

    def forward_equation(self, dtec):
        """
        Calculate real and imaginary parts of gains from dtec.

        :param dtec: float_type, Tensor [b0,...,bB]
            The DTECs
        :return: float_type, Tensor [b0,...,bB,Nf]
            Real part
        :return: float_type, Tensor [b0,...,bB,Nf]
            Imag part
        """
        #Nf
        invfreqs = -8.448e9*tf.reciprocal(self.freqs)
        #..., Nf
        phase = dtec[..., None] * invfreqs
        real_part = tf.cos(phase)
        imag_part = tf.sin(phase)
        return real_part, imag_part

    def _logp(self, y_sigma, variance, lengthscales, a, b, velocity, dtec):
        """
        Calculate the log probability of the gains given a model.

        :param y_sigma:
        :param variance:
        :param lengthscales:
        :param a:
        :param b:
        :param velocity:
        :param dtec:
        :return:
        """

        g_real, g_imag = self.forward_equation(dtec)
        likelihood_real = tfp.distributions.Normal(loc=g_real, scale=y_sigma)
        likelihood_imag = tfp.distributions.Normal(loc=g_imag, scale=y_sigma)
        logp = tf.reduce_sum(likelihood_real.log_prob(self.Y_real[None, :, :]), axis=[1, 2])
        likelihood = tfp.distributions.Normal(loc=Y_model_cos, scale=y_sigma)
        logp = tf.reduce_sum(likelihood.log_prob(Y[None, :, :]), axis=[1, 2]) + tf.reduce_sum(
        likelihood_cos.log_prob(Y_cos[None, :, :]),


        kern = DTECFrozenFlow(
            variance=bijector_params.variance,
            lengthscales=bijector_params.lengthscales,
            velocity=bijector_params.velocity,
            a=bijector_params.a,
            b=bijector_params.b,
            resolution=3,
            fed_kernel='RBF',
            obs_type=obs_type)


