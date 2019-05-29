import tensorflow_probability as tfp
import tensorflow as tf
from collections import namedtuple

from .settings import float_type
from .parameters import Parameter, ScaledLowerBoundedBijector
from .misc import safe_cholesky
from .kernels import DTECIsotropicTimeGeneral

class Process(object):
    def __init__(self, bijectors=None, distributions=None, unconstrained_values=None):
        self._setup = False
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
            if not isinstance(d, tfp.distributions.Distribution):
                raise ValueError("{} is not a tfp.distributions.Distribution".format(type(d)))
        if unconstrained_values is not None:
            self.parameters = [Parameter(bijector=b, distribution=d, unconstrained_value=v) for (b,d,v) in zip(bijectors, distributions,unconstrained_values)]
        else:
            self.parameters = [Parameter(bijector=b, distribution=d) for (b, d) in
                           zip(bijectors, distributions)]

    @property
    def get_default_parameters(self):
        raise NotImplementedError("must subclass")

    @property
    def bijectors(self):
        return [p.bijector for p in self.parameters]

    def log_prob(self, *unconstrained_states):
        """
        If the joint distribution of data Y and params M is P(Y,M) then this represents,
            P(M | Y)
        and may be unnormalised.

        :param states: List(tf.Tensor)
            List of starts where first dimension is represents independent realizations.
            First dimension of each state is size `num_chains`.
        :return: float_type, tf.Tensor, [num_chains]
            The log probability of each chain.
        """
        raise NotImplementedError("Subclass this.")

    @property
    def Params(self):
        """A namedtuple representing hyperparams of this Target."""
        return self._Params

    @property
    def _Params(self):
        """Implementation of Params."""
        raise NotImplementedError("Subclass this.")

    def unstack_state(self,state):
        """
        Unstacks a hyperparam state that is packed into a single tensor.

        :param state:
        :return: namedtuple of tf.Tensor
            tuple of named states
        """
        return self.Params(
            *[tf.reshape(state[..., i:i + 1], (-1, 1)) for i in
              range(len(self.parameters))])

    def stack_state(self,state):
        """
        Stacks a namedtuple into tensor.

        :param state: namedtuple
            The hyperparam state
        :return: tf.Tensor
            The packed hyperparams
        """
        return tf.concat(list(state), axis=-1)

    def unconstrained_state(self, state):
        """
        Gets the unconstrained state from namedtuple of constrained states

        :param state: namedtuple
            constrained states in namedtuple
        :return: namedtuple
            Unconstrained states in namedtuple
        """
        return self.Params(*[self.parameters[i].bijector.inverse(state[i]) for i in range(len(self.parameters))])


    def constrained_state(self, state):
        """
        Gets the constrained state from typle of unconstrained states

        :param state: namedtuple
            The unconstrained states in namedtuple
        :return:namedtuple
            The constrained states in namedtuple
        """
        return self.Params(*[self.parameters[i].bijector.forward(state[i]) for i in range(len(self.parameters))])

    def init_variables(self, **initial_hyperparams):
        """
        Creates the initial hyperparam tensor.

        :param initial_hyperparams: initial hyperparams as keywords
        :return: tf.Tensor
            The stacked unconstrained initial hyperparams.
        """
        #makes a namedtuple
        initial_hyperparams = self.get_default_parameters._replace(**initial_hyperparams)
        initial_hyperparams = initial_hyperparams._replace(
            **{k:tf.convert_to_tensor(v, dtype=float_type) for k,v in initial_hyperparams._asdict().items()})

        bijectors = self.bijectors

        unconstrained_vars = self.Params(
            *[tf.reshape(b.inverse(v), (-1,1)) for (b, v) in zip(bijectors, initial_hyperparams)])
        variables = self.stack_state(unconstrained_vars)#tf.get_variable('state_vars', initializer=self.stack_state(unconstrained_vars))

        return variables

    @property
    def setup(self):
        return self._setup

class DTECProcess(Process):
    @property
    def _Params(self):
        return namedtuple('DTECToGainsParams',
                          ['variance', 'lengthscales', 'a', 'b', 'timescale'])

    def __init__(self,
                 initial_hyperparams={},
                 variables=None):
        """
        Creates an instance of the target distribution for complex gains modelled by DTEC.

        :param initial_hyperparams: dict
            The initial parameters for the DTEC process
        :param variables: float tf.Tensor or None
            If None then will initialise variables from initial_hyperparams or the default.
        """
        self._setup = False

        bijectors = self.Params(
            variance=ScaledLowerBoundedBijector(1e-3, 1.),
            lengthscales=ScaledLowerBoundedBijector(3., 15.),
            a=ScaledLowerBoundedBijector(100., 250.),
            b=ScaledLowerBoundedBijector(10., 100.),
            timescale=ScaledLowerBoundedBijector(10., 50.))

        super(DTECProcess, self).__init__(bijectors=bijectors, distributions=None, unconstrained_values=None)

        if variables is None:
            variables = self.init_variables(**initial_hyperparams)

        self.hyperparams = variables
        self.constrained_hyperparams = self.constrained_state(self.unstack_state(self.hyperparams))

    @property
    def get_default_parameters(self):
        return self.Params(variance=1., lengthscales=15., a=250., b=100., timescale=100.)

    def setup_process(self,X, Xstar, fed_kernel = 'RBF', obs_type='DDTEC',
                     kernel_params={}, recalculate_prior = True,
                     L=None, m=None):
        recalculate_prior = tf.convert_to_tensor(recalculate_prior, tf.bool)
        self.obs_type = obs_type
        self.fed_kernel = fed_kernel
        # N, ndims
        self.X = X
        self.N = tf.shape(self.X)[0]
        if Xstar is not None:
            # Ns, ndims
            self.Xstar = Xstar
            self.Ns = tf.shape(self.Xstar)[0]
            self.Xconcat = tf.concat([self.X, self.Xstar], axis=0)
        else:
            self.Xstar = None
            self.Ns = 0
            self.Xconcat = self.X
        self.Nh = tf.shape(self.Xconcat)[0]

        kern = DTECIsotropicTimeGeneral(
            variance=tf.ones_like(self.constrained_hyperparams.variance), # <---- bayesian replacement
            lengthscales=self.constrained_hyperparams.lengthscales,
            timescale=self.constrained_hyperparams.timescale,
            a=self.constrained_hyperparams.a,
            b=self.constrained_hyperparams.b,
            fed_kernel=self.fed_kernel,
            obs_type=self.obs_type,
            squeeze=True,
            kernel_params=kernel_params)

        # (batch), N+Ns, N+Ns
        new_K = kern.K(self.Xconcat)
        # multiply by 1000 m/km to get right units.
        new_L = safe_cholesky(new_K)
        new_m = tf.zeros([self.Nh], dtype=float_type)
        L = L if L is not None else tf.eye(tf.shape(self.Xconcat)[0], dtype=float_type)
        m = m if m is not None else tf.zeros(tf.shape(self.Xconcat)[0:1], dtype=float_type)



        # N+Ns, N+Ns
        self.L = tf.cond(recalculate_prior, lambda: new_L, lambda: L)

        # N+Ns
        self.m = tf.cond(recalculate_prior, lambda: new_m, lambda: m)

        self._setup = True