import tensorflow as tf
import tensorflow_probability as tfp
from .kernels import DTECIsotropicTimeGeneral
from .parameters import Parameter, ScaledPositiveBijector, ConstrainedBijector, ScaledLowerBoundedBijector
from collections import namedtuple
from .misc import diagonal_jitter, log_normal_solve_fwhm, K_parts
from .settings import float_type

def constrained_scaled_positive(a,b,scale):
    return tfp.bijectors.Chain([ConstrainedBijector(a,b),ScaledPositiveBijector(scale)])

class Target(object):
    def __init__(self, bijectors=None, distributions=None, unconstrained_values=None):
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
    def bijectors(self):
        return [p.bijector for p in self.parameters]

    def unconstrained_states(self, *states):
        # with tf.control_dependencies([tf.print("constrained_states -> ",*[(tf.shape(s),s) for s in states])]):
        return [b.inverse(s) for (s,b) in zip(states, self.bijectors)]

    def constrained_states(self, *unconstrained_states):
        # with tf.control_dependencies([tf.print("unconstrained_states -> ",*[(tf.shape(s),s) for s in unconstrained_states])]):
        return [b.forward(s) for (s,b) in zip(unconstrained_states, self.bijectors)]

    def logp(self,*unconstrained_states):
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


class DTECToGains(Target):
    DTECToGainsParams = namedtuple('DTECToGainsParams',
                                      ['y_sigma', 'variance', 'lengthscales', 'a', 'b', 'timescale'])

    def __init__(self, X, Xstar, Y_real, Y_imag, freqs,
                 y_sigma=0.2, variance=0.07, lengthscales=10.0,
                 a=250., b=50., timescale=30.,  fed_kernel = 'RBF', obs_type='DDTEC', num_chains=1, ss=None):

        self.obs_type = obs_type
        self.fed_kernel = fed_kernel
        self.num_chains = num_chains
        self.ss = ss

        kern = DTECIsotropicTimeGeneral(
            variance=variance,
            lengthscales=lengthscales,
            timescale=timescale,
            a=a,
            b=b,
            resolution=3,
            fed_kernel=self.fed_kernel,
            obs_type=self.obs_type,
            squeeze=True)



        bijectors = [ScaledLowerBoundedBijector(1e-2,y_sigma),
                     ScaledLowerBoundedBijector(1e-2,variance),
                     ScaledLowerBoundedBijector(3., lengthscales),
                     ScaledLowerBoundedBijector(100.,a),
                     ScaledLowerBoundedBijector(30.,b),
                     ScaledLowerBoundedBijector(10.,timescale)]
        distributions = [
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(1e-2, 10., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(1e-4, 10., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(1., 40., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(100, 1000., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(10., 200., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(10., 100., 0.5))
        ]

        super(DTECToGains, self).__init__(bijectors=bijectors, distributions=distributions)
        #N, ndims
        self.X = X
        self.N = tf.shape(self.X)[0]
        #Ns, ndims
        self.Xstar = Xstar
        self.Ns = tf.shape(self.Xstar)[0]
        self.Xconcat = tf.concat([self.X, self.Xstar],axis=0)
        self.Nh = tf.shape(self.Xconcat)[0]
        #N, 1
        self.Y_real = Y_real
        #N, 1
        self.Y_imag = Y_imag
        #Nf
        self.freqs = freqs


        # N+Ns, N+Ns
        K = kern.K(self.Xconcat)
        # with tf.control_dependencies([tf.print("asserting initial K finite"), tf.assert_equal(tf.reduce_all(tf.is_finite(K)), True)]):
        # N+Ns, N+Ns
        self.L = tf.cholesky(K + diagonal_jitter(self.Nh))

    def transform_samples(self, y_sigma, variance, lengthscales, a, b, timescale, dtec):
        """
        Calculate transformed samples.

        :param y_sigma: float_type tf.Tensor [S, num_chains]
        :param variance: float_type tf.Tensor [S, num_chains]
        :param lengthscales: float_type tf.Tensor [S, num_chains]
        :param a: float_type tf.Tensor [S, num_chains]
        :param b: float_type tf.Tensor [S, num_chains]
        :param timescale: float_type tf.Tensor [S, num_chains]
        :param dtec: float_type tf.Tensor [S, num_chains, M]
        :return: the unconstrained parameters in a list
        """

        state = DTECToGains.DTECToGainsParams(
            *self.constrained_states(y_sigma, variance, lengthscales, a, b, timescale))

        kern = DTECIsotropicTimeGeneral(
            variance=state.variance,
            lengthscales=state.lengthscales,
            timescale=state.timescale,
            a=state.a,
            b=state.b,
            resolution=3,
            fed_kernel=self.fed_kernel,
            obs_type=self.obs_type,
            squeeze=False)

        # num_chains, N+Ns, N+Ns
        K = kern.K(self.Xconcat)
        # with tf.control_dependencies([tf.print("asserting final transform K finite"), tf.assert_equal(tf.reduce_all(tf.is_finite(K)), True)]):

        # num_chains, N+Ns, N+Ns
        L = tf.cholesky(K + diagonal_jitter(self.Nh))

        # transform
        # num_chains, N+Ns
        dtec_transformed = tf.matmul(L, dtec[:, :, None])[:, :, 0]

        return state, dtec_transformed


    def get_initial_point(self, y_sigma, variance, lengthscales, a, b, timescale, dtec):
        """
        Get an initial point from constrained values

        :param y_sigma: float_type tf.Tensor []
        :param variance: float_type tf.Tensor []
        :param lengthscales: float_type tf.Tensor []
        :param a: float_type tf.Tensor []
        :param b: float_type tf.Tensor []
        :param timescale: float_type tf.Tensor []
        :param dtec: float_type tf.Tensor [M]
        :return: the unconstrained parameters in a list
        """
        # with tf.control_dependencies([tf.print(tf.shape(self.L), tf.shape(dtec))]):
        return self.unconstrained_states(y_sigma, variance, lengthscales, a, b, timescale) \
                   + [tf.linalg.triangular_solve(self.L, dtec[:, None])[:,0]]

    def forward_equation(self, dtec):
        """
        Calculate real and imaginary parts of gains from dtec.

        :param dtec: float_type, Tensor [b0,...,bB]
            The DTECs
        :return: float_type, Tensor [b0,...,bB,Nf]
            Real part
        :return: float_type, Tensor [b0,...,bB,Nf]
            Imag part
        TODO how to specify multiple returns in a tuple
        """
        #Nf
        invfreqs = -8.448e9*tf.reciprocal(self.freqs)
        #..., Nf
        phase = dtec[..., None] * invfreqs
        real_part = tf.cos(phase)
        imag_part = tf.sin(phase)
        return real_part, imag_part

    def logp(self, y_sigma, variance, lengthscales, a, b, timescale, dtec):
        """
        Calculate the log probability of the gains given a model.

        :param y_sigma: float_type, tf.Tensor, [num_chains]
            The uncertainty of gain measurements.
        :param variance: float_type, tf.Tensor, [num_chains]
            The variance of FED.
        :param lengthscales: float_type, tf.Tensor, [num_chains]
            The lengthscales of FED (isotropic)
        :param a: float_type, tf.Tensor, [num_chains]
            The mean height of ionosphere layer.
        :param b: float_type, tf.Tensor, [num_chains]
            The mean weidth of the ionosphere layer.
        :param timescale: float_type, tf.Tensor, [num_chains]
            The timescale of the FED layer variability
        :param dtec: float_type, tf.Tensor, [num_chains, N+Ns]
            The differential TEC that model the gains.
        :return: float_type, tf.Tensor, [num_chains]
            The log-probability of the data given model.
        """

        state = DTECToGains.DTECToGainsParams(*self.constrained_states(y_sigma, variance, lengthscales, a, b, timescale))

        kern = DTECIsotropicTimeGeneral(
            variance=state.variance,
            lengthscales=state.lengthscales,
            timescale=state.timescale,
            a=state.a,
            b=state.b,
            resolution=3,
            fed_kernel=self.fed_kernel,
            obs_type=self.obs_type,
            squeeze=False)

        # num_chains, N+Ns, N+Ns
        K = kern.K(self.Xconcat)
        # num_chains, N+Ns, N+Ns
        with tf.control_dependencies(
                [tf.assert_equal(tf.is_finite(K), True),
                 tf.print(*[(n, getattr(state,n)) for n in state._fields],*self.ss)]):
            L = tf.cholesky(K + diagonal_jitter(self.Nh))

        # transform
        # num_chains, N+Ns
        dtec_transformed = tf.matmul(L, dtec[:, :, None])[:,:,0]

        #marginal
        #num_chains, N
        dtec_marginal = dtec_transformed[:, :self.N]
        # num_chains, N, Nf
        g_real, g_imag = self.forward_equation(dtec_marginal)
        likelihood_real = tfp.distributions.Laplace(loc=g_real, scale=state.y_sigma[:, :, None])
        likelihood_imag = tfp.distributions.Laplace(loc=g_imag, scale=state.y_sigma[:, :, None])
        # with tf.control_dependencies([tf.print('logp',
        #                                        ('g_real', tf.shape(g_real), g_real),
        #                                        ('y_sigma', tf.shape(y_sigma), y_sigma),
        #                                        ('Y_real', tf.shape(self.Y_real), self.Y_real))]):
        # num_chains
        logp = tf.reduce_sum(likelihood_real.log_prob(self.Y_real[None, :, :]), axis=[1, 2]) + \
               tf.reduce_sum(likelihood_imag.log_prob(self.Y_imag[None, :, :]), axis=[1, 2])

        # dtec_prior = tfp.distributions.MultivariateNormalDiag(loc=None,scale_identity_multiplier=None)
        # num_chains

        logdet = tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=1)
        # dtec_logp = dtec_prior.log_prob(dtec) - logdet
        dtec_logp = -0.5*tf.reduce_sum(tf.square(dtec),axis=1) - logdet # - 0.5*(self.N+self.Ns)*np.log(2*np.pi)


        res = logp + dtec_logp# + sum([p.constrained_prior.log_prob(s) for (p,s) in zip(self.parameters, state)])

        res.set_shape(tf.TensorShape([self.num_chains]))
        # with tf.control_dependencies(
        #         [tf.print("logp",tf.shape(res), res)]):
        return tf.identity(res)



class DTECToGainsSAEM(Target):
    DTECToGainsParams = namedtuple('DTECToGainsParams',
                                      ['y_sigma', 'variance', 'lengthscales', 'a', 'b', 'timescale'])

    def __init__(self, X, Xstar, Y_real, Y_imag, freqs,
                 y_sigma=0.2, variance=0.07, lengthscales=10.0,
                 a=250., b=50., timescale=30.,  fed_kernel = 'RBF', obs_type='DDTEC', num_chains=1, variables=None, initialize=True):

        self.obs_type = obs_type
        self.fed_kernel = fed_kernel
        self.num_chains = num_chains

        initial_values = DTECToGainsSAEM.DTECToGainsParams(
            tf.convert_to_tensor(y_sigma, dtype=float_type), tf.convert_to_tensor(variance, dtype=float_type),
            tf.convert_to_tensor(lengthscales, dtype=float_type), tf.convert_to_tensor(a, dtype=float_type),
            tf.convert_to_tensor(b, dtype=float_type), tf.convert_to_tensor(timescale, dtype=float_type))

        bijectors = DTECToGainsSAEM.DTECToGainsParams(
            ScaledLowerBoundedBijector(1e-2,y_sigma),
            ScaledLowerBoundedBijector(1e-2,variance),
            ScaledLowerBoundedBijector(3., lengthscales),
            ScaledLowerBoundedBijector(100.,a),
            ScaledLowerBoundedBijector(30.,b),
            ScaledLowerBoundedBijector(10.,timescale))

        distributions = DTECToGainsSAEM.DTECToGainsParams(
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(1e-2, 10., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(1e-4, 10., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(1., 40., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(100, 1000., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(10., 200., 0.5)),
            tfp.distributions.LogNormal(*log_normal_solve_fwhm(10., 100., 0.5)))

        constrained_vars_split = tf.stack([b.inverse(v) for (b,v) in zip(bijectors, initial_values)],axis=0)

        if variables is None:
            variables = tf.get_variable('state_vars', initializer=constrained_vars_split)
        self.variables = variables

        self.variables_split = DTECToGainsSAEM.DTECToGainsParams(*[tf.reshape(self.variables[i:i+1], (-1, 1)) for i in range(len(bijectors))])

        super(DTECToGainsSAEM, self).__init__(bijectors=bijectors, distributions=distributions, unconstrained_values=self.variables_split)

        self.state = DTECToGainsSAEM.DTECToGainsParams(*self.parameters)

        if initialize:
            #N, ndims
            self.X = X
            self.N = tf.shape(self.X)[0]
            #Ns, ndims
            self.Xstar = Xstar
            self.Ns = tf.shape(self.Xstar)[0]
            self.Xconcat = tf.concat([self.X, self.Xstar],axis=0)
            self.Nh = tf.shape(self.Xconcat)[0]
            #N, 1
            self.Y_real = Y_real
            #N, 1
            self.Y_imag = Y_imag
            #Nf
            self.freqs = freqs

            kern = DTECIsotropicTimeGeneral(
                variance=self.state.variance.constrained_value,
                lengthscales=self.state.lengthscales.constrained_value,
                timescale=self.state.timescale.constrained_value,
                a=self.state.a.constrained_value,
                b=self.state.b.constrained_value,
                resolution=3,
                fed_kernel=self.fed_kernel,
                obs_type=self.obs_type,
                squeeze=True)


            # N+Ns, N+Ns
            K = kern.K(self.Xconcat)
            # with tf.control_dependencies([tf.print("asserting initial K finite"), tf.assert_equal(tf.reduce_all(tf.is_finite(K)), True)]):
            # N+Ns, N+Ns
            self.L = tf.cholesky(K + diagonal_jitter(self.Nh))
            self.L.set_shape(tf.TensorShape([None,None]))

    def transform_samples(self, dtec):
        """
        Calculate transformed samples.

        :param dtec: float_type tf.Tensor [S, M]
        :return: the unconstrained parameters in a list
        """

        # transform
        # S, num_chains, N+Ns
        dtec_transformed = tf.matmul(dtec, self.L, transpose_b=True)#tf.einsum("ab,sb->sa",self.L, dtec)

        return dtec_transformed


    def get_initial_point(self, dtec):
        """
        Get an initial point from constrained values

        :param dtec: float_type tf.Tensor [M]
            the constrained dtec
        :return: float_type tf.Tensor [M]
            the unconstrained dtec
        """
        # with tf.control_dependencies([tf.print(tf.shape(self.L), tf.shape(dtec))]):
        return tf.linalg.triangular_solve(self.L, dtec[:, None])[:,0]

    def forward_equation(self, dtec):
        """
        Calculate real and imaginary parts of gains from dtec.

        :param dtec: float_type, Tensor [b0,...,bB]
            The DTECs
        :return: float_type, Tensor [b0,...,bB,Nf]
            Real part
        :return: float_type, Tensor [b0,...,bB,Nf]
            Imag part
        TODO how to specify multiple returns in a tuple
        """
        #Nf
        invfreqs = -8.448e9*tf.reciprocal(self.freqs)
        #..., Nf
        phase = dtec[..., None] * invfreqs
        real_part = tf.cos(phase)
        imag_part = tf.sin(phase)
        return real_part, imag_part

    def logp(self, dtec):
        """
        Calculate the log probability of the gains given a model.

        :param dtec: float_type, tf.Tensor, [num_chains, N+Ns]
            The differential TEC that model the gains.
        :return: float_type, tf.Tensor, [num_chains]
            The log-probability of the data given model.
        """

        # transform
        # num_chains, N+Ns
        dtec_transformed = tf.matmul(dtec, self.L, transpose_b=True)#tf.einsum('ab,nb->na',self.L, dtec)

        #marginal
        #num_chains, N
        dtec_marginal = dtec_transformed[:, :self.N]
        # num_chains, N, Nf
        g_real, g_imag = self.forward_equation(dtec_marginal)
        likelihood_real = tfp.distributions.Laplace(loc=g_real, scale=self.state.y_sigma.constrained_value[:, :, None])
        likelihood_imag = tfp.distributions.Laplace(loc=g_imag, scale=self.state.y_sigma.constrained_value[:, :, None])

        # num_chains
        logp = tf.reduce_sum(likelihood_real.log_prob(self.Y_real[None, :, :]), axis=[1, 2]) + \
               tf.reduce_sum(likelihood_imag.log_prob(self.Y_imag[None, :, :]), axis=[1, 2])

        # dtec_prior = tfp.distributions.MultivariateNormalDiag(loc=None,scale_identity_multiplier=None)
        # num_chains
        #[1]
        logdet = tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)), axis=0, keepdims=True)
        # dtec_logp = dtec_prior.log_prob(dtec) - logdet
        dtec_logp = -0.5*tf.reduce_sum(tf.square(dtec),axis=1) - logdet # - 0.5*(self.N+self.Ns)*np.log(2*np.pi)

        #print(logp, dtec_logp)
        res = logp# + dtec_logp# + sum([p.constrained_prior.log_prob(s) for (p,s) in zip(self.parameters, state)])

        # res.set_shape(tf.TensorShape([self.num_chains]))
        # with tf.control_dependencies(
        #         [tf.print("logp",tf.shape(res), res)]):
        return res



