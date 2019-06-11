import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from .sgd import adam_stochastic_gradient_descent, natural_adam_stochastic_gradient_descent, natural_adam_stochastic_gradient_descent_with_linesearch, natural_adam_stochastic_gradient_descent_with_linesearch_minibatch
from . import float_type, TEC_CONV
from .misc import sqrt_with_finite_grads, safe_cholesky, flatten_batch_dims
from .kernels import DTECIsotropicTimeGeneral


class Likelihood(object):
    def __init__(self):
        pass

    def log_prob(self, *args):
        pass


class VariationalPosterior(object):
    def __init__(self, event_size):
        self._event_size = event_size
        self._distribution = None

    def sample(self, num_samples):
        if self._distribution is None:
            raise ValueError("no distribution defined")
        return self._distribution.sample(num_samples)

    def _build_distribution(self, *params):
        """
        Build the distribution.

        :param params:
        :return:
        """
        raise NotImplementedError()

    def initial_variational_params(self, batch_size):
        raise NotImplementedError()


class WhitenedVariationalPosterior(VariationalPosterior):
    def __init__(self, event_size):
        super(WhitenedVariationalPosterior, self).__init__(event_size=event_size)

    def _build_distribution(self, loc, scale):
        """
        Build the MultivariateNormalDiagWithSoftplusScale distribution

        :param loc:
        :param scale:
        :return:
        """
        return tfp.distributions.MultivariateNormalDiag(loc, scale_diag=tf.nn.softplus(scale))

    def initial_variational_params(self, batch_size=None):
        """
        Gets the initial parameters

        :param batch_size:
        :return:
        """
        if batch_size is not None:
            m = tf.zeros(shape=[batch_size, self._event_size], dtype=float_type)
            S_inverse = tfp.distributions.softplus_inverse(
                tf.ones(shape=[batch_size, self._event_size], dtype=float_type))
            return m, S_inverse
        m = tf.zeros(shape=[self._event_size], dtype=float_type)
        S_inverse = tfp.distributions.softplus_inverse(
            tf.ones(shape=[self._event_size], dtype=float_type))
        return m, S_inverse


class LaplaceLikelihood(Likelihood):
    def __init__(self, Yreal, Yimag, freqs, transform_fn):
        super(LaplaceLikelihood, self).__init__()
        self._Yreal = Yreal
        self._Yimag = Yimag
        self._invfreqs = tf.constant(TEC_CONV, float_type) * tf.math.reciprocal(freqs)
        self._transform_fn = transform_fn

    def log_prob(self, white_dtec, y_sigma):
        """
        Represents log P(Yreal, Yimag | white_dtec, hyperparams)
        where P is the product of Laplace distributions over frequency and coordinate index
        log P = Sum_i Sum_nu (-log(2) - log(y_sigma) - (|Yreal(i,nu) - Yreal_model(i, nu)| + |Yimag(i, nu) - Yimag_model(i, nu)|) / y_sigma)

        :param white_dtec: tf.Tensor
            [A, N]
        :param log_y_sigma: tf.Tensor
            [B, 1]
        :return: tf.Tensor
            [A, B]
        """

        Nf = tf.cast(tf.shape(self._invfreqs)[0],float_type)
        # A, B, N
        dtec = self._transform_fn(white_dtec)
        # [A, B, N, Nf]
        phase = dtec[..., None] * self._invfreqs
        Yreal_model = tf.cos(phase)
        Yimag_model = tf.sin(phase)
        # B, 1
        log_y_sigma = tf.math.log(y_sigma)
        # [A, B, N, Nf]
        likelihood = -tf.math.reciprocal(y_sigma[..., None]) * sqrt_with_finite_grads(
            tf.math.square(self._Yimag - Yimag_model) + tf.math.square(self._Yreal - Yreal_model))\
                     - log_y_sigma[..., None] - tf.math.log(tf.constant(2., float_type))
        # A, B
        #TODO: is div by Nf right?

        likelihood = tf.reduce_sum(likelihood, axis=[-2, -1])/Nf
        prior = tf.reduce_mean(tfp.distributions.Normal(loc=tf.constant(0.05, float_type), scale=tf.constant(0.05, float_type)).log_prob(y_sigma))
        return likelihood + prior

class VariationalBayesHeirarchical(object):
    def __init__(self, Yreal, Yimag, freqs, X, Xstar, dtec_samples=10, hyperparam_samples=10, mean_hyperparam_approx=True, obs_type='DTEC',
                 fed_kernel='RBF'):
        self._Yreal = Yreal
        self._Yimag = Yimag
        self._freqs = freqs
        self._invfreqs = tf.constant(TEC_CONV, float_type) * tf.math.reciprocal(freqs)
        self._X = X
        self._Xstar = Xstar
        self._Xconcat = tf.concat([self._X, self._Xstar], axis=0)
        self.N = tf.shape(self._X)[0]
        self.Ns = tf.shape(self._Xstar)[0]
        self._obs_type = obs_type
        self._fed_kernel = fed_kernel
        self._dtec_samples = tf.convert_to_tensor(dtec_samples, tf.int32, name='num_dtec_samples')
        self._hyperparam_samples = tf.convert_to_tensor(hyperparam_samples, tf.int32, name='num_hyperparam_samples')
        self._mean_hyperparam_approx = mean_hyperparam_approx

        self._white_posterior = WhitenedVariationalPosterior(event_size=tf.shape(self._X)[0])
        # amp, lengthscales, a, b, timescales, y_sigma
        self._hyperparam_posterior = WhitenedVariationalPosterior(event_size=6)

        self._hyperparam_bijectors = [
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(3., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(15., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(250., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(100., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(50., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(0.1, float_type)), tfp.bijectors.Softplus()])
        ]

    def _initial_states(self, batch_size=None):
        return self._white_posterior.initial_variational_params(
            batch_size), self._hyperparam_posterior.initial_variational_params(batch_size)

    def _constrain_hyperparams(self, sampled_hyperparams):
        """
        Constrains the samples of hyperparams.

        :param sampled_hyperparams: tf.Tensor
            [samples, 6]
        :return: Tuple of tf.Tensor
            Each of shape [samples, 1]
        """
        constrained_hyperparams = []
        for i in range(len(self._hyperparam_bijectors)):
            bijector = self._hyperparam_bijectors[i]
            # num_hyperparams, 1
            s = sampled_hyperparams[:, i:i + 1]
            constrained_hyperparams.append(bijector.forward(s))

        return constrained_hyperparams

    def _loss_fn(self, white_dtec_mean, white_dtec_scale, hyperparam_mean, hyperparam_scale):

        white_vi_params, hyperparam_vi_params = (white_dtec_mean, white_dtec_scale), (hyperparam_mean, hyperparam_scale)

        hyperparam_dist = self._hyperparam_posterior._build_distribution(*hyperparam_vi_params)


        if self._mean_hyperparam_approx:
            #1, 6
            sampled_hyperparams = hyperparam_vi_params[0][None,:]
        else:
            # num_hyperparams, 6
            sampled_hyperparams = hyperparam_dist.sample(self._hyperparam_samples)

        amp, lengthscales, a, b, timescale, y_sigma = self._constrain_hyperparams(sampled_hyperparams)

        kern = DTECIsotropicTimeGeneral(variance=tf.math.square(amp),
                                        lengthscales=lengthscales,
                                        a=a,
                                        b=b,
                                        timescale=timescale,
                                        fed_kernel=self._fed_kernel,
                                        obs_type=self._obs_type,
                                        squeeze=False)
        # num_hyperparams, N, N
        K = kern.K(self._X, None)
        # num_hyperparams, N, N
        L = safe_cholesky(K)

        # no mean right now

        def transform_fn(white_dtec):
            """
            Constrain white_dtec to tec
        mean_approx_hyperparams
            :param white_dtec: tf.Tensor
                [b0,..., bB, N]
            :param data_only: tf.bool
            :return: tf.Tensor
                [b0,...,bB, b0,...,bC,N]
            """
            # TODO: add mean
            # L[d,i,j].white_dtec[b,j] -> [b,d,i]
            # b0,..., bB, , b0,...,bC,N
            return tf.tensordot(white_dtec, L, axes=[[-1], [-1]])

        white_dist = self._white_posterior._build_distribution(*white_vi_params)
        # num_dtec, N
        white_dtec = white_dist.sample(self._dtec_samples)

        likelihood = LaplaceLikelihood(self._Yreal, self._Yimag, self._freqs, transform_fn=transform_fn)

        # num_hyperparams
        #TODO: derive better var_exp
        var_exp = tf.reduce_mean(likelihood.log_prob(white_dtec, y_sigma), axis=0)

        # num_hyperparams
        dtec_prior_KL = self._dtec_prior_kl(white_vi_params, L)
        # scalar
        hyperparam_prior_KL = self._hyperparams_prior_kl(hyperparam_vi_params)
        # scalar
        elbo = tf.reduce_mean(var_exp - dtec_prior_KL, axis=0) - hyperparam_prior_KL
        with tf.control_dependencies([tf.print('elbo', elbo,
                                               'var_exp', var_exp, 'dtec_prior', dtec_prior_KL,
                                               'hyperparam_prior', hyperparam_prior_KL,
                                               'amp', amp, 'lengthscales', lengthscales, 'a', a, 'b', b, 'timescale',
                                               timescale, 'y_sigma', y_sigma)]):

            loss = tf.math.negative(elbo, name='loss')
        return loss

    def _hyperparams_prior_kl(self, hyperparams_params):
        """The KL-div[ Q(hyperparams) || P(hyperparams) ]

        P(hyperparams)
        = U[-infty, infty](hyperparams)
        = N[0, infty](hyperparams)

        :param hyperparams_params: tf.Tensor
            [6] mean
            [6] diag_scale
        :return: tf.Tensor
            scalar
        """
        variance = tf.math.square(tf.nn.softplus(hyperparams_params[1]))
        entropy = tf.reduce_sum(tf.constant(0.5, float_type) * tf.math.log(tf.constant(2 * np.pi * np.exp(1), float_type) * variance))
        return -entropy

    def _dtec_prior_kl(self, white_dtec_params, L):
        """
        Get the KL-div [ Q(white_dtec) || P(white_dtec | hyperparams)]
        where
        Q = N[m, S] and S is diagonal
        and
        P = |L|^{-1} N[0,I]

        KL-div [ N[m, S] || |L|^{-1} N[0,I]] =
        KL-div [ N[m, S] || |L|^{-1} N[0,I]] + log |L|

        :param white_dtec_params: tuple of tf.Tensor
            [N+Ns] mean
            [N+Ns] unconstrained scale
        :param L: tf.Tensor
            [num_hyperparams, N, N]
        :return: tf.Tensor
            [num_hyperparams]
        """
        # [N+Ns]
        mean, S = white_dtec_params
        variance = tf.math.square(tf.nn.softplus(S))
        # num_hyperparams
        logdetL = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=-1)
        return 0.5 * tf.reduce_sum(
            variance + tf.math.square(mean) - tf.constant(1., float_type) - 2. * tf.math.log(variance),
            axis=-1) + logdetL

    def _build_variational_posteriors(self, white_vi_params, hyperparam_vi_params):
        hyperparam_dist = self._hyperparam_posterior._build_distribution(*hyperparam_vi_params)
        white_dist = self._white_posterior._build_distribution(*white_vi_params)

        return white_dist, hyperparam_dist

    def solve_variational_posterior(self, param_warmstart, hyperparams_warmstart,
                                    iters=100, learning_rate=0.001, parallel_iterations=10):
        (white_dtec_mean, white_dtec_scale), (hyperparam_mean, hyperparam_scale) = self._initial_states()

        ((white_dtec_mean, white_dtec_scale), (hyperparam_mean, hyperparam_scale)) = \
            tf.cond(tf.reduce_all(tf.equal(param_warmstart[0], 0.)),
                    lambda: ((white_dtec_mean, white_dtec_scale), (hyperparam_mean, hyperparam_scale)),
                    lambda: (param_warmstart, hyperparams_warmstart), strict=True)

        # [white_dtec_mean, white_dtec_scale, hyperparam_mean, hyperparam_scale], loss = \
        #     adam_stochastic_gradient_descent(self._loss_fn,
        #                                      [white_dtec_mean, white_dtec_scale, hyperparam_mean, hyperparam_scale],
        #                                      iters=iters,
        #                                      learning_rate=learning_rate,
        #                                      parallel_iterations=parallel_iterations)

        [white_dtec_mean, white_dtec_scale], [hyperparam_mean, hyperparam_scale], loss = \
            natural_adam_stochastic_gradient_descent(self._loss_fn,
                                                     [white_dtec_mean, white_dtec_scale],
                                                     [hyperparam_mean, hyperparam_scale],
                                                     iters=iters,
                                                     learning_rate=learning_rate,
                                                     parallel_iterations=parallel_iterations)



        ###
        # produce the posterior distributions needed

        hyperparam_dist = self._hyperparam_posterior._build_distribution(hyperparam_mean, hyperparam_scale)

        if self._mean_hyperparam_approx:
            # 1, 6
            sampled_hyperparams = hyperparam_mean[None, :]
        else:
            # num_hyperparams, 6
            sampled_hyperparams = hyperparam_dist.sample(self._hyperparam_samples)
        amp, lengthscales, a, b, timescale, y_sigma = self._constrain_hyperparams(sampled_hyperparams)


        kern = DTECIsotropicTimeGeneral(variance=tf.math.square(amp),
                                        lengthscales=lengthscales,
                                        a=a,
                                        b=b,
                                        timescale=timescale,
                                        fed_kernel=self._fed_kernel,
                                        obs_type=self._obs_type,
                                        squeeze=False)

        # num_hyperparams, N, N
        K_xx = kern.K(self._X, None)
        # num_hyperparams, N, N
        L_xx = safe_cholesky(K_xx)
        # num_hyperparams, M, N
        K_yx = kern.K(self._X, self._Xstar)

        q_mean, q_sqrt = white_dtec_mean, tf.nn.softplus(white_dtec_scale)
        dtec_data_dist = conditional_same_points(q_mean, q_sqrt, L_xx)
        dtec_screen_dist = conditional_different_points(q_mean, q_sqrt, L_xx, K_xx, K_yx)

        return loss, dtec_data_dist, dtec_screen_dist, (amp, lengthscales, a, b, timescale, y_sigma), (
        white_dtec_mean, white_dtec_scale), (hyperparam_mean, hyperparam_scale)

class VariationalBayesZIsX(object):
    def __init__(self, Yreal, Yimag, freqs, X, Xstar, y_sigma, dtec_samples=10, kernel_params=None, minibatch_size=None, quadrature_var_exp=False):
        self._Yreal = Yreal
        self._Yimag = Yimag
        self._freqs = freqs
        self._y_sigma = y_sigma
        self._invfreqs = tf.constant(TEC_CONV, float_type) * tf.math.reciprocal(freqs)
        self._X = X
        self._Xstar = Xstar
        self._Xconcat = tf.concat([self._X, self._Xstar], axis=0)
        self.N = tf.shape(self._X)[0]
        self.Ns = tf.shape(self._Xstar)[0]
        self._kernel_params = kernel_params
        self._dtec_samples = tf.convert_to_tensor(dtec_samples, tf.int32, name='num_dtec_samples')
        self._minibatch_size = tf.convert_to_tensor(minibatch_size,tf.int64) if minibatch_size is not None else None
        if self._minibatch_size is not None:
            self._scale = tf.cast(self.N, float_type)/tf.cast(self._minibatch_size, float_type)
        else:
            self._scale = tf.constant(1., float_type)
        self._quadrature_var_exp = quadrature_var_exp

        self._white_posterior = WhitenedVariationalPosterior(event_size=tf.shape(self._X)[0])

        self._hyperparam_bijectors = [
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(3., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(15., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(250., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(100., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(50., float_type)), tfp.bijectors.Softplus()])
        ]

    def _initial_states(self, batch_size=None):
        return self._white_posterior.initial_variational_params(
            batch_size), (tfp.distributions.softplus_inverse(tf.ones((1,5),float_type)),)

    def _constrain_hyperparams(self, sampled_hyperparams):
        """
        Constrains the samples of hyperparams.

        :param sampled_hyperparams: tf.Tensor
            [samples, 6]
        :return: Tuple of tf.Tensor
            Each of shape [samples, 1]
        """
        constrained_hyperparams = []
        for i in range(len(self._hyperparam_bijectors)):
            bijector = self._hyperparam_bijectors[i]
            # num_hyperparams, 1
            s = sampled_hyperparams[:, i:i + 1]
            constrained_hyperparams.append(bijector.forward(s))

        return constrained_hyperparams

    def _loss_fn(self, white_dtec_mean, white_dtec_scale, hyperparams_unconstrained):
        white_vi_params = (white_dtec_mean, white_dtec_scale)

        #each 1,1
        amp, lengthscales, a, b, timescale = self._constrain_hyperparams(hyperparams_unconstrained)

        kern = DTECIsotropicTimeGeneral(variance=tf.math.square(amp),
                                        lengthscales=lengthscales,
                                        a=a,
                                        b=b,
                                        timescale=timescale,
                                        squeeze=False,
                                        **self._kernel_params)
        # num_hyperparams, N, N
        K = kern.K(self._X, None)
        # num_hyperparams, N, N
        L = safe_cholesky(K)

        # no mean right now

        def transform_fn(white_dtec):
            """
            Constrain white_dtec to tec
            L is [B, N, N]

            :param white_dtec: tf.Tensor
                [A, N]
            :return: tf.Tensor
                [A,B,N]
            """
            # TODO: add mean
            # L[d,i,j].white_dtec[b,j] -> [b,d,i]
            # A, B, N
            return tf.tensordot(white_dtec, L, axes=[[-1], [-1]])

        var_exp = self._calculate_var_exp(transform_fn, white_vi_params)
        dtec_prior_KL = self._dtec_prior_kl(white_vi_params, L)

        # scalar
        elbo = var_exp*self._scale - dtec_prior_KL
        with tf.control_dependencies([tf.print('elbo', elbo,
                                               'var_exp', var_exp, 'dtec_prior', dtec_prior_KL,
                                               'amp', amp, 'lengthscales', lengthscales, 'a', a, 'b', b, 'timescale',
                                               timescale, 'y_sigma', self._y_sigma)]):

            loss = tf.math.negative(elbo, name='loss')
        return loss

    def _calculate_var_exp(self, transform_fn, white_vi_params):
        if not self._quadrature_var_exp:
            white_dist = self._white_posterior._build_distribution(*white_vi_params)
            # num_dtec, N
            white_dtec = white_dist.sample(self._dtec_samples)
            likelihood = LaplaceLikelihood(self._Yreal, self._Yimag, self._freqs, transform_fn=transform_fn)
            # TODO: derive better var_exp
            var_exp = tf.reduce_mean(likelihood.log_prob(white_dtec, self._y_sigma))
            return var_exp

        ## Use Gauss Hermite Quadrature


    def _dtec_prior_kl(self, white_dtec_params, L):
        """
        Get the KL-div [ Q(white_dtec) || P(white_dtec | hyperparams)]
        where
        Q = N[m, S] and S is diagonal
        and
        P = |L|^{-1} N[0,I]

        KL-div [ N[m, S] || |L|^{-1} N[0,I]] =
        KL-div [ N[m, S] || |L|^{-1} N[0,I]] + log |L|

        :param white_dtec_params: tuple of tf.Tensor
            [N+Ns] mean
            [N+Ns] unconstrained scale
        :param L: tf.Tensor
            [num_hyperparams, N, N]
        :return: tf.Tensor
            [num_hyperparams]
        """

        logdetL = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        # [N+Ns]
        q_mean, q_scale = white_dtec_params
        q_sqrt = tf.nn.softplus(q_scale)
        q_var = tf.math.square(q_sqrt)
        trace = tf.reduce_sum(q_var)
        mahalanobis = tf.reduce_sum(tf.math.square(q_mean))
        constant = -tf.cast(tf.size(q_mean, out_type=tf.int64), float_type)
        logdet_qcov  = tf.reduce_sum(tf.math.log(q_var))

        twoKL = mahalanobis + constant - logdet_qcov + trace - logdetL
        return 0.5 * twoKL

        # # num_hyperparams
        #
        # return 0.5 * tf.reduce_sum(
        #     variance + tf.math.square(mean) - tf.constant(1., float_type) - 2. * tf.math.log(variance),
        #     axis=-1) + logdetL

    def _build_variational_posteriors(self, white_vi_params):
        white_dist = self._white_posterior._build_distribution(*white_vi_params)

        return white_dist

    def solve_variational_posterior(self, param_warmstart, hyperparams_warmstart,
                                    solver_params=None, parallel_iterations=10):
        (white_dtec_mean, white_dtec_scale), (hyperparams_unconstrained,) = self._initial_states()

        # ((white_dtec_mean, white_dtec_scale), (hyperparams_unconstrained,)) = \
        #     tf.cond(tf.reduce_all(tf.equal(param_warmstart[0], 0.)),
        #             lambda: ((white_dtec_mean, white_dtec_scale), (hyperparams_unconstrained,)),
        #             lambda: (param_warmstart, hyperparams_warmstart), strict=True)

        (white_dtec_mean, white_dtec_scale) = \
            tf.cond(tf.reduce_all(tf.equal(param_warmstart[0], 0.)),
                    lambda: (white_dtec_mean, white_dtec_scale),
                    lambda: param_warmstart, strict=True)

        # TODO: mini batch and choose larger basis
        # TODO: speed up kernel computation ^^ help
        # TODO: fix screen approximation

        [white_dtec_mean, white_dtec_scale], [hyperparams_unconstrained], loss = \
            natural_adam_stochastic_gradient_descent_with_linesearch(self._loss_fn,
                                                                     [white_dtec_mean, white_dtec_scale],
                                                                     [hyperparams_unconstrained],
                                                                     parallel_iterations=parallel_iterations,
                                                                     **solver_params)

        ###
        # produce the posterior distributions needed

        amp, lengthscales, a, b, timescale = self._constrain_hyperparams(hyperparams_unconstrained)

        kern = DTECIsotropicTimeGeneral(variance=tf.math.square(amp),
                                        lengthscales=lengthscales,
                                        a=a,
                                        b=b,
                                        timescale=timescale,
                                        squeeze=False,
                                        **self._kernel_params)

        # num_hyperparams, N, N
        K_x_x = kern.K(self._X, None)
        # num_hyperparams, N, N
        L_x_x = safe_cholesky(K_x_x)
        # num_hyperparams, M, N
        K_x_xstar = kern.K(self._X, self._Xstar)
        K_xstar_xstar = kern.K(self._Xstar, None)

        q_mean, q_sqrt = white_dtec_mean, tf.nn.softplus(white_dtec_scale)
        dtec_data_dist = conditional_same_points(q_mean, q_sqrt, L_x_x)
        dtec_screen_dist = conditional_different_points(q_mean, q_sqrt, L_x_x, K_xstar_xstar, K_x_xstar)

        return loss, dtec_data_dist, dtec_screen_dist, (amp, lengthscales, a, b, timescale), (
        white_dtec_mean, white_dtec_scale), (hyperparams_unconstrained,)


class VariationalBayes(object):
    def __init__(self, Yreal, Yimag, freqs, X, Xstar, Z, y_sigma, dtec_samples=10, kernel_params=None, minibatch_size=None):
        self._Yreal = Yreal
        self._Yimag = Yimag
        self._freqs = freqs
        self._y_sigma = y_sigma
        self._invfreqs = tf.constant(TEC_CONV, float_type) * tf.math.reciprocal(freqs)
        self._X = X
        self._Xstar = Xstar
        self._Z = Z
        self._Xconcat = tf.concat([self._X, self._Xstar], axis=0)
        self.N = tf.shape(self._X)[0]
        self.Ns = tf.shape(self._Xstar)[0]
        self.Nz = tf.shape(self._Z)[0]
        self._kernel_params = kernel_params
        self._dtec_samples = tf.convert_to_tensor(dtec_samples, tf.int32, name='num_dtec_samples')
        self._minibatch_size = tf.convert_to_tensor(minibatch_size,tf.int64) if minibatch_size is not None else None
        if self._minibatch_size is not None:
            self._scale = tf.cast(self.N, float_type)/tf.cast(self._minibatch_size, float_type)
        else:
            self._scale = tf.constant(1., float_type)

        self._white_posterior = WhitenedVariationalPosterior(event_size=self.Nz)

        self._hyperparam_bijectors = [
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(3., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(15., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(250., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(100., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(50., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(2., float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(0.03, float_type)), tfp.bijectors.Softplus()]),
            tfp.bijectors.Chain(
                [tfp.bijectors.AffineScalar(scale=tf.constant(10., float_type)), tfp.bijectors.Softplus()])
        ]

    def _initial_states(self, batch_size=None):
        return self._white_posterior.initial_variational_params(
            batch_size), (tfp.distributions.softplus_inverse(tf.ones((1,8),float_type)),)

    def _constrain_hyperparams(self, sampled_hyperparams):
        """
        Constrains the samples of hyperparams.

        :param sampled_hyperparams: tf.Tensor
            [samples, 6]
        :return: Tuple of tf.Tensor
            Each of shape [samples, 1]
        """
        constrained_hyperparams = []
        for i in range(len(self._hyperparam_bijectors)):
            bijector = self._hyperparam_bijectors[i]
            # num_hyperparams, 1
            s = sampled_hyperparams[:, i:i + 1]
            constrained_hyperparams.append(bijector.forward(s))

        return constrained_hyperparams

    def _loss_fn(self, q_mean, q_scale, hyperparams_unconstrained, X, Y):


        #each 1,1
        amp, lengthscales, a, b, timescale, pert_amp, pert_dir_lengthscale, pert_ant_lengthscale = self._constrain_hyperparams(hyperparams_unconstrained)

        kern = DTECIsotropicTimeGeneral(variance=tf.math.square(amp),
                                        lengthscales=lengthscales,
                                        a=a,
                                        b=b,
                                        timescale=timescale,
                                        squeeze=False,
                                        **self._kernel_params)

        pert_dir_kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=pert_amp[0,:],
                                                                             length_scale=pert_dir_lengthscale[0,:])
        pert_ant_kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(length_scale=pert_ant_lengthscale[0,:])

        # 1, 1, Nz, Nz
        K_z_z = kern.K(self._Z, None) + pert_dir_kern.matrix(self._Z[:, 1:4], self._Z[:, 1:4])*pert_ant_kern.matrix(self._Z[:, 4:7], self._Z[:, 4:7])
        L_z_z = safe_cholesky(K_z_z)


        # q_mean, q_scale = white_vi_params
        q_sqrt = tf.nn.softplus(q_scale)

        dtec_prior_KL = self._dtec_prior_kl(q_mean, q_sqrt, L_z_z)

        if self._minibatch_size is not None:
            K_z_xmini = kern.K(self._Z, X) + pert_dir_kern.matrix(self._Z[:, 1:4], X[:, 1:4])*pert_ant_kern.matrix(self._Z[:, 4:7], X[:, 4:7])
            K_xmini_xmini = kern.K(X, None) + pert_dir_kern.matrix(X[:, 1:4], X[:, 1:4])*pert_ant_kern.matrix(X[:, 4:7], X[:, 4:7])
            q_dist = conditional_different_points(q_mean, q_sqrt, L_z_z, K_xmini_xmini, K_z_xmini)
            dtec_samples = q_dist.sample(self._dtec_samples)

            likelihood = LaplaceLikelihood(Y[0], Y[1], self._freqs, transform_fn=lambda x: x)
            # TODO: derive better var_exp
            var_exp = tf.reduce_mean(likelihood.log_prob(dtec_samples, self._y_sigma))

        else:
            # num_dtec, num_hyperparams, N, N
            L_expanded = tf.tile(tf.expand_dims(L_z_z, 0), [self._dtec_samples, 1, 1, 1])
            def transform_fn(white_dtec):
                """
                Constrain white_dtec to tec
                L is [A, B, N, N]

                :param white_dtec: tf.Tensor
                    [A, B, N, 1]
                :return: tf.Tensor
                    [A,B,N]
                """
                # white_dtec[a,b,j,1].L[a,b,i,j] -> white_dtec^T[a,b,1,j].L^T[a,b,j,i] -> [a,b, 1, i]
                return tf.matmul(white_dtec, L_expanded, transpose_a=True, transpose_b=True)[:,:,0, :]

                # # A, B, N
                # return tf.tensordot(white_dtec, L_z_z, axes=[[-1], [-1]])

            white_dist = self._white_posterior._build_distribution(q_mean, q_scale)
            # num_dtec, N
            white_dtec = white_dist.sample(self._dtec_samples)
            # num_dtec, 1, N, 1
            white_dtec = white_dtec[:, None, :, None]
            likelihood = LaplaceLikelihood(Y[0], Y[1], self._freqs, transform_fn=transform_fn)
            var_exp = tf.reduce_mean(likelihood.log_prob(white_dtec, self._y_sigma))

        # scalar
        elbo = var_exp*self._scale - dtec_prior_KL
        # with tf.control_dependencies([tf.print('elbo', elbo,
        #                                        'var_exp', var_exp, 'dtec_prior', dtec_prior_KL,
        #                                        'amp', amp, 'lengthscales', lengthscales, 'a', a, 'b', b, 'timescale',
        #                                        timescale, 'y_sigma', self._y_sigma)]):

        loss = tf.math.negative(elbo, name='loss')
        return loss

    def _dtec_prior_kl(self, q_mean,  q_sqrt, L):
        """
        Get the KL-div [ Q(white_dtec) || P(white_dtec | hyperparams)]
        where
        Q = N[m, S] and S is diagonal
        and
        P = |L|^{-1} N[0,I]

        KL-div [ N[m, S] || |L|^{-1} N[0,I]] =
        KL-div [ N[m, S] || |L|^{-1} N[0,I]] + log |L|

        :param white_dtec_params: tuple of tf.Tensor
            [N+Ns] mean
            [N+Ns] unconstrained scale
        :param L: tf.Tensor
            [num_hyperparams, N, N]
        :return: tf.Tensor
            [num_hyperparams]
        """

        logdetL = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        # [N+Ns]
        q_var = tf.math.square(q_sqrt)
        trace = tf.reduce_sum(q_var)
        mahalanobis = tf.reduce_sum(tf.math.square(q_mean))
        constant = -tf.cast(tf.size(q_mean, out_type=tf.int64), float_type)
        logdet_qcov  = tf.reduce_sum(tf.math.log(q_var))

        twoKL = mahalanobis + constant - logdet_qcov + trace - logdetL
        return 0.5 * twoKL

        # # num_hyperparams
        #
        # return 0.5 * tf.reduce_sum(
        #     variance + tf.math.square(mean) - tf.constant(1., float_type) - 2. * tf.math.log(variance),
        #     axis=-1) + logdetL

    def _build_variational_posteriors(self, white_vi_params):
        white_dist = self._white_posterior._build_distribution(*white_vi_params)

        return white_dist

    def solve_variational_posterior(self, param_warmstart,
                                    solver_params=None, parallel_iterations=10):
        param_init, (hyperparams_unconstrained,) = self._initial_states()

        param_warmstart = \
            tf.cond(tf.reduce_all(tf.equal(param_warmstart[0], 0.)),
                    lambda: param_init,
                    lambda: param_warmstart, strict=True)

        # TODO: speed up kernel computation

        with tf.device('/device:GPU:0' if tf.test.is_gpu_available() else '/device:CPU:0'):

            learned_params, [learned_hyperparams_unconstrained], loss, t = \
                natural_adam_stochastic_gradient_descent_with_linesearch_minibatch(self._loss_fn,
                                                                                   self._X,
                                                                                   (self._Yreal, self._Yimag),
                                                                                   self._minibatch_size,
                                                                                   param_warmstart,
                                                                                   [hyperparams_unconstrained],
                                                                                   parallel_iterations=parallel_iterations,
                                                                                   **solver_params)
        ###
        # produce the posterior distributions needed



        # each 1,1
        amp, lengthscales, a, b, timescale, pert_amp, pert_dir_lengthscale, pert_ant_lengthscale = self._constrain_hyperparams(
            learned_hyperparams_unconstrained)

        kern = DTECIsotropicTimeGeneral(variance=tf.math.square(amp),
                                        lengthscales=lengthscales,
                                        a=a,
                                        b=b,
                                        timescale=timescale,
                                        squeeze=False,
                                        **self._kernel_params)

        pert_dir_kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=pert_amp[0,:],
                                                                                 length_scale=pert_dir_lengthscale[0,:])
        pert_ant_kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(length_scale=pert_ant_lengthscale[0,:])

        # 1, 1, Nz, Nz
        K_z_z = kern.K(self._Z, None) + pert_dir_kern.matrix(self._Z[:, 1:4], self._Z[:, 1:4]) * pert_ant_kern.matrix(
            self._Z[:, 4:7], self._Z[:, 4:7])
        L_z_z = safe_cholesky(K_z_z)

        # num_hyperparams, M, N
        K_z_xstar = kern.K(self._Z, self._Xstar) + pert_dir_kern.matrix(self._Z[:, 1:4], self._Xstar[:, 1:4]) * pert_ant_kern.matrix(
            self._Z[:, 4:7], self._Xstar[:, 4:7])
        K_xstar_xstar = kern.K(self._Xstar, None)+ pert_dir_kern.matrix(self._Xstar[:, 1:4], self._Xstar[:, 1:4]) * pert_ant_kern.matrix(
            self._Xstar[:, 4:7], self._Xstar[:, 4:7])

        q_mean, q_scale = learned_params
        q_sqrt = tf.nn.softplus(q_scale)

        dtec_screen_dist = conditional_different_points(q_mean, q_sqrt, L_z_z, K_xstar_xstar, K_z_xstar)

        # num_hyperparams, M, N
        K_z_x = kern.K(self._Z, self._X) + pert_dir_kern.matrix(self._Z[:, 1:4],
                                                                        self._X[:, 1:4]) * pert_ant_kern.matrix(
            self._Z[:, 4:7], self._X[:, 4:7])
        K_x_x = kern.K(self._X, None) + pert_dir_kern.matrix(self._X[:, 1:4],
                                                                         self._X[:, 1:4]) * pert_ant_kern.matrix(
            self._X[:, 4:7], self._X[:, 4:7])
        dtec_data_dist = conditional_different_points(q_mean, q_sqrt, L_z_z, K_x_x, K_z_x)

        dtec_basis_dist = conditional_same_points(q_mean, q_sqrt, L_z_z)

        return t, loss, dtec_basis_dist, dtec_data_dist, dtec_screen_dist, (amp, lengthscales, a, b, timescale, pert_amp, pert_dir_lengthscale, pert_ant_lengthscale), (
        q_mean, q_scale)

def conditional_same_points(q_mean, q_sqrt, L, prior_mean=None):
    """
    Computes P(tau(X) | Y)
    = int P(tau(X) | x(X)) Q(x(X)) dx(X)
    = N[prior_mean + L.q_mean, L.q_sqrt^2.L^T]

    :param q_mean: tf.Tensor
        [N]
    :param q_sqrt: tf.Tensor
        [N]
    :param L: tf.Tensor
        [num_hyperparams, N, N]
    :param prior_mean: tf.Tensor
        [num_hyperparams, N]
    :return: tfp.distributions.MultivariateNormalTriL
        batch_shape is [num_hyperparams]
        event_shape is [N]
    """
    #num_hyperparams, N
    mean = tf.tensordot(L, q_mean, axes=[[-1], [-1]])
    # num_hyperparams, N, N
    scale_tril = L*q_sqrt[None, :]#tf.tensordot(L, q_sqrt, axes=[[-1], [-1]])
    if prior_mean is None:
        return tfp.distributions.MultivariateNormalTriL(loc=mean,
                                                        scale_tril=scale_tril)
    return tfp.distributions.MultivariateNormalTriL(loc=prior_mean + mean,
                                                    scale_tril=scale_tril)


def conditional_different_points(q_mean, q_sqrt, L, K_xstar_xstar, K_x_xstar, prior_mean=None):
    """
    Computes P(tau(X) | Y)
    = int P(tau(Xstar) | x(X)) Q(x(X)) dx(X)
    = |L(X,X)| N[m(Xstar) + K(Xstar, X) L(X,X)^-T.q_mean, K(Xstar,Xstar) + K(Xstar,X) L(X,X)^-T(q_sqrt^2 - I) L(X,X)^-1 K(X,Xstar)]

    :param q_mean: tf.Tensor
        [N]
    :param q_sqrt: tf.Tensor
        [N]
    :param L: tf.Tensor
        [num_hyperparams, N, N]
    :param K_xx: tf.Tensor
        [num_hyperparams, M, M]
    :param K_yx: tf.Tensor
        [num_hyperparams, N, M]
    :param prior_mean: tf.Tensor
        [num_hyperparams, M]
    :return: tfp.distributions.MultivariateNormalTriL
        batch_shape is [num_hyperparams]
        event_shape is [N]
    """
    ###
    # conditional one first
    # [num_hyperparams, M, N]
    A = tf.linalg.triangular_solve(L, K_x_xstar)
    B = q_sqrt[:, None] * A
    # num_hyperparams, N
    mean = tf.tensordot(A, q_mean, axes=[[-2], [-1]])
    f_cov = K_xstar_xstar - tf.matmul(A,A,transpose_a=True) + tf.matmul(B,B,transpose_a=True)
    L = safe_cholesky(f_cov)

    if prior_mean is None:
        return tfp.distributions.MultivariateNormalTriL(loc=mean,
                                                        scale_tril=L)
    return tfp.distributions.MultivariateNormalTriL(loc=prior_mean + mean,
                                                    scale_tril=L)

