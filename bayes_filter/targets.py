import tensorflow as tf
import tensorflow_probability as tfp
from .parameters import Parameter, ScaledLowerBoundedBijector
from collections import namedtuple
from .misc import sqrt_with_finite_grads
from .settings import float_type
from .processes import Process, DTECProcess
from .misc import random_sample
from . import TEC_CONV
import numpy as np



class DTECToGainsSAEM(Process):

    @property
    def _Params(self):
        return namedtuple('DTECToGainsParams',
                                      ['amp', 'y_sigma', 'dtec', 'dtec_prior'])

    def __init__(self,
                 Lp:tf.Tensor,
                 mp:tf.Tensor,
                 dtec_process:DTECProcess):
        """
        Creates an instance of the target distribution for complex gains modelled by DTEC.

        :param initial_hyperparams: dict
            The initial parameters for the DTEC process
        :param variables: float tf.Tensor or None
            If None then will initialise variables from initial_hyperparams or the default.
        """
        self._setup = False

        self.dtec_process = dtec_process

        self.Lp = Lp

        self.logdetLp = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lp)))
        self.mp = mp

        def _vec_bijector(m, *L):
            """
            Make the appropriate bijector for L.x + m = y

            where x is tf.Tensor of shape [S, N]

            :param L: tf.Tensor
                lower triangular [N, N]
            :param m: tf.Tensor
                mean [N]
            :return:
            """

            _L = L[0]
            for i in range(1,len(L)):
                _L = tf.matmul(_L, L[i])

            def _forward(x):
                # ij,sj-> si
                return tf.matmul(x,_L,transpose_b=True) + m

            def _inverse(y):
                # ij, sj -> si
                return tf.transpose(tf.linalg.triangular_solve(_L, tf.transpose(y - m)))

            def _inverse_log_jac(y):
                logdetjac = [-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(l))) for l in L]
                return sum(logdetjac)

            return tfp.bijectors.Inline(forward_fn=_forward,
                                        inverse_fn=_inverse,
                                        inverse_log_det_jacobian_fn=_inverse_log_jac,
                                        forward_min_event_ndims=1)

        bijectors = self.Params(
            amp = tfp.bijectors.Exp(),#ScaledLowerBoundedBijector(0.1, 1.),
            y_sigma = ScaledLowerBoundedBijector(1e-2, 0.1),
            #dtec = L.(Lp.y + mp) + m
            dtec = _vec_bijector(tf.matmul(self.dtec_process.L, mp[:,None])[:,0] + self.dtec_process.m,
                                 self.dtec_process.L, Lp),
            dtec_prior = _vec_bijector(mp, Lp)
            )
        super(DTECToGainsSAEM, self).__init__(bijectors=bijectors, distributions=None, unconstrained_values=None)

    @staticmethod
    def init_variables(num_chains, full_block_size, tf_seed=0):
        amp_bijector = tfp.bijectors.Exp()#ScaledLowerBoundedBijector(0.1, 1.)
        y_sigma_bijector = ScaledLowerBoundedBijector(1e-2, 0.1)
        init_y_sigma = y_sigma_bijector.inverse(
            tf.random.truncated_normal(mean=tf.constant(0.1, dtype=float_type),
                                       stddev=tf.constant(0.03, dtype=float_type),
                                       shape=[num_chains, 1], dtype=float_type, seed=tf_seed))
        init_amp = amp_bijector.inverse(
            tf.random.truncated_normal(mean=tf.constant(1., dtype=float_type),
                                       stddev=tf.constant(0.5, dtype=float_type),
                                       shape=[num_chains, 1], dtype=float_type, seed=tf_seed))
        init_dtec = tf.random.truncated_normal(shape=[num_chains, full_block_size], dtype=float_type, seed=tf_seed)

        return init_amp, init_y_sigma, init_dtec

    def setup_target(self,Y_real, Y_imag, freqs,
                     full_posterior=True):
        self.full_posterior = full_posterior
        # N, Nf
        self.Y_real = Y_real
        # N, Nf
        self.Y_imag = Y_imag
        # Nf
        self.freqs = freqs

        self.N = self.dtec_process.N
        self.Ns = self.dtec_process.Ns
        self.Nh = self.dtec_process.Nh

        # [1]
        self.logdetLp = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.Lp)), axis=-1, keepdims=True)

        # dtec        = L.x + m (reparameterisation of prior)
        # x           = Lp.y + mp (decorrelation of posterior)
        # dtec        = L.(Lp.y + mp) + m


        self._setup = True

    def forward_equation(self, dtec):
        """
        Calculate real and imaginary parts of gains from dtec.

        :param dtec: float_type, Tensor [b0,...,bB]
            The DTECs
        :returns: tuple
            float_type, tf.Tensor [b0,...,bB,Nf] Real part
            float_type, tf.Tensor [b0,...,bB,Nf] Imag part
        """
        #Nf
        invfreqs = TEC_CONV*tf.math.reciprocal(self.freqs)
        #..., Nf
        phase = dtec[..., None] * invfreqs
        real_part = tf.cos(phase)
        imag_part = tf.sin(phase)
        return real_part, imag_part

    def log_prob_gains(self, constrained_params):
        """
        Get log Prob(gains | dtec)

        :param constrained_y_sigma: float, tf.Tensor
            y_sigma [num_samples, 1]
        :param constrained_dtec: float_type, tf.Tensor, [S, M]
            Cosntrained dtec
        :return: float_type, tf.Tensor, scalar
            The log probability
        """
        # marginal
        # S, N
        dtec_marginal = constrained_params.dtec[:, :self.N]
        # S, N, Nf
        g_real, g_imag = self.forward_equation(dtec_marginal)
        likelihood_real = tfp.distributions.Laplace(loc=g_real, scale=constrained_params.y_sigma[:, :, None])
        likelihood_imag = tfp.distributions.Laplace(loc=g_imag, scale=constrained_params.y_sigma[:, :, None])

        # S
        logp = tf.reduce_sum(likelihood_real.log_prob(self.Y_real[None, :, :]), axis=[1, 2]) + \
               tf.reduce_sum(likelihood_imag.log_prob(self.Y_imag[None, :, :]), axis=[1, 2])

        return logp

    def log_prob(self, amp, y_sigma, dtec):
        """
        Calculate the log probability of the gains given a model.

        :param amp: float_type tf.Tensor [num_chains, 1]
            Unconstrained amp
        :param y_sigma: float_type tf.Tensor [num_chains, 1]
            Unconstrained y_sigma
        :param dtec: float_type tf.Tensor [num_chains, N+Ns]
            Unconstrained dtec

        :return: float_type, tf.Tensor, [num_chains]
            The log-probability of the data given model.
        """
        if not self.setup:
            raise ValueError("setup is not complete, must run setup_target")

        unconstrained_params = self.Params(amp=amp, #[num_chains, 1]
                             y_sigma=y_sigma, #[num_chains,1]
                             dtec=dtec, #[num_chains,N+Ns]
                             dtec_prior=dtec) #[num_chains,N+Ns]

        constrained_params = self.constrained_state(unconstrained_params)
        constrained_params = constrained_params._replace(dtec=constrained_params.amp * constrained_params.dtec)

        # dtec        = L.x + m
        # x           = Lp.y + mp
        # dtec        = L.(Lp.y + mp) + m
        # P(dtec)        = P(y) | ddtec/dy |^(-1)
        # N[m, L.L^T] |L||Lp| = P(y)
        # log P(y) = -1/2 (L.(Lp.y + mp) + m - m)^T L^-T L^-1 (L.(Lp.y + mp) + m - m) - D/2log(2pi) - log(|L|) + log(|L|) + log(|Lp|)
        #          = -1/2 ((Lp.y + mp))^T ((Lp.y + mp)) - D/2log(2pi) + log(|Lp|)

        # num_chains
        log_prob_gains = self.log_prob_gains(constrained_params)

        # num_chains
        log_prob_dtec_prior = -0.5*tf.reduce_sum(tf.square(constrained_params.dtec_prior),axis=1) + self.logdetLp - 0.5*tf.cast(self.N+self.Ns, float_type)*np.log(2*np.pi)

        #num_chains
        log_prob_y_sigma_prior = tf.reduce_sum(tfp.distributions.Normal(loc=tf.constant(0.1,dtype=float_type),
                                                    scale=tf.constant(0.05,dtype=float_type)).log_prob(
                                    constrained_params.y_sigma), axis=-1)

        # num_chains
        log_prob_amp_prior = tf.reduce_sum(tfp.distributions.Normal(loc=tf.constant(1.0, dtype=float_type),
                                                                        scale=tf.constant(0.75,
                                                                                          dtype=float_type)).log_prob(
            constrained_params.amp), axis=-1)

        # log_prob_amp = tfp.distributions.Normal(loc=tf.constant(0.1, dtype=float_type),
        #                                             scale=tf.constant(0.05, dtype=float_type)).log_prob(
        #     constrained_params.amp)
        if self.full_posterior:
            res = log_prob_gains + log_prob_dtec_prior + log_prob_y_sigma_prior + log_prob_amp_prior
        else:
            res = log_prob_gains

        return res


class DTECToGainsTarget(object):

    @property
    def Params(self):
        return namedtuple('DTECToGainsParams',
                                      ['amp', 'y_sigma', 'dtec'])

    def __init__(self,
                 dtec_process:DTECProcess):
        """
        Creates an instance of the target distribution for complex gains modelled by DTEC.

        :param initial_hyperparams: dict
            The initial parameters for the DTEC process
        :param variables: float tf.Tensor or None
            If None then will initialise variables from initial_hyperparams or the default.
        """
        self._setup = False

        self.dtec_process = dtec_process

    @staticmethod
    def init_variables(num_chains, full_block_size, tf_seed=0):
        """
        Get initial variables for the target.

        :param num_chains:
        :param full_block_size:
        :param tf_seed:
        :return:
        """

        init_y_sigma = tf.math.log(
            tf.random.uniform(shape=[num_chains, 1],
                              minval=tf.constant(0.05, dtype=float_type),
                              maxval=tf.constant(0.15, dtype=float_type),
                               dtype=float_type, seed=tf_seed))
        init_amp = tf.math.log(
            tf.random.uniform(shape=[num_chains, 1],
                              minval=tf.constant(0.5, dtype=float_type),
                              maxval=tf.constant(3., dtype=float_type),
                               dtype=float_type, seed=tf_seed))
        init_dtec = 0.3*tf.random.truncated_normal(shape=[num_chains, full_block_size], dtype=float_type, seed=tf_seed)

        return init_amp, init_y_sigma, init_dtec

    def setup_target(self,Y_real, Y_imag, freqs,
                     full_posterior=True):
        """
        Taken from the instatiation to allow cacheing.

        :param Y_real:
        :param Y_imag:
        :param freqs:
        :param full_posterior:
        :return:
        """
        self.full_posterior = full_posterior
        # N, Nf
        self.Y_real = Y_real
        # N, Nf
        self.Y_imag = Y_imag
        # Nf
        self.freqs = freqs

        # Nf
        self.invfreqs = TEC_CONV * tf.math.reciprocal(self.freqs)

        self.N = self.dtec_process.N
        self.Ns = self.dtec_process.Ns
        self.Nh = self.dtec_process.Nh

        self.L_data = self.dtec_process.L[:self.N, :]
        self.L = self.dtec_process.L

        self._setup = True

    def transform_state(self,log_amp, log_y_sigma, f, data_only=False):
        """
        Transform the input state into constrained variables.

        :param log_amp:
            [S, 1]
        :param log_y_sigma:
            [S, 1]
        :param f:
            [S, N]
        :returns: tuple of
            tf.Tensor [S, 1]
            tf.Tensor [S, 1]
            tf.Tensor [S, N]
        """
        y_sigma = tf.exp(log_y_sigma)
        amp = tf.exp(log_amp)
        if data_only:
            # L_ij f_sj -> f_sj L_ji
            dtec = amp * tf.matmul(f, self.L_data, transpose_b=True)
        else:
            # L_ij f_sj -> f_sj L_ji
            dtec = amp * tf.matmul(f, self.L, transpose_b=True)

        return self.Params(amp=amp, y_sigma=y_sigma, dtec=dtec)


    def forward_equation(self, dtec):
        """
        Calculate real and imaginary parts of gains from dtec.

        :param dtec: float_type, Tensor [b0,...,bB]
            The DTECs
        :returns: tuple
            float_type, tf.Tensor [b0,...,bB,Nf] Real part
            float_type, tf.Tensor [b0,...,bB,Nf] Imag part
        """
        #..., Nf
        phase = dtec[..., None] * self.invfreqs
        real_part = tf.cos(phase)
        imag_part = tf.sin(phase)
        return real_part, imag_part

    def log_prob(self, log_amp, log_y_sigma, f):
        """
        Calculate the log probability of the gains given a model.

        :param amp: float_type tf.Tensor [num_chains, 1]
            Unconstrained amp
        :param y_sigma: float_type tf.Tensor [num_chains, 1]
            Unconstrained y_sigma
        :param dtec: float_type tf.Tensor [num_chains, N+Ns]
            Unconstrained dtec

        :return: float_type, tf.Tensor, [num_chains]
            The log-probability of the data given model.
        """
        # num_chains = tf.shape(f)[0]
        # shuffle = tf.random.shuffle(tf.range(num_chains))
        # log_y_sigma = tf.gather(log_y_sigma, shuffle, axis=0)
        # log_amp = tf.gather(log_amp, shuffle, axis=0)
        # f = tf.gather(f, shuffle, axis=0)
        # print('f',f)

        #TODO: once working try with only data prior
        transformed = self.transform_state(log_amp, log_y_sigma, f, data_only=True)

        # num_chains
        prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f),
                                                         scale_identity_multiplier=1.).log_prob(f)

        # phase_model = transformed.dtec[:,:,None]*self.invfreqs
        # Yimag_model = tf.sin(phase_model)
        # Yreal_model = tf.cos(phase_model)
        #TODO: do slicing on L first reduce complexity
        Yreal_model, Yimag_model = self.forward_equation(transformed.dtec)

        likelihood = -tf.math.reciprocal(transformed.y_sigma[:, :, None]) * sqrt_with_finite_grads(
            tf.math.square(self.Y_imag[None, :, :] - Yimag_model) + tf.math.square(
                self.Y_real[None, :, :] - Yreal_model)) - log_y_sigma[:, :, None]

        # # num_chains, N, Nf
        # likelihood = tfp.distributions.Laplace(loc=self.Y_imag[None, :, :], scale=transformed.y_sigma[:, :, None]).log_prob(
        #     Yimag_model) \
        #              + tfp.distributions.Laplace(loc=self.Y_real[None, :, :], scale=transformed.y_sigma[:, :, None]).log_prob(
        #     Yreal_model)

        #num_chains
        y_sigma_prior = tfp.distributions.Normal(
            loc=tf.constant(0.1, dtype=float_type), scale=tf.constant(0.1, dtype=float_type)).log_prob(transformed.y_sigma[:, 0])

        # num_chains
        logp = tf.reduce_sum(likelihood, axis=[1, 2]) + prior
        # + y_sigma_prior
        # print('logp',logp)

        return logp
