import tensorflow as tf
import tensorflow_probability as tfp
from .feeds import init_feed, DataFeed, CoordinateFeed, ContinueFeed, CoordinateDimFeed
from .settings import float_type
from collections import namedtuple
from .targets import DTECToGains, DTECToGainsSAEM
from .misc import flatten_batch_dims, timer, random_sample, graph_store_get, graph_store_set, dict2namedtuple
from .hyper_parameter_opt import py_function_optimise_hyperparams
from . import logging
import gpflow as gp
import numpy as np

# from .misc import plot_graph
# import pylab as plt


SampleParams = namedtuple('SampleParams',['num_leapfrog_steps', 'num_adapation_steps', 'target_rate', 'num_samples', 'num_burnin_steps'])

class FreeTransitionSAEM(object):
    def __init__(self, freqs, data_feed: DataFeed, coord_feed: CoordinateFeed, star_coord_feed: CoordinateFeed):
        self.coord_feed = coord_feed
        self.coord_dim_feed = CoordinateDimFeed(self.coord_feed)
        self.star_coord_feed = star_coord_feed
        self.star_coord_dim_feed = CoordinateDimFeed(self.star_coord_feed)
        self.data_feed = data_feed
        self.continue_feed = ContinueFeed(self.coord_feed.time_feed)
        self.freqs = tf.convert_to_tensor(freqs,float_type)


    def init_filter(self, init_kern_hyperparams={}, initial_stepsize=5e-3):
        self.full_block_size = (self.coord_feed.N + self.star_coord_feed.N) * self.coord_feed.time_feed.slice_size
        temp_target = DTECToGainsSAEM(initial_hyperparams=init_kern_hyperparams)
        self.hyperparams_var = temp_target.variables
        graph_store_set('hyperparms_var',self.hyperparam_vars)
        init_index_adjust = graph_store_get('index_adjust').initializer
        self.index_inc_func = self.coord_feed.time_feed.index_feed.inc_adjustment

        self.step_sizes = [tf.get_variable(
                            name='step_size_dtec',
                            initializer=lambda: tf.convert_to_tensor(initial_stepsize, dtype=float_type),
                            use_resource=True,
                            dtype=float_type,
                            trainable=False)]

        joint_dataset = tf.data.Dataset.zip((self.data_feed.feed,
                                             self.coord_feed.feed,
                                             self.star_coord_feed.feed,
                                             self.continue_feed.feed,
                                             self.coord_dim_feed.feed,
                                             self.star_coord_dim_feed.feed))

        self.joint_iterator = joint_dataset.make_initializable_iterator()

        graph_store_set('init0',init_index_adjust)
        graph_store_set('init1',tf.group([self.joint_iterator.initializer] + [self.hyperparam_vars.initializer] \
                         + [step_size.initializer for step_size in self.step_sizes]))

    def filter_step(self,
                    dtec_warmstart,
                    num_samples=10,
                    parallel_iterations=10,
                    num_leapfrog_steps=2,
                    target_rate=0.6,
                    num_burnin_steps=100,
                    which_kernel=0,
                    kernel_params={},
                    num_adapation_steps=100):
        """

        :param dtec_warmstart: float_type, tf.Tensor
            The warm start DDTEC of data and screen of shape [num_chains, N + Ns]
        :param num_samples:
        :param num_chains:
        :param parallel_iterations:
        :param num_leapfrog_steps:
        :param target_rate:
        :param num_burnin_steps:
        :param num_saem_samples:
        :param saem_maxsteps:
        :param initial_stepsize: float
            The initial step size for dtec samples, should be ~stddev of the samples.
        :param init_kern_hyperparams:
        :param which_kernel:
        :param kernel_params:
        :param saem_batchsize:
        :param slice_size:
        :param saem_population:
        :return:
        """

        sample_params = SampleParams(num_leapfrog_steps=num_leapfrog_steps, #tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0]
                                   num_adapation_steps=num_adapation_steps,
                                   target_rate=target_rate,
                                   num_samples=num_samples,
                                   num_burnin_steps=num_burnin_steps
                                   )
        if sample_params.num_burnin_steps <= sample_params.num_adapation_steps:
            logging.warn("Number of burnin steps ({}) should be higher than number of adaptation steps ({}).".format(sample_params.num_burnin_steps, sample_params.num_adapation_steps))



        (Y_real, Y_imag), X, Xstar, cont, X_dim, Xstar_dim = self.joint_iterator.get_next()
        N = tf.shape(X)[0]
        Ns = tf.shape(Xstar)[0]

        asserts = graph_store_set('valid_warmstart', tf.assert_equal(N+Ns, tf.shape(dtec_warmstart)[-2]))

        # # Nf
        # invfreqs = -8.448e9 * tf.reciprocal(self.freqs)
        # phase = tf.atan2(Y_imag, Y_real)
        # # N
        # dtec_init_data = tf.reduce_mean(phase / invfreqs, axis=-1)
        # dtec_screen_init = tf.zeros(tf.shape(Xstar)[0:1],float_type)
        # dtec_init = tf.concat([dtec_init_data,dtec_screen_init],axis=0)
        # dtec_init = target.get_initial_point(dtec_init)
        # q0_init = [tf.tile(tf.reshape(dtec_init, (-1,))[None, :], (num_chains, 1))]

        def _percentiles(t, q=[10,50,90]):
            """
            Returns the percentiles down `axis` stacked on first axis.

            :param t: float_type, tf.Tensor, [S, f0, ..., fF]
                tensor to get percentiles for down first axis
            :param q: list of float_type
                Percentiles
            :return: float_type, tf.Tensor, [len(q), f0, ... ,fF]
            """
            return tfp.stats.percentile(t, q, axis=0)

        # update_variables = saem_step(variables)
        t0 = timer()
        with tf.control_dependencies([t0]):

            target = DTECToGainsSAEM(X, Xstar, Y_real, Y_imag, self.freqs,
                                     fed_kernel='RBF', obs_type='DDTEC',
                                     variables=self.hyperparams_var, which_kernel=which_kernel,
                                     kernel_params=kernel_params, L=None, full_posterior=True)

            with tf.control_dependencies([tf.print("Sampling with m:",target.constrained_hyperparams)]):

                def trace_step(states, previous_kernel_results):
                    """Trace the transformed dtec, stepsize, log_acceptance, log_prob"""
                    return previous_kernel_results.extra.step_size_assign, previous_kernel_results.target_log_prob
                    # dtec_constrained = target.transform_samples(states[0])
                    # return dtec_constrained, \
                    #        previous_kernel_results.step_size, \
                    #        previous_kernel_results.log_acceptance_correction, \
                    #        previous_kernel_results.target_log_prob

                ###
                hmc = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target.logp,
                    num_leapfrog_steps=sample_params.num_leapfrog_steps,
                    step_size=self.step_sizes,
                    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=sample_params.num_adapation_steps,
                                                                                     decrement_multiplier=0.1,
                                                                                     increment_multiplier=0.1,
                                                                                     target_rate=sample_params.target_rate),
                    state_gradients_are_stopped=True)

                # Run the chain (with burn-in maybe).
                samples, stepsize, target_log_prob = tfp.mcmc.sample_chain(
                    num_results=sample_params.num_samples,
                    num_burnin_steps=sample_params.num_burnin_steps,
                    trace_fn=trace_step,
                    return_final_kernel_results=False,
                    current_state=[dtec_warmstart],
                    kernel=hmc,
                    parallel_iterations=parallel_iterations)

                with tf.control_dependencies([samples[0]]):
                    t1 = timer()

                # last state as initial point

                rhat = tfp.mcmc.potential_scale_reduction(samples)
                ess = tfp.mcmc.effective_sample_size(samples)
                flat_samples = flatten_batch_dims(samples[0])

                # test_logp = tf.reduce_mean(target.logp(flat_samples[:,N:]))/tf.cast(Ns,float_type)
                post_logp = tf.reduce_mean(target_log_prob / tf.cast(N + Ns, float_type))
                #tf.reduce_mean(target.logp(flat_samples)) / tf.cast(N + Ns, float_type)

                ddtec_transformed = target.transform_samples(flat_samples)
                Y_real_samples, Y_imag_samples = target.forward_equation(ddtec_transformed)

                # avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                #                                   name='avg_acc_ratio')
                # posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                #                                name='marginal_log_likelihood')/tf.cast(N+Ns,float_type)

                def _get_learn_indices(X, cutoff=0.3):
                    """Get the indices of non-redundant antennas
                    :param X: np.array, float64, [N, 3]
                        Antenna locations
                    :param cutoff: float
                        Mark redundant if antennas within this in km
                    :return: np.array, int64
                        indices such that all antennas are at least cutoff apart
                    """
                    X = X.numpy()
                    N = X.shape[0]
                    Xa, inverse = np.unique(X, return_inverse=True, axis=0)
                    Na = len(Xa)
                    keep = []
                    for i in range(Na):
                        if np.all(np.linalg.norm(Xa[i:i + 1, :] - Xa[keep, :], axis=1) > cutoff):
                            keep.append(i)
                    logging.info("Training on antennas: {}".format(keep))
                    return (np.where(np.isin(inverse, keep, assume_unique=True))[0]).astype(np.int64)

                idx_learn = tf.py_function(_get_learn_indices, [X[:,4:7]], [tf.int64])
                X_learn = tf.gather(X,idx_learn,axis=0)
                ddtec_mean = tf.reduce_mean(ddtec_transformed[:,:N],axis=0)
                ddtec_var = tf.reduct_mean(tf.square(ddtec_transformed[:,:N])) - tf.square(ddtec_mean)
                ddtec_mean = tf.gather(ddtec_mean, idx_learn,axis=0)
                ddtec_var = tf.gather(ddtec_var, idx_learn,axis=0)

                learned_hyperparams = py_function_optimise_hyperparams(X_learn, ddtec_mean, ddtec_var,
                                                 target.constrained_hyperparams)
                learned_unconstrained_vars = target.stack_state(target.unconstrained_state(learned_hyperparams))

                self.update_hyperparams = tf.assign(self.hyperparams_var, learned_unconstrained_vars)
                with tf.control_dependencies([self.update_hyperparams]):
                    self.updated_hyperparams_var = tf.identity(self.update_hyperparams)
                    t2 = timer()

        graph_store_set('sample_time',t1-t0)
        graph_store_set('hyperparam_opt_time',t2-t1)

        TAResult = namedtuple('TAResult', ['hyperparam_opt_op','hyperparams', 'hyperparams_var','dtec', 'Y_real', 'Y_imag','post_logp',
                                           'dtec_star', 'Y_real_star', 'Y_imag_star',
                                           'cont', 'ess', 'rhat','extra','sample_time', 'phase', 'phase_star', 'stepsize'])

        ExtraResults = namedtuple('ExtraResults',['Y_real_data', 'Y_imag_data', 'X', 'Xstar', 'X_dim', 'Xstar_dim', 'freqs'])

        dtec_post = _percentiles(ddtec_transformed)
        dtec_X = tf.reshape(dtec_post[:,:N], tf.concat([[3], X_dim],axis=0))
        dtec_Xstar = tf.reshape(dtec_post[:,N:], tf.concat([[3], Xstar_dim],axis=0))

        phase_post = _percentiles(tf.atan2(Y_imag_samples, Y_real_samples))
        phase_X = tf.reshape(phase_post[:,:N,:], tf.concat([[3], X_dim, [-1]],axis=0))
        phase_Xstar = tf.reshape(phase_post[:, N:,:], tf.concat([[3], Xstar_dim, [-1]], axis=0))

        Y_real_post = _percentiles(Y_real_samples)
        Y_real_X = tf.reshape(Y_real_post[:,:N,:], tf.concat([[3], X_dim, [-1]],axis=0))
        Y_real_Xstar = tf.reshape(Y_real_post[:, N:,:], tf.concat([[3], Xstar_dim, [-1]], axis=0))

        Y_imag_post = _percentiles(Y_imag_samples)
        Y_imag_X = tf.reshape(Y_imag_post[:, :N,:], tf.concat([[3], X_dim, [-1]], axis=0))
        Y_imag_Xstar = tf.reshape(Y_imag_post[:, N:,:], tf.concat([[3], Xstar_dim, [-1]], axis=0))

        Y_real = tf.reshape(Y_real, tf.concat([X_dim, [-1]],axis=0))
        Y_imag = tf.reshape(Y_imag, tf.concat([X_dim, [-1]], axis=0))

        output = TAResult(
            hyperparam_opt_op = self.update_hyperparams,
            hyperparams = learned_hyperparams,
            hyperparams_var=self.updated_hyperparams_var,
            dtec = dtec_X,
            Y_real = Y_real_X,
            Y_imag = Y_imag_X,
            dtec_star = dtec_Xstar,
            Y_real_star = Y_real_Xstar,
            Y_imag_star = Y_imag_Xstar,
            # acc_ratio = avg_acceptance_ratio,
            post_logp = post_logp,
            # test_logp = posterior_log_prob,
            cont = cont,
            # step_sizes = kernel_results.extra.step_size_assign[0],
            ess = ess[0],
            rhat = rhat[0],
            extra = ExtraResults(Y_real, Y_imag, X, Xstar, X_dim, Xstar_dim, self.freqs),
            sample_time=t1-t0,
            phase=phase_X,
            phase_star=phase_Xstar,
            stepsize=stepsize)
        return output

    def filter(self, num_chains = 1,
               num_samples=10,
               parallel_iterations=10,
               num_leapfrog_steps=2,
               target_rate=0.6,
               num_burnin_steps=100,
               which_kernel=0,
               kernel_params={},
               num_adapation_steps=100
               ):

        def body(cont, dtec_warmstart):
            results = self.filter_step(dtec_warmstart,
                             num_samples=num_samples,
                             parallel_iterations=parallel_iterations,
                             num_leapfrog_steps=2,
                             target_rate=0.6,
                             num_burnin_steps=100,
                             which_kernel=0,
                             kernel_params={},
                             num_adapation_steps=100)

        def cond(cont, dtec_warmstart):
            return cont

        dtec_init = tf.zeros(shape=[num_chains, self.full_block_size],dtype=float_type)
        cont, dtec_out = tf.while_loop(cond,
                      body,
                      [tf.constant(True), dtec_init],
                      parallel_iterations=2)

