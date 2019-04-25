import tensorflow as tf
import tensorflow_probability as tfp
from .feeds import init_feed, DataFeed, CoordinateFeed, ContinueFeed, CoordinateDimFeed, DatapackFeed
from .settings import float_type
from collections import namedtuple
from .targets import DTECToGains, DTECToGainsSAEM
from .misc import flatten_batch_dims, timer, random_sample, graph_store_get, graph_store_set, dict2namedtuple
from .hyper_parameter_opt import KernelHyperparameterSolveCallback
from . import logging
import gpflow as gp
import numpy as np

from .hyper_parameter_opt import KernelHyperparameterSolveCallback
from .callbacks import DatapackStoreCallback, GetLearnIndices
# from .misc import plot_graph
# import pylab as plt


SampleParams = namedtuple('SampleParams',['num_leapfrog_steps', 'num_adapation_steps', 'target_rate', 'num_samples', 'num_burnin_steps'])

class FreeTransitionSAEM(object):
    def __init__(self, datapack_feed: DatapackFeed):
        self.datapack_feed = datapack_feed

    def init_filter(self, init_kern_hyperparams={}, initial_stepsize=5e-3):
        self.full_block_size = (self.datapack_feed.coord_feed.N + self.datapack_feed.star_coord_feed.N) * self.datapack_feed.coord_feed.time_feed.slice_size
        temp_target = DTECToGainsSAEM(initial_hyperparams=init_kern_hyperparams)
        self.hyperparams_var = temp_target.variables
        graph_store_set('hyperparms_var',self.hyperparams_var)

        self.step_sizes = [tf.get_variable(
                            name='step_size_dtec',
                            initializer=lambda: tf.convert_to_tensor(initial_stepsize, dtype=float_type),
                            use_resource=True,
                            dtype=float_type,
                            trainable=False)]


        self.datapack_feed_iterator = tf.data.Dataset.zip(self.datapack_feed.index_feed, self.datapack_feed.feed).make_initializable_iterator()

        graph_store_set('init0',tf.group([self.datapack_feed_iterator.initializer] + [self.hyperparams_var.initializer] \
                         + [step_size.initializer for step_size in self.step_sizes]))

    @property
    def initializer(self):
        return graph_store_get('init0')

    def filter_step(self,
                    dtec_warmstart,
                    hyperparams_warmstart,
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



        (index, next_index), ((Y_real, Y_imag), freqs, X, Xstar, X_dim, Xstar_dim, cont) = self.datapack_feed_iterator.get_next()
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

        # update_variables = saem_step(variables)
        t0 = timer()
        with tf.control_dependencies([t0]):

            target = DTECToGainsSAEM(variables=hyperparams_warmstart)
            target.setup_target(X, Xstar, Y_real, Y_imag, freqs,
                                     fed_kernel='RBF', obs_type='DDTEC',which_kernel=which_kernel,
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
                # last state as initial point (mean of each chain)
                samples, (stepsize, target_log_prob) = tfp.mcmc.sample_chain(
                    num_results=sample_params.num_samples,
                    num_burnin_steps=sample_params.num_burnin_steps,
                    trace_fn=trace_step,
                    return_final_kernel_results=True,
                    current_state=[dtec_warmstart],
                    kernel=hmc,
                    parallel_iterations=parallel_iterations)

                next_dtec_warmstart = tf.reduce_mean(samples,axis=0)

                with tf.control_dependencies([samples[0]]):
                    t1 = timer()



                # rhat = tfp.mcmc.potential_scale_reduction(samples)
                # ess = tfp.mcmc.effective_sample_size(samples)
                flat_samples = flatten_batch_dims(samples[0])

                # test_logp = tf.reduce_mean(target.logp(flat_samples[:,N:]))/tf.cast(Ns,float_type)
                # post_logp = tf.reduce_mean(target_log_prob / tf.cast(N + Ns, float_type))
                #tf.reduce_mean(target.logp(flat_samples)) / tf.cast(N + Ns, float_type)

                ddtec_transformed = target.transform_samples(flat_samples)

                Y_real_samples, Y_imag_samples = target.forward_equation(ddtec_transformed)

                # avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                #                                   name='avg_acc_ratio')
                # posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                #                                name='marginal_log_likelihood')/tf.cast(N+Ns,float_type)


                idx_learn = GetLearnIndices(dist_cutoff=0.3)(X[:,4:7])
                X_learn = tf.gather(X,idx_learn,axis=0)
                ddtec_transformed_data = ddtec_transformed[:, :N]
                ddtec_mean = tf.reduce_mean(ddtec_transformed_data,axis=0)
                ddtec_var = tf.reduce_mean(tf.square(ddtec_transformed_data)) - tf.square(ddtec_mean)
                ddtec_mean = tf.gather(ddtec_mean, idx_learn,axis=0)
                ddtec_var = tf.gather(ddtec_var, idx_learn,axis=0)

                hyperparam_opt_callback = KernelHyperparameterSolveCallback(
                    resolution=kernel_params.get('resolution', 5),
                    maxiter=100,
                    obs_type='DDTEC',
                    fed_kernel=kernel_params.get('fed_kernel', 'RBF'))

                learned_hyperparams = hyperparam_opt_callback(X_learn, ddtec_mean, ddtec_var,
                                                              target.constrained_hyperparams.variance,
                                                              target.constrained_hyperparams.lengthscales,
                                                              target.constrained_hyperparams.a,
                                                              target.constrained_hyperparams.b,
                                                              target.constrained_hyperparams.timescale)


                next_hyperparams_warmstart = target.stack_state(target.unconstrained_state(
                    target.DTECToGainsParams(target.constrained_hyperparams.y_sigma, *learned_hyperparams)))

                # self.update_hyperparams = tf.assign(self.hyperparams_var, learned_unconstrained_vars)
                with tf.control_dependencies([next_hyperparams_warmstart]):
                    t2 = timer()

        graph_store_set('sample_time',t1-t0)
        graph_store_set('hyperparam_opt_time',t2-t1)



        def _percentiles(t, q=[15.,50.,85.]):
            """
            Returns the percentiles down `axis` stacked on first axis.

            :param t: float_type, tf.Tensor, [S, f0, ..., fF]
                tensor to get percentiles for down first axis
            :param q: list of float_type
                Percentiles
            :return: float_type, tf.Tensor, [len(q), f0, ... ,fF]
            """
            return tfp.stats.percentile(t, q, axis=0)

        # TODO: experiment with means and var instead of median
        # 3, N+Ns
        dtec_post = _percentiles(ddtec_transformed, [15.,50.,85.])

        # # 3, N+Ns, Nf
        # phase_post = _percentiles(tf.atan2(Y_imag_samples, Y_real_samples))

        Y_real_post = _percentiles(Y_real_samples, [50.])
        Y_imag_post = _percentiles(Y_imag_samples, [50.])
        effective_phase = tf.atan2(Y_imag_post[0,:,:], Y_real_post[0,:,:])

        Posterior = namedtuple('Solutions', ['tec', 'phase', 'weights_tec'])

        data_posterior = Posterior(
            tec = tf.reshape(dtec_post[1,:N], X_dim),
            phase = tf.reshape(effective_phase[:N, :], tf.concat([X_dim, [-1]],axis=0)),
            weights_tec=tf.reshape(dtec_post[2, :N] - dtec_post[0, :N], X_dim),
        )

        screen_posterior = Posterior(
            tec=tf.reshape(dtec_post[1, N:], Xstar_dim),
            phase=tf.reshape(effective_phase[N:, :], tf.concat([Xstar_dim, [-1]], axis=0)),
            weights_tec=tf.reshape(dtec_post[2, N:] - dtec_post[0, N:], Xstar_dim),
        )

        FilterResult = namedtuple('FilterResult', ['data_posterior', 'screen_posterior', 'next_dtec_warmstart', 'next_hyperparams_warmstart', 'index', 'next_index', 'cont'])


        output = FilterResult(
            data_posterior = data_posterior,
            screen_posterior = screen_posterior,
            next_dtec_warmstart=next_dtec_warmstart,
            next_hyperparams_warmstart=next_hyperparams_warmstart,
            index=index,
            next_index=next_index,
            cont=cont)
        return output

    def filter(self,
               num_chains = 1,
               num_samples=10,
               parallel_iterations=10,
               num_leapfrog_steps=2,
               target_rate=0.6,
               num_burnin_steps=100,
               which_kernel=0,
               kernel_params={},
               num_adapation_steps=100,
               num_parallel_filters=1,
               tf_seed=None
               ):

        def body(cont, dtec_warmstart, hyperparams_warmstart):
            results = self.filter_step(dtec_warmstart,
                             hyperparams_warmstart,
                             num_samples=num_samples,
                             parallel_iterations=parallel_iterations,
                             num_leapfrog_steps=num_leapfrog_steps,
                             target_rate=target_rate,
                             num_burnin_steps=num_burnin_steps,
                             which_kernel=which_kernel,
                             kernel_params=kernel_params,
                             num_adapation_steps=num_adapation_steps)


            store_arrays = [results.data_posterior.tec,
                            results.data_posterior.phase,
                            results.data_posterior.weights_tec,
                            results.screen_posterior.tec,
                            results.screen_posterior.phase,
                            results.screen_posterior.weights_tec
                            ]

            store_dtec_callbacks = [DatapackStoreCallback(self.datapack_feed.datapack,
                                                          self.datapack_feed.posterior_solset,'tec',
                                                        **self.datapack_feed.selection),
                                    DatapackStoreCallback(self.datapack_feed.datapack,
                                                        self.datapack_feed.posterior_solset, 'phase',
                                                        **self.datapack_feed.selection),
                                    DatapackStoreCallback(self.datapack_feed.datapack,
                                                        self.datapack_feed.posterior_solset, 'weights_tec',
                                                        **self.datapack_feed.selection),
                                    DatapackStoreCallback(self.datapack_feed.datapack,
                                                          self.datapack_feed.screen_solset, 'tec',
                                                          **self.datapack_feed.selection),
                                    DatapackStoreCallback(self.datapack_feed.datapack,
                                                          self.datapack_feed.screen_solset, 'phase',
                                                          **self.datapack_feed.selection),
                                    DatapackStoreCallback(self.datapack_feed.datapack,
                                                          self.datapack_feed.screen_solset, 'weights_tec',
                                                          **self.datapack_feed.selection)
                                    ]
            lock = tf.no_op()
            store_ops = []
            for array, callback in zip(store_arrays, store_dtec_callbacks):
                with tf.control_dependencies([lock]):
                    store_ops.append(callback(results.index, results.next_index, array))
                    lock = store_ops[-1]
            with tf.control_dependencies([lock]):
                return [tf.identity(results.cont), results.next_dtec_warmstart, results.next_hyperparams_warmstart]

        def cond(cont, dtec_warmstart, hyperparams_warmstart):
            return cont

        dtec_init = tf.random_normal(shape=[num_chains, self.full_block_size],dtype=float_type, seed=tf_seed)
        hyperparams_init = self.hyperparams_var
        cont, dtec_out, hyperparams_out = tf.while_loop(cond,
                      body,
                      [tf.constant(True), dtec_init, hyperparams_init],
                      parallel_iterations=num_parallel_filters)

        return tf.group([cont, dtec_out, hyperparams_out])


