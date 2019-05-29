import tensorflow as tf
import tensorflow_probability as tfp
from .feeds import DatapackFeed
from .settings import float_type
from collections import namedtuple
from .targets import DTECToGainsTarget
from .processes import DTECProcess
from .sample import sample_chain
from .misc import flatten_batch_dims, timer
from . import logging
import os
import numpy as np
from .misc import safe_cholesky
from threading import Lock
from .vi import VariationalBayes, WhitenedVariationalPosterior, VariationalBayesHeirarchical

from .hyper_parameter_opt import KernelHyperparameterSolveCallback
from .callbacks import callback_sequence, Chain, DatapackStoreCallback, GetLearnIndices, PlotResults, PlotResultsV2, \
    StoreHyperparameters, StoreHyperparametersV2, PlotAcceptRatio, PlotEss, PlotRhat, PlotStepsizes, PlotELBO

SampleParams = namedtuple('SampleParams',['num_leapfrog_steps', 'num_adapation_steps', 'target_rate', 'num_samples', 'num_burnin_steps'])
HMCParams = namedtuple('HMCParams',['amp','y_sigma', 'dtec'])
DTECBijection = namedtuple('DTECBijection',['Lp','mp'])

class FreeTransitionSAEM(object):
    def __init__(self, datapack_feed: DatapackFeed, output_folder='./output_folder'):
        self.datapack_feed = datapack_feed
        self._output_folder = os.path.abspath(output_folder)
        logging.info("Using output directory: {}".format(output_folder))
        os.makedirs(self._output_folder,exist_ok=True)

    @property
    def output_folder(self):
        return self._output_folder

    @property
    def plot_folder(self):
        return os.path.join(self.output_folder, 'plot_folder')

    @property
    def hyperparam_store(self):
        return os.path.join(self.output_folder,'hyperparam_store.npz')

    @property
    def initializer(self):
        return self.init0

    def init_filter(self,
                    num_chains=1,
                    init_kern_hyperparams=dict(),
                    init_stepsizes=dict(amp=0.093, y_sigma=0.015, dtec= 0.39),
                    tf_seed=0):

        self.full_block_size = (self.datapack_feed.coord_feed.N + self.datapack_feed.star_coord_feed.N) * self.datapack_feed.time_feed.slice_size
        temp_dtec_process = DTECProcess(initial_hyperparams=init_kern_hyperparams)
        self.init_hyperparams = temp_dtec_process.hyperparams

        init_amp, init_y_sigma, init_dtec = DTECToGainsTarget.init_variables(num_chains, self.full_block_size, tf_seed)

        self.init_params = HMCParams(amp=init_amp,
                                     y_sigma=init_y_sigma,
                                     dtec=init_dtec)

        self.init_stepsizes = HMCParams(
            amp=tf.convert_to_tensor(init_stepsizes['amp'],dtype=float_type),
            y_sigma=tf.convert_to_tensor(init_stepsizes['y_sigma'], dtype=float_type),
            dtec=tf.convert_to_tensor(init_stepsizes['dtec'], dtype=float_type))



        self.datapack_feed_iterator = tf.data.Dataset.zip((self.datapack_feed.index_feed.feed, self.datapack_feed.feed)).make_initializable_iterator()

        self.init0 = self.datapack_feed_iterator.initializer

    def filter_step(self,
                    params_warmstart,
                    hyperparams_warmstart,
                    stepsize_warmstart,
                    num_samples=10,
                    parallel_iterations=10,
                    num_leapfrog_steps=2,
                    target_rate=0.6,
                    num_burnin_steps=100,
                    recalculate_period=1,
                    kernel_params=dict(resolution=5),
                    hyperparam_opt_params=dict(maxiter=100),
                    obs_type='DDTEC',
                    fed_kernel='RBF',
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

        def _maybe_cast(val, dtype):
            if val is not None:
                return tf.cast(val, dtype)
            return None

        sample_params = SampleParams(num_leapfrog_steps=_maybe_cast(num_leapfrog_steps, dtype=tf.int32), #tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0]
                                   num_adapation_steps=_maybe_cast(num_adapation_steps, dtype=tf.int32),
                                   target_rate=_maybe_cast(target_rate, dtype=tf.float64),
                                   num_samples=_maybe_cast(num_samples, dtype=tf.int32),
                                   num_burnin_steps=_maybe_cast(num_burnin_steps, dtype=tf.int32)
                                   )
        recalculate_period = tf.convert_to_tensor(recalculate_period,dtype=tf.int32)

        if num_adapation_steps is not None:
            if num_burnin_steps <= num_adapation_steps:
                logging.warn("Number of burnin steps ({}) should be higher than number of adaptation steps ({}).".format(num_burnin_steps, num_adapation_steps))
        else:
            logging.warn("Chain is not stationary because adaptation happens continuously.")



        (index, next_index), ((Y_real, Y_imag), freqs, X, Xstar, X_dim, Xstar_dim, cont) = self.datapack_feed_iterator.get_next()
        N = tf.shape(X)[0]
        Ns = tf.shape(Xstar)[0]

        # asserts = graph_store_set('valid_warmstart', tf.assert_equal(N+Ns, tf.shape(params_warmstart.dtec)[-2]))

        t0 = timer()
        with tf.control_dependencies([t0]):
            dtec_process = DTECProcess(variables=hyperparams_warmstart)

            dtec_process.setup_process(X,Xstar,fed_kernel=fed_kernel,obs_type=obs_type, kernel_params=kernel_params,
                                       recalculate_prior=True,#tf.equal(tf.math.mod(index, recalculate_period), 0),
                                       L = None,
                                       m = None)

            target = DTECToGainsTarget(dtec_process = dtec_process)
            target.setup_target(Y_real, Y_imag, freqs, full_posterior=True)

            with tf.control_dependencies([tf.print("Sampling with hyperparams:",dtec_process.constrained_hyperparams,
                                                   'Stepsizes:',stepsize_warmstart)]):
                override_stepsizes = self.init_stepsizes
                    # [ tf.constant(0.13, dtype=float_type), tf.constant(0.014, dtype=float_type),
                    #          tf.constant(0.9, dtype=float_type)]

                ###
                hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target.log_prob,
                    num_leapfrog_steps=sample_params.num_leapfrog_steps,
                    step_size=override_stepsizes),#list(stepsize_warmstart)),
                    num_adaptation_steps=sample_params.num_adapation_steps,
                    target_accept_prob=sample_params.target_rate,
                    adaptation_rate=0.05)

                # Run the chain (with burn-in maybe).
                # last state as initial point (mean of each chain)

                def trace_fn(_, pkr):
                    # print(pkr)
                    return (pkr.inner_results.log_accept_ratio,
                            pkr.inner_results.accepted_results.step_size)

                samples, (log_accept_ratio, stepsizes) = sample_chain(
                    num_results=sample_params.num_samples,
                    num_burnin_steps=sample_params.num_burnin_steps,
                    trace_fn=trace_fn,
                    return_final_kernel_results=False,
                    current_state=list(params_warmstart),
                    kernel=hmc,
                    parallel_iterations=parallel_iterations)

                stepsizes = HMCParams(*stepsizes)

                samples = HMCParams(*samples)

                next_stepsize_warmstart = []
                next_param_warmstart = []
                for s in samples:
                    # mean standard deviation per parameter
                    next_stepsize_warmstart.append(tf.sqrt(tf.reduce_mean(tfp.stats.variance(s))))#
                    next_param_warmstart.append(tf.reduce_mean(s,axis=0))#num_chains, dims
                next_stepsize_warmstart = HMCParams(*next_stepsize_warmstart)
                next_param_warmstart = HMCParams(*next_param_warmstart)


                with tf.control_dependencies(list(samples)):
                    t1 = timer()

                rhat = HMCParams(*tfp.mcmc.potential_scale_reduction(samples))
                ess = HMCParams(*tfp.mcmc.effective_sample_size(samples))

                flat_dtec_samples = flatten_batch_dims(samples.dtec)
                flat_y_sigma_samples = flatten_batch_dims(samples.y_sigma)
                flat_amp_samples = flatten_batch_dims(samples.amp)

                #[S, 1] [S, 1] [S, N]
                transformed_samples = target.transform_state(flat_amp_samples, flat_y_sigma_samples, flat_dtec_samples)

                Y_real_samples, Y_imag_samples = target.forward_equation(transformed_samples.dtec)

                # avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                #                                   name='avg_acc_ratio')
                # posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                #                                name='marginal_log_likelihood')/tf.cast(N+Ns,float_type)

                idx_learn = GetLearnIndices(dist_cutoff=0.1)(X[:,4:7])[0]
                X_learn = tf.gather(X,idx_learn,axis=0)
                dtec_transformed_data = transformed_samples.dtec[:, :N]
                dtec_mean = tfp.stats.percentile(dtec_transformed_data,50,axis=0)
                dtec_var = tfp.stats.variance(dtec_transformed_data, sample_axis=0)#tf.reduce_mean(tf.square(dtec_transformed_data), axis=0) - tf.square(dtec_mean)
                dtec_mean = tf.gather(dtec_mean, idx_learn,axis=0)
                dtec_var = tf.gather(dtec_var, idx_learn,axis=0)

                hyperparam_opt_callback = KernelHyperparameterSolveCallback(
                    resolution=kernel_params.get('resolution', 5),
                    maxiter=hyperparam_opt_params.get('maxiter',100),
                    obs_type=obs_type,
                    fed_kernel=fed_kernel)

                learned_hyperparams = hyperparam_opt_callback(X_learn, dtec_mean, dtec_var,
                                                              tfp.stats.percentile(tf.square(transformed_samples.amp),50,axis=0, keep_dims=True),
                                                              dtec_process.constrained_hyperparams.lengthscales,
                                                              dtec_process.constrained_hyperparams.a,
                                                              dtec_process.constrained_hyperparams.b,
                                                              dtec_process.constrained_hyperparams.timescale)


                next_hyperparams_warmstart = dtec_process.stack_state(dtec_process.unconstrained_state(
                    dtec_process.Params(*learned_hyperparams)))
                solved_hyperparams = dtec_process.stack_state(dtec_process.Params(*learned_hyperparams))
                next_hyperparams_warmstart.set_shape(hyperparams_warmstart.shape)

                # self.update_hyperparams = tf.assign(self.hyperparams_var, learned_unconstrained_vars)
                with tf.control_dependencies([next_hyperparams_warmstart]):
                    t2 = timer()

        # graph_store_set('sample_time',t1-t0)
        # graph_store_set('hyperparam_opt_time',t2-t1)

        def reduce_median(X, axis=0, keepdims=False):
            return tfp.stats.percentile(X,50, axis=axis,keep_dims=keepdims)

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

        def _posterior(t):
            """
            Returns the percentiles down `axis` stacked on first axis.

            :param t: float_type, tf.Tensor, [S, f0, ..., fF]
                tensor to get percentiles for down first axis
            :param q: list of float_type
                Percentiles
            :return: float_type, tf.Tensor, [len(q), f0, ... ,fF]
            """
            mean = tf.reduce_mean(t, axis=0)
            var = tfp.stats.variance(t, sample_axis=0)
            return mean, var

        # TODO: experiment with means and var instead of median
        # 3, N+Ns
        # dtec_post = _percentiles(ddtec_transformed, [15.,50.,85.])
        # # 3, N+Ns, Nf
        # phase_post = _percentiles(tf.atan2(Y_imag_samples, Y_real_samples))
        # Y_real_post = _percentiles(Y_real_samples, [50.])
        # Y_imag_post = _percentiles(Y_imag_samples, [50.])


        # 2, N+Ns
        dtec_post = _posterior(transformed_samples.dtec)
        Y_real_post = reduce_median(Y_real_samples, axis=0)
        Y_imag_post = reduce_median(Y_imag_samples, axis=0)
        effective_phase = tf.atan2(Y_imag_post, Y_real_post)

        Posterior = namedtuple('Solutions', ['tec', 'phase', 'weights_tec'])

        data_posterior = Posterior(
            # tec = tf.reshape(dtec_post[1,:N], X_dim),
            tec=tf.reshape(dtec_post[0][:N], X_dim),
            phase = tf.reshape(effective_phase[:N, :], tf.concat([X_dim, [-1]],axis=0)),
            weights_tec=tf.reshape(dtec_post[1][:N], X_dim)
        )

        screen_posterior = Posterior(
            tec=tf.reshape(dtec_post[0][N:], Xstar_dim),
            phase=tf.reshape(effective_phase[N:, :], tf.concat([Xstar_dim, [-1]], axis=0)),
            weights_tec=tf.reshape(dtec_post[1][N:], Xstar_dim)
        )

        Performance = namedtuple('Performance',['rhat', 'ess', 'log_accept_ratio','stepsizes'])

        FilterResult = namedtuple('FilterResult', ['performance','data_posterior', 'screen_posterior', 'next_hyperparams_warmstart', 'next_param_warmstart', 'next_stepsize_warmstart','solved_hyperparams', 'y_sigma_posterior', 'amp_posterior','index', 'next_index', 'mean_time','cont'])


        output = FilterResult(
            performance=Performance(rhat = rhat,
                                    ess = ess,
                                    log_accept_ratio=log_accept_ratio,
                                    stepsizes=stepsizes),
            data_posterior = data_posterior,
            screen_posterior = screen_posterior,
            next_hyperparams_warmstart=next_hyperparams_warmstart,
            next_param_warmstart = next_param_warmstart,
            next_stepsize_warmstart=next_stepsize_warmstart,
            solved_hyperparams=solved_hyperparams,
            y_sigma_posterior=transformed_samples.y_sigma,
            amp_posterior=transformed_samples.amp,
            index=index,
            next_index=next_index,
            mean_time=tf.reduce_mean(X[:,0]),
            cont=cont)
        return output

    def filter(self,
               num_samples=10,
               parallel_iterations=10,
               num_leapfrog_steps=2,
               target_rate=0.6,
               num_burnin_steps=100,
               kernel_params={},
               hyperparam_opt_params=dict(maxiter=100),
               num_adapation_steps=100,
               num_parallel_filters=1,
               ):

        def body(cont, param_warmstart, hyperparams_warmstart, stepsize_warmstart):
            results = self.filter_step(self.init_params,#param_warmstart,
                             self.init_hyperparams,#hyperparams_warmstart,
                             self.init_stepsizes,#stepsize_warmstart,
                             num_samples=num_samples,
                             parallel_iterations=int(parallel_iterations),
                             num_leapfrog_steps=num_leapfrog_steps,
                             target_rate=target_rate,
                             num_burnin_steps=num_burnin_steps,
                             kernel_params=kernel_params,
                             hyperparam_opt_params=hyperparam_opt_params,
                             num_adapation_steps=num_adapation_steps)

            plt_lock = Lock()
            store_lock = Lock()

            store_callbacks = [
                DatapackStoreCallback(self.datapack_feed.datapack,
                                  self.datapack_feed.posterior_solset, 'tec',
                                  (0, 2, 3, 1), # pol,time,dir,ant->pol,dir,ant,time
                                      lock=store_lock,
                                **self.datapack_feed.selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.posterior_solset, 'phase',
                                      (0, 2, 3, 4, 1),  # pol,time,dir,ant,freq->pol,dir,ant,freq,time
                                      lock=store_lock,
                                      **self.datapack_feed.selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.posterior_solset, 'weights_tec',
                                      (0, 2, 3, 1),  # pol,time,dir,ant->pol,dir,ant,time
                                      lock=store_lock,
                                      **self.datapack_feed.selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.screen_solset, 'tec',
                                      (0, 2, 3, 1),  # pol,time,dir,ant->pol,dir,ant,time
                                      lock=store_lock,
                                      **self.datapack_feed.selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.screen_solset, 'phase',
                                      (0, 2, 3, 4, 1),  # pol,time,dir,ant,freq->pol,dir,ant,freq,time
                                      lock=store_lock,
                                      **self.datapack_feed.selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.screen_solset, 'weights_tec',
                                      (0, 2, 3, 1),  # pol,time,dir,ant->pol,dir,ant,time
                                      lock=store_lock,
                                      **self.datapack_feed.selection),
                StoreHyperparameters(self.hyperparam_store)
            ]
            store_args = [(results.index, results.next_index,results.data_posterior.tec[None, ...]),
                            (results.index, results.next_index,results.data_posterior.phase[None, ...]),
                            (results.index, results.next_index,results.data_posterior.weights_tec[None, ...]),
                            (results.index, results.next_index,results.screen_posterior.tec[None, ...]),
                            (results.index, results.next_index,results.screen_posterior.phase[None, ...]),
                            (results.index, results.next_index,results.screen_posterior.weights_tec[None, ...]),
                            (results.mean_time, results.solved_hyperparams, results.y_sigma_posterior, results.amp_posterior)
                          ]

            store_op = callback_sequence(store_callbacks, store_args, async=True)

            performance_callbacks = [PlotRhat(plt_lock, self.plot_folder),
               PlotEss(plt_lock, self.plot_folder),
               PlotAcceptRatio(plt_lock, self.plot_folder),
               PlotStepsizes(plt_lock, self.plot_folder)]

            performance_args = [
                          (results.index, results.next_index, results.performance.rhat.dtec,
                           results.performance.rhat.amp, results.performance.rhat.y_sigma,
                           "dtec", 'amp', 'y_sigma'),
                          (results.index, results.next_index, results.performance.ess.dtec,
                           results.performance.ess.amp, results.performance.ess.y_sigma,
                           "dtec", 'amp', 'y_sigma'),
                          (results.index, results.next_index, results.performance.log_accept_ratio),
                          (results.index, results.next_index, results.performance.stepsizes.dtec,
                           results.performance.stepsizes.amp, results.performance.stepsizes.y_sigma,
                           "dtec", 'amp', 'y_sigma')
                          ]

            performance_op = callback_sequence(performance_callbacks, performance_args, async=True)

            plotres_callbacks = [
                    PlotResults(hyperparam_store=self.hyperparam_store,
                           datapack=self.datapack_feed.datapack,
                           solset=self.datapack_feed.solset,
                           posterior_name=self.datapack_feed.posterior_name,
                           lock=plt_lock,
                           plot_directory=self.plot_folder,
                           **self.datapack_feed.selection)]

            plotres_callbacks[0].controls = [store_op]

            plotres_args = [(results.index, results.next_index)]

            plotres_op = callback_sequence(plotres_callbacks, plotres_args, async=True)

            with tf.control_dependencies([store_op,performance_op, plotres_op]):
                return [tf.identity(results.cont), results.next_param_warmstart, results.next_hyperparams_warmstart, results.next_stepsize_warmstart]

        def cond(cont, param_warmstart, hyperparams_warmstart, stepsize_warmstart):
            return cont

        cont, params_out, hyperparams_out, stepsize_out = tf.while_loop(cond,
                      body,
                      [tf.constant(True),
                       self.init_params,
                       self.init_hyperparams,
                       self.init_stepsizes],
                      parallel_iterations=int(num_parallel_filters))

        return tf.group([cont, params_out, hyperparams_out, stepsize_out])


class FreeTransitionVariationalBayes(object):
    def __init__(self, datapack_feed: DatapackFeed, output_folder='./output_folder'):
        self.datapack_feed = datapack_feed
        self._output_folder = os.path.abspath(output_folder)
        logging.info("Using output directory: {}".format(output_folder))
        os.makedirs(self._output_folder,exist_ok=True)

    @property
    def output_folder(self):
        return self._output_folder

    @property
    def plot_folder(self):
        return os.path.join(self.output_folder, 'plot_folder')

    @property
    def hyperparam_store(self):
        return os.path.join(self.output_folder,'hyperparam_store.npz')

    @property
    def initializer(self):
        return self.init0

    def init_filter(self):

        self.full_block_size = (self.datapack_feed.coord_feed.N + self.datapack_feed.star_coord_feed.N) * self.datapack_feed.time_feed.slice_size

        white_dtec_posterior_temp = WhitenedVariationalPosterior(self.datapack_feed.coord_feed.N)
        self.init_params = white_dtec_posterior_temp.initial_variational_params()
        self.init_hyperparams = (tfp.distributions.softplus_inverse(tf.ones((1,6), float_type)),)

        self.datapack_feed_iterator = tf.data.Dataset.zip((self.datapack_feed.index_feed.feed, self.datapack_feed.feed)).make_initializable_iterator()

        self.init0 = self.datapack_feed_iterator.initializer

    def filter_step(self,
                    param_warmstart,
                    hyperparams_warmstart,
                    y_sigma,
                    num_mcmc_param_samples_learn,
                    num_mcmc_param_samples_infer,
                    solver_params=None,
                    parallel_iterations=10,
                    kernel_params=None):
        """

        :param param_warmstart:
        :param hyperparams_warmstart:
        :param solve_iters:
        :param num_mcmc_param_samples_learn:
        :param num_mcmc_hyperparam_samples_learn:
        :param num_mcmc_param_samples_infer:
        :param num_mcmc_hyperparam_samples_infer:
        :param learning_rate:
        :param mean_hyperparam_approx:
        :param parallel_iterations:
        :param obs_type:
        :param fed_kernel:
        :return:
        """


        def _maybe_cast(val, dtype):
            if val is not None:
                return tf.cast(val, dtype)
            return None


        sample_params = dict(
                             num_mcmc_param_samples_learn=_maybe_cast(num_mcmc_param_samples_learn, tf.int32),
                             num_mcmc_param_samples_infer=_maybe_cast(num_mcmc_param_samples_infer, tf.int32)
                             )

        (index, next_index), ((Yreal, Yimag), freqs, X, Xstar, X_dim, Xstar_dim, cont) = self.datapack_feed_iterator.get_next()

        variational_bayes = VariationalBayes(Yreal, Yimag, freqs, X, Xstar,y_sigma,
                                             dtec_samples=sample_params['num_mcmc_param_samples_learn'],
                                             kernel_params=kernel_params)

        t0 = timer()
        with tf.control_dependencies([t0]):
            loss, dtec_data_dist, dtec_screen_dist, (
                amp, lengthscales, a, b,
                timescale), next_param_warmstart, next_hyperparams_warmstart = variational_bayes.solve_variational_posterior(
                param_warmstart,
                hyperparams_warmstart,
                solver_params=solver_params,
                parallel_iterations=parallel_iterations)
            # num_hyperparams, 6
            solved_hyperparams = (amp, lengthscales, a, b, timescale, y_sigma)

        def _posterior(t):
            """
            Returns the percentiles down `axis` stacked on first axis.

            :param t: float_type, tf.Tensor, [S, f0, ..., fF]
                tensor to get percentiles for down first axis
            :param q: list of float_type
                Percentiles
            :return: float_type, tf.Tensor, [len(q), f0, ... ,fF]
            """
            mean = tf.reduce_mean(t, axis=0)
            var = tfp.stats.variance(t, sample_axis=0)
            return mean, var

        # S, H, N
        dtec_data = dtec_data_dist.sample(sample_params['num_mcmc_param_samples_infer'])
        dtec_data_post = _posterior(flatten_batch_dims(dtec_data, -1))
        #TODO: why isn't screen working
        # S, H, M
        dtec_screen = dtec_screen_dist.sample(sample_params['num_mcmc_param_samples_infer'])
        dtec_screen_post = _posterior(flatten_batch_dims(dtec_screen, -1))


        # S, H, N, Nf
        phase_data = dtec_data[..., None]*variational_bayes._invfreqs
        # S, H, M, Nf
        phase_screen = dtec_screen[..., None] * variational_bayes._invfreqs

        Yreal_data = tf.math.cos(phase_data)
        Yimag_data = tf.math.sin(phase_data)
        next_y_sigma = 0.5 * tf.reduce_mean(
            tf.math.abs(Yreal - tfp.stats.percentile(Yreal_data, 50., axis=[0, 1]))) + 0.5 * tf.reduce_mean(
            tf.math.abs(Yimag - tfp.stats.percentile(Yimag_data, 50., axis=[0, 1])))
        eff_phase_data = tf.math.atan2(tf.reduce_mean(Yimag_data, axis=[0, 1]), tf.reduce_mean(Yreal_data, axis=[0, 1]))

        Yreal_screen = tf.math.cos(phase_screen)
        Yimag_screen = tf.math.sin(phase_screen)
        eff_phase_screen = tf.math.atan2(tf.reduce_mean(Yimag_screen, axis=[0, 1]), tf.reduce_mean(Yreal_screen, axis=[0, 1]))

        Posterior = namedtuple('Solutions', ['tec', 'phase', 'weights_tec'])

        data_posterior = Posterior(
            tec=tf.reshape(dtec_data_post[0], X_dim),
            phase = tf.reshape(eff_phase_data, tf.concat([X_dim, [-1]],axis=0)),
            weights_tec=tf.reshape(dtec_data_post[1], X_dim)
        )

        screen_posterior = Posterior(
            tec=tf.reshape(dtec_screen_post[0], Xstar_dim),
            phase=tf.reshape(eff_phase_screen, tf.concat([Xstar_dim, [-1]], axis=0)),
            weights_tec=tf.reshape(dtec_screen_post[1], Xstar_dim)
        )

        Performance = namedtuple('Performance',['loss'])

        FilterResult = namedtuple('FilterResult', ['performance','data_posterior', 'screen_posterior', 'solved_hyperparams','next_param_warmstart','next_hyperparams_warmstart','next_y_sigma','index', 'next_index', 'mean_time','cont'])


        output = FilterResult(
            performance=Performance(loss = loss),
            data_posterior = data_posterior,
            screen_posterior = screen_posterior,
            solved_hyperparams=solved_hyperparams,
            next_param_warmstart=next_param_warmstart,
            next_hyperparams_warmstart=next_hyperparams_warmstart,
            next_y_sigma = next_y_sigma,
            index=index,
            next_index=next_index,
            mean_time=tf.reduce_mean(X[:,0]),
            cont=cont)
        return output

    def filter(self,
               parallel_iterations=10,
               kernel_params=None,
               num_parallel_filters=1,
               solver_params=None,
               num_mcmc_param_samples_learn = 1,
               num_mcmc_param_samples_infer=10,
               y_sigma=0.1
               ):

        def body(cont, param_warmstart, hyperparams_warmstart, y_sigma):
            results = self.filter_step(param_warmstart, hyperparams_warmstart,
                                       y_sigma,
                                       parallel_iterations=int(parallel_iterations),
                                       num_mcmc_param_samples_learn=num_mcmc_param_samples_learn,
                                       num_mcmc_param_samples_infer=num_mcmc_param_samples_infer,
                                       solver_params=solver_params,
                                       kernel_params = kernel_params
                                       )

            plt_lock = Lock()
            store_lock = Lock()

            store_callbacks = [
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.posterior_solset, 'tec',
                                      (0, 2, 3, 1),  # pol,time,dir,ant->pol,dir,ant,time
                                      lock=store_lock,
                                      index_map=self.datapack_feed.index_map,
                                      **self.datapack_feed.selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.posterior_solset, 'phase',
                                      (0, 2, 3, 4, 1),  # pol,time,dir,ant,freq->pol,dir,ant,freq,time
                                      lock=store_lock,
                                      index_map=self.datapack_feed.index_map,
                                      **self.datapack_feed.selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.posterior_solset, 'weights_tec',
                                      (0, 2, 3, 1),  # pol,time,dir,ant->pol,dir,ant,time
                                      lock=store_lock,
                                      index_map=self.datapack_feed.index_map,
                                      **self.datapack_feed.selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.screen_solset, 'tec',
                                      (0, 2, 3, 1),  # pol,time,dir,ant->pol,dir,ant,time
                                      lock=store_lock,
                                      index_map=self.datapack_feed.index_map,
                                      **self.datapack_feed.screen_selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.screen_solset, 'phase',
                                      (0, 2, 3, 4, 1),  # pol,time,dir,ant,freq->pol,dir,ant,freq,time
                                      lock=store_lock,
                                      index_map=self.datapack_feed.index_map,
                                      **self.datapack_feed.screen_selection),
                DatapackStoreCallback(self.datapack_feed.datapack,
                                      self.datapack_feed.screen_solset, 'weights_tec',
                                      (0, 2, 3, 1),  # pol,time,dir,ant->pol,dir,ant,time
                                      lock=store_lock,
                                      index_map=self.datapack_feed.index_map,
                                      **self.datapack_feed.screen_selection),
                StoreHyperparametersV2(self.hyperparam_store)
            ]

            store_args = [(results.index, results.next_index,results.data_posterior.tec[None, ...]),
                            (results.index, results.next_index,results.data_posterior.phase[None, ...]),
                            (results.index, results.next_index,results.data_posterior.weights_tec[None, ...]),
                            (results.index, results.next_index,results.screen_posterior.tec[None, ...]),
                            (results.index, results.next_index,results.screen_posterior.phase[None, ...]),
                            (results.index, results.next_index,results.screen_posterior.weights_tec[None, ...]),
                            (results.mean_time,) + results.solved_hyperparams + (results.next_y_sigma,)
                          ]

            store_op = callback_sequence(store_callbacks, store_args, async=True)

            performance_callbacks = [PlotELBO(plt_lock, self.plot_folder, index_map=self.datapack_feed.index_map,)]

            performance_args = [
                          (results.index, results.next_index, results.performance.loss)
                          ]

            performance_op = callback_sequence(performance_callbacks, performance_args, async=True)

            plotres_callbacks = [
                PlotResultsV2(hyperparam_store=self.hyperparam_store,
                              datapack=self.datapack_feed.datapack,
                              solset=self.datapack_feed.solset,
                              posterior_name=self.datapack_feed.posterior_name,
                              index_map=self.datapack_feed.index_map,
                              lock=plt_lock,
                              plot_directory=self.plot_folder,
                              **self.datapack_feed.selection)]

            plotres_callbacks[0].controls = [store_op]

            plotres_args = [(results.index, results.next_index)]

            plotres_op = callback_sequence(plotres_callbacks, plotres_args, async=True)

            next_param_warmstart, next_hyperparams_warmstart = results.next_param_warmstart, results.next_hyperparams_warmstart
            [n.set_shape(p.shape) for n,p in zip(next_param_warmstart, param_warmstart)]
            [n.set_shape(p.shape) for n, p in zip(hyperparams_warmstart, hyperparams_warmstart)]

            with tf.control_dependencies([store_op,performance_op, plotres_op]):
                return [tf.identity(results.cont), next_param_warmstart, next_hyperparams_warmstart, results.next_y_sigma]

        def cond(cont, param_warmstart, hyperparams_warmstart, y_sigma):
            return cont

        self._init_y_sigma = tf.convert_to_tensor(y_sigma, float_type, name='y_sigma_init')

        cont, params_out, hyperparams_out, _ = tf.while_loop(cond,
                      body,
                      [tf.constant(True),
                       self.init_params,
                       self.init_hyperparams,
                       self._init_y_sigma
                      ],
                      parallel_iterations=int(num_parallel_filters))

        return tf.group([cont, params_out, hyperparams_out])
