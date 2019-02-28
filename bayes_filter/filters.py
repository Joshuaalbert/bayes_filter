import tensorflow as tf
import tensorflow_probability as tfp
from .data_feed import init_feed, DataFeed, CoordinateFeed, ContinueFeed, CoordinateDimFeed
from .settings import float_type
from collections import namedtuple
from .logprobabilities import DTECToGains, DTECToGainsSAEM
from .misc import flatten_batch_dims, timer, random_sample

class FreeTransitionSAEM(object):
    def __init__(self, freqs, data_feed: DataFeed, coord_feed: CoordinateFeed, star_coord_feed: CoordinateFeed):
        self.coord_feed = coord_feed
        self.coord_dim_feed = CoordinateDimFeed(self.coord_feed)
        self.star_coord_feed = star_coord_feed
        self.star_coord_dim_feed = CoordinateDimFeed(self.star_coord_feed)
        self.data_feed = data_feed
        self.continue_feed = ContinueFeed(self.coord_feed.time_feed)
        self.freqs = tf.convert_to_tensor(freqs,float_type)

    def filter_step(self, num_samples=10, num_chains=1, parallel_iterations=10, num_leapfrog_steps=2, target_rate=0.6,
                    num_burnin_steps=0, num_saem_samples = 10, saem_maxsteps=5, initial_stepsize=5e-3,
                    init_kern_params=None, which_kernel=0, kernel_params={}, saem_batchsize=500, slice_size=None,
                    saem_population=20):
        """

        :param num_samples:
        :param num_chains:
        :param parallel_iterations:
        :param num_leapfrog_steps:
        :param target_rate:
        :param num_burnin_steps:
        :param num_saem_samples:
        :param saem_maxsteps:
        :param initial_stepsize:
        :param init_kern_params:
        :param which_kernel:
        :param kernel_params:
        :param saem_batchsize:
        :return:
        """
        joint_dataset = tf.data.Dataset.zip((self.data_feed.feed,
                                             self.coord_feed.feed,
                                             self.star_coord_feed.feed,
                                             self.continue_feed.feed,
                                             self.coord_dim_feed.feed,
                                             self.star_coord_dim_feed.feed))

        joint_iterator = joint_dataset.make_initializable_iterator()

        step_sizes = [
            tf.get_variable(
                name='step_size_dtec',
                initializer=lambda: tf.constant(initial_stepsize, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False)
        ]

        (Y_real, Y_imag), X, Xstar, cont, X_dim, Xstar_dim = joint_iterator.get_next()
        N = tf.shape(X)[0]
        Ns = tf.shape(Xstar)[0]

        dtec_init_data = tf.zeros(tf.shape(X)[0:1], float_type)
        dtec_screen_init = tf.zeros(tf.shape(Xstar)[0:1], float_type)
        dtec_init = tf.concat([dtec_init_data, dtec_screen_init], axis=0)

        if init_kern_params is None:
            init_kern_params = {}
        proxy_target = DTECToGainsSAEM(X, Xstar, Y_real, Y_imag, self.freqs,
                                 fed_kernel='RBF', obs_type='DDTEC', full_posterior=True,
                                 which_kernel=which_kernel, kernel_params=kernel_params,Nh=slice_size,
                                 **init_kern_params)

        L_variable = tf.get_variable('L_variable',initializer=proxy_target.L)#tf.zeros((slice_size, slice_size),float_type))


        variables = proxy_target.variables

        init1 = joint_iterator.initializer
        init2 = L_variable.initializer
        init3 = tf.group([step_size.initializer for step_size in step_sizes] + [joint_iterator.initializer] + [variables.initializer])
        # with tf.control_dependencies([tf.group([step_size.initializer for step_size in step_sizes] + [joint_iterator.initializer] + [proxy_target.variables.initializer])]):
        #     with tf.control_dependencies([L_variable.initializer]):
        #         inits = tf.tuple([joint_iterator.initializer])


        # # Nf
        # invfreqs = -8.448e9 * tf.reciprocal(self.freqs)
        # phase = tf.atan2(Y_imag, Y_real)
        # # N
        # dtec_init_data = tf.reduce_mean(phase / invfreqs, axis=-1)
        # dtec_screen_init = tf.zeros(tf.shape(Xstar)[0:1],float_type)
        # dtec_init = tf.concat([dtec_init_data,dtec_screen_init],axis=0)
        # dtec_init = target.get_initial_point(dtec_init)
        # q0_init = [tf.tile(tf.reshape(dtec_init, (-1,))[None, :], (num_chains, 1))]

        def _sample(X, Y_real, Y_imag, dtec_init, Xstar=None, variables=None, L=None, num_samples=100, full_posterior=True):
            # uses variables if not None, else initial params
            target = DTECToGainsSAEM(X, Xstar, Y_real, Y_imag, self.freqs,
                                          fed_kernel='RBF', obs_type='DDTEC',
                                          variables=variables, which_kernel=which_kernel,
                                          kernel_params=kernel_params,L=L,full_posterior=full_posterior,
                                          **init_kern_params)


            # def trace(states, previous_kernel_results):
            #     """Trace the transformed dtec, stepsize, log_acceptance, log_prob"""
            #     dtec_constrained = target.transform_samples(states[0])
            #     return dtec_constrained, \
            #            previous_kernel_results.step_size, \
            #            previous_kernel_results.log_acceptance_correction, \
            #            previous_kernel_results.target_log_prob

            ###
            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target.logp,
                num_leapfrog_steps=num_leapfrog_steps,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
                step_size=step_sizes,
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=None,
                                                                                 decrement_multiplier=0.1,
                                                                                 increment_multiplier=0.1,
                                                                                 target_rate=target_rate),
                state_gradients_are_stopped=True)

            # Run the chain (with burn-in maybe).
            samples = tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin_steps,
                trace_fn = None,
                return_final_kernel_results=False,
                current_state=[dtec_init],
                kernel=hmc,
                parallel_iterations=parallel_iterations)

            # last state as initial point

            rhat = tfp.mcmc.potential_scale_reduction(samples)

            nonconverged_rhat = [tfp.stats.percentile(rh, 90.) for rh in rhat]
            with tf.control_dependencies([tf.print(nonconverged_rhat)]):
                do_again = tf.reduce_any([tf.greater(r, 1.2) for r in nonconverged_rhat])

                def resample(samples=samples):
                    state_init = [s[-1, ...] for s in samples]
                    more_samples = tfp.mcmc.sample_chain(
                        num_results=num_samples,
                        num_burnin_steps=0,
                        trace_fn=None,
                        return_final_kernel_results=False,
                        current_state=state_init,
                        kernel=hmc,
                        parallel_iterations=parallel_iterations)
                    return [tf.concat([s, ms],axis=0) for (s,ms) in zip(samples, more_samples)]

                samples = tf.cond(do_again, resample, lambda: samples, strict=True)

            return samples, target

        ###
        # Hyper parameter solution

        # def saem_step(variables):
        #     """
        #     Proceeds by doing a maximum likelihood estimate of the ddtec,
        #     then do hyperparameter inference analytically on it, with differential evolution.
        #     Gradient based would also work.
        #     :param variables:
        #     :return:
        #     """
        #     var_copy = tf.identity(variables)
        #
        #     def log_prob(variables):
        #         X_learn, Y_real_learn, Y_imag_learn, dtec_init_learn = random_sample(
        #             [X, Y_real, Y_imag, dtec_init_data],
        #             saem_batchsize)
        #
        #         dtec_init_learn = tf.random_normal(shape=tf.concat([[num_chains], tf.shape(dtec_init_learn)], axis=0),
        #                                            stddev=tf.constant(0.1, dtype=float_type),
        #                                            dtype=float_type) + dtec_init_learn
        #
        #         unconstrained_samples, target_saem = _sample(X_learn, Y_real_learn, Y_imag_learn, dtec_init_learn, Xstar=None,
        #                                         variables=variables,L=None, num_samples=num_saem_samples)
        #
        #         rhat_dtec = tfp.mcmc.potential_scale_reduction(unconstrained_samples[0])
        #         unconstrained_dtec = tf.stop_gradient(flatten_batch_dims(unconstrained_samples[0]))
        #
        #         constrained_dtec = target_saem.transform_samples(unconstrained_dtec)
        #
        #         mean_dtec = tf.reduce_mean(constrained_dtec, axis=0)
        #         var_dtec = tf.reduce_mean(tf.square(constrained_dtec), axis=0) - tf.square(mean_dtec)
        #         mean_variance = tf.reduce_mean(var_dtec)
        #         ###
        #
        #         posterior_log_prob = target_saem.logp_dtec(mean_dtec, mean_variance) \
        #                              + tf.reduce_mean(target_saem.logp_gains(constrained_dtec)) #\
        #                              # + target_saem.logp_params(variables)
        #         with tf.control_dependencies([tf.print("Rhat:",tfp.stats.percentile(rhat_dtec,[10., 50., 90.]),
        #                                                'logProb:',posterior_log_prob,
        #                                                'm:',target_saem.constrained_states(variables))]):
        #             return tf.identity(posterior_log_prob)
        #
        #     def saem_objective(variables):
        #         objective = -tf.map_fn(log_prob,variables)
        #         return objective
        #
        #
        #
        #     saem_mstep = tfp.optimizer.differential_evolution_minimize(saem_objective,
        #                                               initial_position=var_copy,
        #                                               population_size=10,
        #                                               max_iterations=saem_maxsteps)
        #
        #     return tf.assign(variables, saem_mstep.position)

        def saem_step(variables, L_start=None):
            """
            Proceeds by doing a maximum likelihood estimate of the ddtec,
            then do hyperparameter inference analytically on it, with differential evolution.
            Gradient based would also work.
            :param variables:
            :return:
            """
            var_copy = tf.identity(variables)

            dtec_init_saem = tf.random_normal(shape=tf.concat([[num_chains], tf.shape(dtec_init_data)], axis=0),
                                              stddev=tf.constant(0.1, dtype=float_type),
                                              dtype=float_type) + dtec_init_data

            unconstrained_samples, target_saem = _sample(X, Y_real, Y_imag, dtec_init_saem,
                                                         Xstar=None,
                                                         variables=var_copy,
                                                         L=L_start[:N,:N],
                                                         num_samples=num_saem_samples,
                                                         full_posterior=False)

            rhat_dtec = tfp.mcmc.potential_scale_reduction(unconstrained_samples[0])
            #S*num_chains, M
            unconstrained_dtec = flatten_batch_dims(unconstrained_samples[0])
            # S*num_chains, M
            constrained_dtec = target_saem.transform_samples(unconstrained_dtec)
            #M
            mean_dtec = tf.reduce_mean(constrained_dtec, axis=0)
            #M
            var_dtec = tf.reduce_mean(tf.square(constrained_dtec), axis=0) - tf.square(mean_dtec)

            def log_prob(variables, mean_dtec=mean_dtec, var_dtec=var_dtec):
                """
                Get the log-probability of variables.

                :param variables: float_type, tf.Tensor, [batch, num_params]
                    Batched variables.
                :param mean_dtec: float_type, tf.Tensor, [M]
                :param var_dtec: float_type, tf.Tensor, [M]
                :return: float_type, tf.Tensor, [batch]
                """
                X_learn, Y_real_learn, Y_imag_learn,mean_dtec, var_dtec = random_sample(
                    [X, Y_real, Y_imag, mean_dtec, var_dtec],
                    saem_batchsize)

                candidate_target = DTECToGainsSAEM(X_learn, None,
                                                   Y_real_learn, Y_imag_learn,self.freqs,
                                                 variables=variables, L=None,
                                                 full_posterior=False)

                ###
                posterior_log_prob = candidate_target.logp_dtec(mean_dtec, var_dtec)

                with tf.control_dependencies([tf.print('logProb:',posterior_log_prob,
                                                       'm:',candidate_target.constrained_states(variables))]):
                    return tf.identity(posterior_log_prob)

            def saem_objective(variables):
                objective = -log_prob(variables) # -tf.map_fn(log_prob, variables)
                return objective

            with tf.control_dependencies([tf.print("Rhat:", tfp.stats.percentile(rhat_dtec, [10., 50., 90.]))]):
                saem_mstep = tfp.optimizer.differential_evolution_minimize(saem_objective,
                                                          initial_position=var_copy,
                                                          population_size=saem_population,
                                                          max_iterations=saem_maxsteps)

            return tf.assign(variables, saem_mstep.position)

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
        with tf.control_dependencies(
                [t0, tf.cond(tf.greater(saem_maxsteps, 0),lambda: saem_step(variables, L_start=L_variable), lambda: variables)]):
            with tf.control_dependencies([tf.print("Sampling with m:",proxy_target.constrained_states(variables))]):

                dtec_init_main = tf.random_normal(shape=tf.concat([[num_chains], tf.shape(dtec_init)], axis=0),
                                                   stddev=tf.constant(0.1, dtype=float_type),
                                                   dtype=float_type) + dtec_init


                unconstrained_samples, target = _sample(X, Y_real, Y_imag, dtec_init_main,
                                                             Xstar=Xstar,
                                                             variables=variables, L=L_variable,
                                                             num_samples=num_samples,
                                                            full_posterior=True)



                ess = tfp.mcmc.effective_sample_size(unconstrained_samples)
                rhat = tfp.mcmc.potential_scale_reduction(unconstrained_samples)

                flat_samples = flatten_batch_dims(unconstrained_samples[0])

                # test_logp = tf.reduce_mean(target.logp(flat_samples[:,N:]))/tf.cast(Ns,float_type)
                post_logp = tf.reduce_mean(target.logp(flat_samples)) / tf.cast(N + Ns, float_type)

                ddtec_transformed = target.transform_samples(flat_samples)
                Y_real_samples, Y_imag_samples = target.forward_equation(ddtec_transformed)

                # avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                #                                   name='avg_acc_ratio')
                # posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                #                                name='marginal_log_likelihood')/tf.cast(N+Ns,float_type)

        TAResult = namedtuple('TAResult', ['parameters', 'dtec', 'Y_real', 'Y_imag','post_logp',
                                           'dtec_star', 'Y_real_star', 'Y_imag_star',
                                           'cont', 'ess', 'rhat','extra','sample_time', 'phase', 'phase_star'])

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


        with tf.control_dependencies([unconstrained_samples[0]]):
            t1 = timer()

        output = TAResult(
            parameters = target.constrained_states(variables),
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
            phase_star=phase_Xstar)
        return output, (init1, init2, init3)
