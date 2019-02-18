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
                    init_kern_params=None, which_kernel=0, kernel_params={}, saem_batchsize=500):
        """
        Run a Bayes filter over the coordinate set.

        :param num_samples: int
        :param num_chains: int
        :param parallel_iteration: int
        :param num_leapfrog_steps:
        :param target_rate:
        :param num_burnin_steps: int
        :return: list of UpdateResult
            The results of each filter step
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
        # Nf
        invfreqs = -8.448e9 * tf.reciprocal(self.freqs)
        phase = tf.atan2(Y_imag, Y_real)
        # N
        dtec_init_data = tf.reduce_mean(phase / invfreqs, axis=-1)
        dtec_screen_init = tf.zeros(tf.shape(Xstar)[0:1],float_type)
        dtec_init = tf.concat([dtec_init_data,dtec_screen_init],axis=0)
        if init_kern_params is None:
            init_kern_params = {}
        target = DTECToGainsSAEM(X, Xstar, Y_real, Y_imag, self.freqs,
                                 fed_kernel='RBF', obs_type='DDTEC', full_posterior=True,
                                 which_kernel=which_kernel, kernel_params=kernel_params,
                                 **init_kern_params)
        variables = target.variables
        inits = tf.group([step_size.initializer for step_size in step_sizes]
                         + [joint_iterator.initializer] + [target.variables.initializer])
        dtec_init = target.get_initial_point(dtec_init)
        q0_init = [tf.tile(tf.reshape(dtec_init, (-1,))[None, :], (num_chains, 1))]
        # q0_init = [tf.Variable(q0_init[0], dtype=float_type, name='dtec_init')]

        init_simplex = target.variables + tf.random_normal((7,6),dtype=float_type)


        ###
        # Hyper parameter solution

        def saem_step(variables):
            var_copy = tf.identity(variables)

            def log_prob(variables):
                X_learn, Y_real_learn, Y_imag_learn, dtec_init_learn = random_sample(
                    [X, Y_real, Y_imag, dtec_init_data],
                    saem_batchsize)

                target_saem = DTECToGainsSAEM(X_learn, None, Y_real_learn, Y_imag_learn, self.freqs,
                                              fed_kernel='RBF', obs_type='DDTEC',
                                              variables=variables,which_kernel=which_kernel, kernel_params=kernel_params)

                ###
                hmc = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_saem.logp,
                    num_leapfrog_steps=num_leapfrog_steps,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
                    step_size=step_sizes,
                    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=None,
                                                                                     decrement_multiplier=0.1,
                                                                                     increment_multiplier=0.1,
                                                                                     target_rate=target_rate),
                    state_gradients_are_stopped=True)

                # Run the chain (with burn-in).
                unconstrained_samples, kernel_results = tfp.mcmc.sample_chain(
                    num_results=num_saem_samples,
                    num_burnin_steps=num_burnin_steps,
                    current_state=[tf.tile(tf.reshape(dtec_init_learn, (-1,))[None, :], (num_chains, 1))],
                    kernel=hmc,
                    parallel_iterations=parallel_iterations)




                # unconstrained_samples = tf.stop_gradient(flatten_batch_dims(unconstrained_samples[0]))
                #
                # constrained_samples = target_saem.transform_samples(unconstrained_samples)
                #
                # mean_dtec = tf.reduce_mean(constrained_samples, axis=0)
                # var_dtec = tf.reduce_mean(tf.square(constrained_samples), axis=0) - tf.square(mean_dtec)
                # mean_variance = tf.reduce_mean(var_dtec)
                # ###

                posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                                    name='marginal_log_likelihood')

                with tf.control_dependencies([tf.print(posterior_log_prob, target_saem.constrained_states(variables))]):
                    ess = tfp.mcmc.effective_sample_size(unconstrained_samples)
                    avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                                                          name='avg_acc_ratio')
                    posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                                        name='marginal_log_likelihood')
                    return posterior_log_prob
                    # return target_saem.logp_dtec(mean_dtec, mean_variance) \
                    #        + tf.reduce_mean(target_saem.logp_gains(constrained_samples)) #\
                    #        # + target_saem.logp_params(variables)

            def saem_objective(variables):
                objective = -tf.map_fn(log_prob,variables)
                return objective
                # return (objective, tf.gradients(objective, variables)[0])

            # obj = saem_objective(var_copy)
            # g = tf.gradients(obj, var_copy)[0]
            # H = tf.stack()
            # H = tf.hessians(obj, var_copy)[0]
            # new_m = var_copy + tf.matrix_solve_ls(H, g[:,None],fast=False)[:,0]

            saem_mstep = tfp.optimizer.differential_evolution_minimize(saem_objective,
                                                      initial_position=var_copy,
                                                      population_size=10,
                                                      max_iterations=saem_maxsteps)

            # saem_mstep = tfp.optimizer.nelder_mead_minimize(saem_objective,
            #                                                 # initial_vertex=var_copy,
            #                                           initial_simplex=init_simplex,
            #                                           max_iterations=saem_maxsteps,
            #                                                 # step_sizes=0.5,
            #                                           parallel_iterations=10)

            # saem_mstep = tfp.optimizer.lbfgs_minimize(saem_objective,
            #                                          var_copy,
            #                                          tolerance=1e-8,
            #                                          x_tolerance=0,
            #                                          f_relative_tolerance=0,
            #                                          max_iterations=saem_maxsteps,
            #                                          parallel_iterations=1)

            # with tf.control_dependencies([tf.print("SAEM:", saem_mstep)]):
            update_variables = tf.assign(variables, saem_mstep.position)#saem_mstep.position
            return update_variables

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

        update_variables = saem_step(variables)
        with tf.control_dependencies(
                [tf.cond(tf.greater(saem_maxsteps, 0),lambda: update_variables, lambda: variables)]):
            with tf.control_dependencies([tf.print("m:",target.constrained_states(variables))]):
                # q0_init = [tf.reduce_mean(unconstrained_samples[0],axis=0)]

                hmc = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target.logp,
                    num_leapfrog_steps=num_leapfrog_steps,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
                    step_size=step_sizes,
                    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=None,
                                                                                     decrement_multiplier=0.1,
                                                                                     increment_multiplier=0.1,
                                                                                     target_rate=target_rate),
                    state_gradients_are_stopped=True)

                # Run the chain (with burn-in).
                unconstrained_samples, kernel_results = tfp.mcmc.sample_chain(
                    num_results=num_samples,
                    num_burnin_steps=num_burnin_steps,
                    current_state=q0_init,
                    kernel=hmc,
                    parallel_iterations=parallel_iterations)

                ess = tfp.mcmc.effective_sample_size(unconstrained_samples)

                flat_samples = flatten_batch_dims(unconstrained_samples[0])

                test_logp = tf.reduce_mean(target.logp(flat_samples[:,N:]))/tf.cast(Ns,float_type)
                post_logp = tf.reduce_mean(target.logp(flat_samples[:, :N])) / tf.cast(N, float_type)

                ddtec_transformed = target.transform_samples(flat_samples)
                Y_real_samples, Y_imag_samples = target.forward_equation(ddtec_transformed)

                avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                                                  name='avg_acc_ratio')
                posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                               name='marginal_log_likelihood')/tf.cast(N+Ns,float_type)

        TAResult = namedtuple('TAResult', ['parameters', 'dtec', 'Y_real', 'Y_imag',
                                           'dtec_star', 'Y_real_star', 'Y_imag_star', 'acc_ratio',
                                           'post_logp', 'test_logp','cont', 'step_sizes', 'ess', 'extra'])

        ExtraResults = namedtuple('ExtraResults',['Y_real_data', 'Y_imag_data', 'X', 'Xstar', 'X_dim', 'Xstar_dim', 'freqs'])

        dtec_post = _percentiles(ddtec_transformed)
        dtec_X = tf.reshape(dtec_post[:,:N], tf.concat([[3], X_dim],axis=0))
        dtec_Xstar = tf.reshape(dtec_post[:,N:], tf.concat([[3], Xstar_dim],axis=0))

        Y_real_post = _percentiles(Y_real_samples)
        Y_real_X = tf.reshape(Y_real_post[:,:N,:], tf.concat([[3], X_dim, [-1]],axis=0))
        Y_real_Xstar = tf.reshape(Y_real_post[:, N:,:], tf.concat([[3], Xstar_dim, [-1]], axis=0))

        Y_imag_post = _percentiles(Y_imag_samples)
        Y_imag_X = tf.reshape(Y_imag_post[:, :N,:], tf.concat([[3], X_dim, [-1]], axis=0))
        Y_imag_Xstar = tf.reshape(Y_imag_post[:, N:,:], tf.concat([[3], Xstar_dim, [-1]], axis=0))

        Y_real = tf.reshape(Y_real, tf.concat([X_dim, [-1]],axis=0))
        Y_imag = tf.reshape(Y_imag, tf.concat([X_dim, [-1]], axis=0))

        output = TAResult(
            parameters = target.constrained_states(variables),
            dtec = dtec_X,
            Y_real = Y_real_X,
            Y_imag = Y_imag_X,
            dtec_star = dtec_Xstar,
            Y_real_star = Y_real_Xstar,
            Y_imag_star = Y_imag_Xstar,
            acc_ratio = avg_acceptance_ratio,
            post_logp = posterior_log_prob,
            test_logp = posterior_log_prob,
            cont = cont,
            step_sizes = kernel_results.extra.step_size_assign[0],
            ess = ess[0],
            extra = ExtraResults(Y_real, Y_imag, X, Xstar, X_dim, Xstar_dim, self.freqs))
        return output, inits
