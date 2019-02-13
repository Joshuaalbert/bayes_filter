import tensorflow as tf
import tensorflow_probability as tfp
from .data_feed import init_feed, DataFeed, CoordinateFeed, ContinueFeed, CoordinateDimFeed
from .settings import float_type
from collections import namedtuple
from .logprobabilities import DTECToGains, DTECToGainsSAEM
from .misc import flatten_batch_dims, timer


# UpdateResult = namedtuple('UpdateResult',['x_samples','z_samples','log_prob', 'acceptance','step_size'])

class FreeTransition(object):
    def __init__(self, freqs, data_feed: DataFeed, coord_feed: CoordinateFeed, star_coord_feed: CoordinateFeed):
        self.coord_feed = coord_feed
        self.star_coord_feed = star_coord_feed
        self.data_feed = data_feed
        self.continue_feed = ContinueFeed(self.coord_feed.time_feed)
        self.freqs = tf.convert_to_tensor(freqs,float_type)

    def filter(self, num_samples=10, num_chains=1, parallel_iterations=10, num_leapfrog_steps=2, target_rate=0.5, num_burnin_steps=0):
        """
        Run a Bayes filter over the coordinate set.

        :param q0: list of float_type, Tensor, [num_chains, M, Np]
            Initial state point
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
                                             self.continue_feed.feed))
        joint_iterator = joint_dataset.make_initializable_iterator()

        # data_iterator = self.data_feed.feed.make_initializable_iterator()
        # coord_iterator = self.coord_feed.feed.make_initializable_iterator()
        # star_coord_iterator = self.star_coord_feed.feed.make_initializable_iterator()
        # continue_feed_iterator = self.continue_feed.feed.make_initializable_iterator()

        #y_sigma, variance, lengthscales, a, b, timescale, dtec
        step_sizes = [
            tf.get_variable(
                name='step_size_y_sigma',
                initializer=lambda: tf.constant(3.3e-5, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False),
            tf.get_variable(
                name='step_size_variance',
                initializer=lambda: tf.constant(3.3e-5, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False),
            tf.get_variable(
                name='step_size_lengthscales',
                initializer=lambda: tf.constant(3.3e-5, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False),
            tf.get_variable(
                name='step_size_a',
                initializer=lambda: tf.constant(3.3e-5, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False),
            tf.get_variable(
                name='step_size_b',
                initializer=lambda: tf.constant(3.3e-5, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False),
            tf.get_variable(
                name='step_size_timescale',
                initializer=lambda: tf.constant(3.3e-5, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False),
            tf.get_variable(
                name='step_size_dtec',
                initializer=lambda: tf.constant(3.3e-5, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False)
        ]

        inits = tf.group([step_size.initializer for step_size in step_sizes]
                         + [joint_iterator.initializer])

        TAResult = namedtuple('TAResult',['y_sigma', 'variance', 'lengthscales', 'a', 'b',
                                      'timescale', 'dtec', 'Y_real', 'Y_imag', 'acc_ratio', 'post_logp'])

        # y_sigma, variance, lengthscales, a, b, timescale, dtec, Y_real, Y_imag
        tas = TAResult(tf.TensorArray(float_type, self.continue_feed.num_blocks, name='y_sigma'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='variance'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='lengthscales'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='a'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='b'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='timescale'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='dtec'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='y_real'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='y_imag'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='acc_ratio'),
               tf.TensorArray(float_type, self.continue_feed.num_blocks, name='post_logp'))

        def cond(unused_i, cont, unused_tas):
            return cont

        def body(i, unused_cont, tas):

            (Y_real, Y_imag), X, Xstar, cont = joint_iterator.get_next()

            # Nf
            invfreqs = -8.448e9 * tf.reciprocal(self.freqs)
            phase = tf.atan2(Y_real, Y_imag)
            # N
            dtec_init = tf.reduce_mean(phase / invfreqs, axis=-1)
            dtec_screen_init = tf.zeros(tf.shape(Xstar)[0:1],float_type)
            # with tf.control_dependencies([tf.print(tf.shape(X), tf.shape(Y_real))]):
            dtec_init = tf.concat([dtec_init,dtec_screen_init],axis=0)

            q0_init = [tf.constant(0.2, float_type), tf.constant(0.07, float_type),#tf.reduce_mean(tf.square(dtec_init / 50.)),
                       tf.constant(15.,float_type), tf.constant(250.,float_type), tf.constant(50.,float_type),
                       tf.constant(30., float_type), dtec_init]

            target = DTECToGains(X, Xstar, Y_real, Y_imag, self.freqs,
                                 y_sigma=q0_init[0], variance=q0_init[1], lengthscales=q0_init[2],
                                 a=q0_init[3], b=q0_init[4], timescale=q0_init[5],
                                 fed_kernel='RBF', obs_type='DDTEC', num_chains = num_chains, ss=step_sizes)

            q0_init = target.get_initial_point(*q0_init)
            # with tf.control_dependencies([tf.print('q0_init',*[(tf.shape(s), s) for s in q0_init])]):
            q0_init = [tf.tile(tf.reshape(q, (-1,))[None, :], (num_chains, 1)) for q in q0_init]

            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target.logp,
                num_leapfrog_steps=num_leapfrog_steps,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
                step_size=step_sizes,
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=None,
                                                                                 decrement_multiplier=0.1,
                                                                                 increment_multiplier=0.1,
                                                                                 target_rate=target_rate),
                state_gradients_are_stopped=True)
            #                         step_size_update_fn=lambda v, _: v)

            # Run the chain (with burn-in).
            z_samples, kernel_results = tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin_steps,
                current_state=q0_init,
                kernel=hmc,
                parallel_iterations=parallel_iterations)

            state_transformed, ddtec_transformed = target.transform_samples(*[flatten_batch_dims(s, 2) for s in z_samples])

            Y_real_samples, Y_imag_samples = target.forward_equation(ddtec_transformed)

            avg_acceptance_ratio = tf.reshape(tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                                                  name='avg_acc_ratio'), (1,))
            posterior_log_prob = tf.reshape(tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                               name='marginal_log_likelihood'), (1,))

            # res = UpdateResult(x_samples, z_samples, posterior_log_prob, avg_acceptance_ratio,
            #                    kernel_results.extra.step_size_assign)

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


            output_tas = tas._replace(y_sigma = tas.y_sigma.write(i, _percentiles(state_transformed.y_sigma)),
                                      variance = tas.variance.write(i, _percentiles(state_transformed.variance)),
                                      lengthscales = tas.lengthscales.write(i, _percentiles(state_transformed.lengthscales)),
                                      a = tas.a.write(i, _percentiles(state_transformed.a)),
                                      b = tas.b.write(i, _percentiles(state_transformed.b)),
                                      timescale = tas.timescale.write(i, _percentiles(state_transformed.timescale)),
                                      dtec = tas.dtec.write(i, _percentiles(ddtec_transformed)),
                                      Y_real = tas.Y_real.write(i, _percentiles(Y_real_samples)),
                                      Y_imag = tas.Y_imag.write(i, _percentiles(Y_imag_samples)),
                                      acc_ratio = tas.acc_ratio.write(i, avg_acceptance_ratio),
                                      post_logp = tas.post_logp.write(i, posterior_log_prob)
                                      )


            return [i + 1, cont, output_tas]

        # t0 = timer()
        with tf.control_dependencies([inits]):
            [_, done, output_tas] = tf.while_loop(cond,
                                               body,
                                               [tf.constant(0), tf.constant(True), tas])

        output = TAResult(*[ta.concat() for ta in output_tas])

        # with tf.control_dependencies([done, tf.print("Time taken:", timer() - t0)]):
        return output


class FreeTransitionSAEM(object):
    def __init__(self, freqs, data_feed: DataFeed, coord_feed: CoordinateFeed, star_coord_feed: CoordinateFeed):
        self.coord_feed = coord_feed
        self.star_coord_feed = star_coord_feed
        self.data_feed = data_feed
        self.continue_feed = ContinueFeed(self.coord_feed.time_feed)
        self.freqs = tf.convert_to_tensor(freqs,float_type)

    def filter_step(self, num_samples=10, num_chains=1, parallel_iterations=10, num_leapfrog_steps=2, target_rate=0.5,
                    num_burnin_steps=0, num_saem_samples = 10, saem_steps=10, saem_learning_rate=0.1):
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
                                             self.continue_feed.feed))
        joint_iterator = joint_dataset.make_initializable_iterator()

        # data_iterator = self.data_feed.feed.make_initializable_iterator()
        # coord_iterator = self.coord_feed.feed.make_initializable_iterator()
        # star_coord_iterator = self.star_coord_feed.feed.make_initializable_iterator()
        # continue_feed_iterator = self.continue_feed.feed.make_initializable_iterator()

        #y_sigma, variance, lengthscales, a, b, timescale, dtec
        step_sizes = [
            tf.get_variable(
                name='step_size_dtec',
                initializer=lambda: tf.constant(1e-3, dtype=float_type),
                use_resource=True,
                dtype=float_type,
                trainable=False)
        ]



        TAResult = namedtuple('TAResult',['y_sigma', 'variance', 'lengthscales', 'a', 'b',
                                      'timescale', 'dtec', 'Y_real', 'Y_imag', 'acc_ratio', 'post_logp', 'cont'])

        # y_sigma, variance, lengthscales, a, b, timescale, dtec, Y_real, Y_imag
        # tas = TAResult(tf.TensorArray(float_type, self.continue_feed.num_blocks, name='y_sigma'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='variance'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='lengthscales'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='a'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='b'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='timescale'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='dtec'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='y_real'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='y_imag'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='acc_ratio'),
        #        tf.TensorArray(float_type, self.continue_feed.num_blocks, name='post_logp'))

        # target0 = DTECToGainsSAEM(None, None, None, None, None,
        #                          fed_kernel='RBF', obs_type='DDTEC', num_chains=num_chains, variables=None, initialize=False)
        # variables = target0.variables

        # def cond(unused_i, cont, unused_tas):
        #     return cont

        # def body(i, unused_cont, tas):

        (Y_real, Y_imag), X, Xstar, cont = joint_iterator.get_next()

        # Nf
        invfreqs = -8.448e9 * tf.reciprocal(self.freqs)
        phase = tf.atan2(Y_real, Y_imag)
        # N
        dtec_init = tf.reduce_mean(phase / invfreqs, axis=-1)
        dtec_screen_init = tf.zeros(tf.shape(Xstar)[0:1],float_type)
        dtec_init = tf.concat([dtec_init,dtec_screen_init],axis=0)

        target = DTECToGainsSAEM(X, Xstar, Y_real, Y_imag, self.freqs,
                             fed_kernel='RBF', obs_type='DDTEC', num_chains = num_chains)

        inits = tf.group([step_size.initializer for step_size in step_sizes]
                         + [joint_iterator.initializer] + [target.variables.initializer])

        q0_init = target.get_initial_point(dtec_init)
        q0_init = [tf.tile(tf.reshape(q0_init, (-1,))[None, :], (num_chains, 1))]

        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target.logp,
            num_leapfrog_steps=num_leapfrog_steps,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
            step_size=step_sizes,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=None,
                                                                             decrement_multiplier=0.1,
                                                                             increment_multiplier=0.1,
                                                                             target_rate=target_rate),
            state_gradients_are_stopped=True)
        #                         step_size_update_fn=lambda v, _: v)

        # Run the chain (with burn-in).
        unconstrained_samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_saem_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=q0_init,
            kernel=hmc,
            parallel_iterations=parallel_iterations)

        flat_samples = flatten_batch_dims(unconstrained_samples)

        posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                                       name='marginal_log_likelihood')

        # def saem_cond(i):
        #     return tf.less(i, saem_steps)
        #
        # def saem_body(i):
        #     def log_prob(variables):
        #         # return tf.reduce_mean(target.logp(tf.stop_gradient(flat_samples)))
        #         return posterior_log_prob
        #     with tf.control_dependencies([target.variables]):
        #         logprob = log_prob(target.variables)
        #         g = tf.gradients(logprob, target.variables)[0]
        #         H = tf.hessians(logprob, target.variables)[0]
        #         dm = tf.matrix_solve_ls(H,g[:,None], fast=False)[:,0]
        #     with tf.control_dependencies([tf.print(i, "dm:",dm),tf.assign_add(target.variables, saem_learning_rate*dm)]):
        #         return [i+1]
        #
        # _ = tf.while_loop(saem_cond,
        #                   saem_body,
        #                   [tf.constant(0)],
        #                   parallel_iterations=1)

        # var_samples = tf.get_variable('var_samples',initializer=lambda: flat_samples)

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

        def log_prob(variables):
            target_saem = DTECToGainsSAEM(X, Xstar, Y_real, Y_imag, self.freqs,
                                     fed_kernel='RBF', obs_type='DDTEC', num_chains=1, variables = variables)
            with tf.control_dependencies([tf.print('Sample shape',tf.shape(flat_samples))]):
                return tf.reduce_mean(target_saem.logp(_percentiles(tf.stop_gradient(flat_samples), tf.linspace(0.,100.,3))))
            # return posterior_log_prob

        # with tf.control_dependencies([target.variables]):
        var_copy = tf.identity(target.variables)
        logprob = log_prob(var_copy)
        g = tf.gradients(logprob, var_copy)[0]
        H = tf.hessians(logprob, var_copy)[0]#tf.diag(tfp.math.diag_jacobian(xs=var_copy, ys=g, sample_shape=tf.shape(logprob))[0])#.hessians(logprob, var_copy)[0]
        dm = tf.matrix_solve_ls(H, g[:, None], fast=False)[:, 0]
        # dm = tf.matrix_solve_ls(H[0,:,0,:], g[:, None], fast=False)[:, 0]

        with tf.control_dependencies(
                [tf.print("dm:", dm),tf.assign_add(target.variables, saem_learning_rate * dm)]):
            flat_samples = tf.identity(flat_samples)
            q0_init = [tf.reduce_mean(unconstrained_samples,axis=0)]

        # hmc = tfp.mcmc.HamiltonianMonteCarlo(
        #     target_log_prob_fn=target.logp,
        #     num_leapfrog_steps=num_leapfrog_steps,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
        #     step_size=step_sizes,
        #     step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=None,
        #                                                                      decrement_multiplier=0.1,
        #                                                                      increment_multiplier=0.1,
        #                                                                      target_rate=target_rate),
        #     state_gradients_are_stopped=True)
        #                         step_size_update_fn=lambda v, _: v)

        # # Run the chain (with burn-in).
        # unconstrained_samples, kernel_results = tfp.mcmc.sample_chain(
        #     num_results=num_samples,
        #     num_burnin_steps=num_burnin_steps,
        #     current_state=q0_init,
        #     kernel=hmc,
        #     parallel_iterations=parallel_iterations)
        #
        # flat_samples = flatten_batch_dims(unconstrained_samples, 2)
        ddtec_transformed = target.transform_samples(flat_samples)
        Y_real_samples, Y_imag_samples = target.forward_equation(ddtec_transformed)

        avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                                              name='avg_acc_ratio')
        posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                           name='marginal_log_likelihood')




        output = TAResult(y_sigma = target.state.y_sigma.constrained_value,
                              variance = target.state.variance.constrained_value,
                              lengthscales = target.state.lengthscales.constrained_value,
                              a = target.state.a.constrained_value,
                              b = target.state.b.constrained_value,
                              timescale = target.state.timescale.constrained_value,
                              dtec = _percentiles(ddtec_transformed),
                              Y_real = _percentiles(Y_real_samples),
                              Y_imag = _percentiles(Y_imag_samples),
                              acc_ratio = avg_acceptance_ratio,
                              post_logp = posterior_log_prob,
                              cont = cont
                              )
        return output, inits

            # return [i + 1, cont, output_tas]

        # # t0 = timer()
        # with tf.control_dependencies([inits]):
        #     [_, done, output_tas] = tf.while_loop(cond,
        #                                        body,
        #                                        [tf.constant(0), tf.constant(True), tas])

        # output = TAResult(*[ta.concat() for ta in output_tas])

        # with tf.control_dependencies([done, tf.print("Time taken:", timer() - t0)]):
        # return output
