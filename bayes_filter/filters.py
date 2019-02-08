import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .data_feed import init_feed, DataFeed, CoordinateFeed
from .settings import float_type
from collections import namedtuple
from .logprobabilities import Target, DTECToGains

UpdateResult = namedtuple('UpdateResult',['x_samples','z_samples','log_prob', 'acceptance','step_size'])

class FreeTransition(object):
    def __init__(self, data_feed: DataFeed, coord_feed: CoordinateFeed, target: Target.__class__):
        self.coord_feed = coord_feed
        self.data_feed = data_feed
        self.target = target

    def filter(self, *q0, num_samples=10, parallel_iteration=1, num_leapfrog_steps=2, target_rate=0.75, init_stepsize=0.1):
        """
        Run a Bayes filter over the coordinate set.

        :param q0: list of float_type, Tensor, [num_chains, M, Np]
            Initial state point
        :param num_samples: int
        :param parallel_iteration: int
        :param num_leapfrog_steps:
        :param target_rate:
        :param init_stepsize:
        :return: list of UpdateResult
            The results of each filter step
        """
        data_iterator = self.data_feed.feed.make_initializable_iterator()
        data_iterator_init = data_iterator.initializer
        coord_iterator = self.coord_feed.feed.make_initializable_iterator()
        coord_iterator_init = coord_iterator.initializer

        step_size = tf.get_variable(
            name='step_size',
            initializer=lambda: tf.constant(init_stepsize, dtype=float_type),
            use_resource=True,
            dtype=float_type,
            trainable=False)

        inits = tf.group([step_size.initializer, data_iterator_init, coord_iterator_init])

        def body():
            next_data = data_iterator.get_next()
            next_coords = coord_iterator.get_next()
            target = self.target(next_coords, self.coord_feed.dims, *next_data, )

            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.target.logp,
                num_leapfrog_steps=num_leapfrog_steps,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
                step_size=step_size,
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(target_rate=target_rate),
                state_gradients_are_stopped=True)
            #                         step_size_update_fn=lambda v, _: v)

            # Run the chain (with burn-in).
            z_samples, kernel_results = tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=0,
                current_state=self.target.unconstrained_states(*q0),
                kernel=hmc,
                parallel_iteration=parallel_iteration)

            x_samples = self.target.constrained_states(*z_samples)

            avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                                                  name='avg_acc_ratio')
            posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                               name='marginal_log_likelihood')

            res = UpdateResult(x_samples, z_samples, posterior_log_prob, avg_acceptance_ratio,
                               kernel_results.extra.step_size_assign)
            return res