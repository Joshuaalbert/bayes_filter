"""
This is a modification of tensorflow probability's tfp.mcmc.sample function that allows dynamic stopping
using rhat criteria. When median rhat per parameters falls by less than a certain percent.

Figure out proper liscensing later.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings
# Dependency imports

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

from tensorflow.python.ops import control_flow_util
import numpy as np


__all__ = [
    "CheckpointableStatesAndTrace",
    "StatesAndTrace",
    "sample_chain",
]


def _reduce_variance(x, axis=None, biased=True, keepdims=False):
    with tf.compat.v1.name_scope('reduce_variance'):
        x = tf.convert_to_tensor(value=x, name='x')
        mean = tf.reduce_mean(input_tensor=x, axis=axis, keepdims=True)
        biased_var = tf.reduce_mean(
            input_tensor=tf.math.squared_difference(x, mean),
            axis=axis,
            keepdims=keepdims)
        if biased:
            return biased_var
        n = _axis_size(x, axis)
        return (n / (n - 1.)) * biased_var

def _axis_size(x, axis=None):
    """Get number of elements of `x` in `axis`, as type `x.dtype`."""
    if axis is None:
        return tf.cast(tf.size(input=x), x.dtype)
    return tf.cast(
        tf.reduce_prod(input_tensor=tf.gather(tf.shape(input=x), axis)), x.dtype)

def _get_rhat_onestate(state, delta_rhat, rhat, sample_sum, count, m, v, decay_length=100, independent_chain_ndims=1):
    tau = tf.math.reciprocal(tf.convert_to_tensor(decay_length, state.dtype))
    tau_ = 1. - tau
    count = tau_*count + tau * tf.constant(1, state.dtype)

    # variance of chain means
    sample_sum = tau_*sample_sum + tau*state
    sample_mean = sample_sum/count
    chain_axis = tf.range(0, independent_chain_ndims)
    b_div_n = _reduce_variance(sample_mean, axis=chain_axis, biased=False)

    delta= state - m
    m = tau_*m + tau * delta/count
    v = tau_*v + tau * delta*(state - m)

    sample_variance = v/count #biased
    w = tf.reduce_mean(sample_variance, axis=chain_axis)

    N = count
    M = _axis_size(state, chain_axis)

    sigma_2_plus = w + b_div_n
    next_rhat = ((M + 1.) / M) * sigma_2_plus / w - (N - 1.) / (M*N)
    delta_rhat = next_rhat - rhat

    # (tau*(1-tau) + tau)*(1-tau) + tau
    # (tau - tau^2 + tau)*(1 - tau) + tau
    # tau*( (1-tau) + 1)(1-tau) + tau
    # tau

    return delta_rhat, next_rhat, sample_sum, count, m, v

def _get_rhat(next_state, _delta_rhat, _rhat, _sample_sum, _count, _m, _v, independent_chain_ndims=1):
    list_like = isinstance(next_state, (tuple, list))
    if not list_like:
        next_state = [next_state]
        _delta_rhat = [_delta_rhat]
        _rhat = [_rhat]
        _sample_sum = [_sample_sum]
        _count = [_count]
        _m = [_m]
        _v = [_v]

    _next_delta_rhat, _next_rhat, _next_sample_sum, _next_count, _next_m, _next_v = [],[],[],[], [],[]

    for (delta_rhat, rhat, sample_sum, count, m, v, s) in zip(_delta_rhat, _rhat,_sample_sum, _count, _m, _v, next_state):
        state = tf.convert_to_tensor(value=s, name='state')
        next_delta_rhat, next_rhat, next_sample_sum, next_count, next_m, next_v = \
            _get_rhat_onestate(
                state,
                delta_rhat,
                rhat,
                sample_sum,
                count,
                m,
                v,
                independent_chain_ndims=independent_chain_ndims)
        _next_delta_rhat.append(next_delta_rhat)
        _next_rhat.append(next_rhat)
        _next_sample_sum.append(next_sample_sum)
        _next_count.append(next_count)
        _next_m.append(next_m)
        _next_v.append(next_v)

    if not list_like:
        _next_delta_rhat, _next_rhat, _next_sample_sum, _next_count, _next_m, _next_v = (
            _next_delta_rhat[0], _next_rhat[0],
            _next_sample_sum[0], _next_count[0], _next_m[0], _next_v[0])

    return _next_delta_rhat, _next_rhat, _next_sample_sum, _next_count, _next_m, _next_v


def _initial_rhat_variables(init_state, independent_chain_ndims=1):
    initial_sample_sum, initial_count, initial_m, initial_v = [], [], [], []

    initial_rhats = []
    initial_delta_rhats = []

    list_like = isinstance(init_state, (tuple, list))
    if not list_like:
        init_state = [init_state]

    for s in init_state:
        state = tf.convert_to_tensor(value=s, name='init_state')

        initial_sample_sum.append(tf.zeros_like(state, name='sample_sum'))
        initial_count.append(tf.constant(0., dtype=state.dtype, name='count'))
        initial_m.append(tf.zeros_like(state, name='m'))
        initial_v.append(tf.zeros_like(state, name='v'))
        initial_rhats.append(tf.constant(1e15,state.dtype)*tf.ones(tf.shape(state)[independent_chain_ndims:], dtype=state.dtype))
        initial_delta_rhats.append(tf.constant(1e15,state.dtype)*tf.ones(tf.shape(state)[independent_chain_ndims:], dtype=state.dtype))


    if not list_like:
        initial_sample_sum, initial_count, initial_m, initial_v = (
            initial_sample_sum[0], initial_count[0], initial_m[0], initial_v[0]
        )
        initial_rhats = initial_rhats[0]
        initial_delta_rhats = initial_delta_rhats[0]

    return initial_delta_rhats, initial_rhats, initial_sample_sum, initial_count, initial_m, initial_v


# def _get_rhat_onestate(state, delta_rhat, rhat, sample_sum, count, delta_mean, M2, independent_chain_ndims=1):
#     count += tf.constant(1, state.dtype)
#     sample_sum += state
#     sample_mean = sample_sum/count
#     chain_axis = tf.range(0, independent_chain_ndims)
#     b_div_n = _reduce_variance(sample_mean, axis=chain_axis, biased=False)
#
#     delta = state - delta_mean
#     delta_mean += delta / count
#     delta2 = state - delta_mean
#     M2 += delta*delta2
#     sample_variance = M2/count #biased
#     w = tf.reduce_mean(sample_variance, axis=chain_axis)
#
#     n = count
#     m = _axis_size(state, chain_axis)
#
#     sigma_2_plus = w + b_div_n
#     next_rhat = ((m + 1.) / m) * sigma_2_plus / w - (n - 1.) / (m * n)
#     delta_rhat = next_rhat - rhat
#
#     return delta_rhat, next_rhat, sample_sum, count, delta_mean, M2

# def _get_rhat(next_state, _delta_rhat, _rhat, _sample_sum, _count, _delta_mean, _M2, independent_chain_ndims=1):
#     list_like = isinstance(next_state, (tuple, list))
#     print(next_state)
#     if not list_like:
#         next_state = [next_state]
#         _delta_rhat = [_delta_rhat]
#         _rhat = [_rhat]
#         _sample_sum = [_sample_sum]
#         _count = [_count]
#         _delta_mean = [_delta_mean]
#         _M2 = [_M2]
#
#     _next_delta_rhat, _next_rhat, _next_sample_sum, _next_count, _next_delta_mean, _next_M2 = [],[],[],[], [],[]
#
#     for (delta_rhat, rhat, sample_sum, count, delta_mean, M2, s) in zip(_delta_rhat, _rhat,_sample_sum, _count, _delta_mean, _M2, next_state):
#         state = tf.convert_to_tensor(value=s, name='state')
#         next_delta_rhat, next_rhat, next_sample_sum, next_count, next_delta_mean, next_M2 = \
#             _get_rhat_onestate(
#                 state,
#                 delta_rhat,
#                 rhat,
#                 sample_sum,
#                 count,
#                 delta_mean,
#                 M2,
#                 independent_chain_ndims=independent_chain_ndims)
#         _next_delta_rhat.append(next_delta_rhat)
#         _next_rhat.append(next_rhat)
#         _next_sample_sum.append(next_sample_sum)
#         _next_count.append(next_count)
#         _next_delta_mean.append(next_delta_mean)
#         _next_M2.append(next_M2)
#
#     if not list_like:
#         _next_delta_rhat, _next_rhat, _next_sample_sum, _next_count, _next_delta_mean, _next_M2 = (
#             _next_delta_rhat[0], _next_rhat[0],
#             _next_sample_sum[0], _next_count[0], _next_delta_mean[0], _next_M2[0])
#
#     return _next_delta_rhat, _next_rhat, _next_sample_sum, _next_count, _next_delta_mean, _next_M2



# def _initial_rhat_variables(init_state, independent_chain_ndims=1):
#     initial_sample_sum, initial_count, initial_delta_mean, initial_M2 = [], [], [], []
#
#     initial_rhats = []
#     initial_delta_rhats = []
#
#     list_like = isinstance(init_state, (tuple, list))
#     if not list_like:
#         init_state = [init_state]
#
#     for s in init_state:
#         state = tf.convert_to_tensor(value=s, name='init_state')
#
#         initial_sample_sum.append(tf.zeros_like(state))
#         initial_count.append(tf.constant(0., dtype=state.dtype))
#         initial_delta_mean.append(tf.zeros_like(state))
#         initial_M2.append(tf.zeros_like(state))
#         initial_rhats.append(tf.constant(1e15,state.dtype)*tf.ones(tf.shape(state)[independent_chain_ndims:], dtype=state.dtype))
#         initial_delta_rhats.append(tf.constant(1e15,state.dtype)*tf.ones(tf.shape(state)[independent_chain_ndims:], dtype=state.dtype))
#
#
#     if not list_like:
#         initial_sample_sum, initial_count, initial_delta_mean, initial_M2 = (
#             initial_sample_sum[0], initial_count[0], initial_delta_mean[0], initial_M2[0]
#         )
#         initial_rhats = initial_rhats[0]
#         initial_delta_rhats = initial_delta_rhats[0]
#
#     return initial_delta_rhats, initial_rhats, initial_sample_sum, initial_count, initial_delta_mean, initial_M2



###
# BEGIN: Change

def trace_scan(loop_fn,
               initial_state,
               elems,
               trace_fn,
               parallel_iterations=10,
               name=None):
    """A simplified version of `tf.scan` that has configurable tracing.

    This function repeatedly calls `loop_fn(state, elem)`, where `state` is the
    `initial_state` during the first iteration, and the return value of `loop_fn`
    for every iteration thereafter. `elem` is a slice of `elements` along the
    first dimension, accessed in order. Additionally, it calls `trace_fn` on the
    return value of `loop_fn`. The `Tensor`s in return values of `trace_fn` are
    stacked and returned from this function, such that the first dimension of
    those `Tensor`s matches the size of `elems`.

    Args:
      loop_fn: A callable that takes in a `Tensor` or a nested collection of
        `Tensor`s with the same structure as `initial_state`, a slice of `elems`
        and returns the same structure as `initial_state`.
      initial_state: A `Tensor` or a nested collection of `Tensor`s passed to
        `loop_fn` in the first iteration.
      elems: A `Tensor` that is split along the first dimension and each element
        of which is passed to `loop_fn`.
      trace_fn: A callable that takes in the return value of `loop_fn` and returns
        a `Tensor` or a nested collection of `Tensor`s.
      parallel_iterations: Passed to the internal `tf.while_loop`.
      name: Name scope used in this function. Default: 'trace_scan'.

    Returns:
      final_state: The final return value of `loop_fn`.
      trace: The same structure as the return value of `trace_fn`, but with each
        `Tensor` being a stack of the corresponding `Tensors` in the return value
        of `trace_fn` for each slice of `elems`.
    """
    with tf.compat.v1.name_scope(
            name, 'trace_scan', [initial_state, elems]), tf.compat.v1.variable_scope(
        tf.compat.v1.get_variable_scope()) as vs:
        if vs.caching_device is None and not tf.executing_eagerly():
            vs.set_caching_device(lambda op: op.device)

        initial_state = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(value=x, name='initial_state'),
            initial_state)

        elems = tf.convert_to_tensor(value=elems, name='elems')

        static_length = elems.shape[0]
        if tf.compat.dimension_value(static_length) is None:
            length = tf.shape(input=elems)[0]
        else:
            length = tf.convert_to_tensor(
                value=static_length, dtype=tf.int32, name='length')

        # This is an TensorArray in part because of XLA, which had trouble with
        # non-statically known indices. I.e. elems[i] errored, but
        # elems_array.read(i) worked.
        elems_array = tf.TensorArray(
            elems.dtype, size=length, element_shape=elems.shape[1:])
        elems_array = elems_array.unstack(elems)

        trace_arrays = tf.nest.map_structure(
            lambda x: tf.TensorArray(x.dtype, size=length, element_shape=x.shape),
            trace_fn(initial_state))

        def _body(i, state, trace_arrays, rhat_and_vars):
            state = loop_fn(state, elems_array.read(i))
            trace_arrays = tf.nest.pack_sequence_as(trace_arrays, [
                a.write(i, v) for a, v in zip(
                    tf.nest.flatten(trace_arrays), tf.nest.flatten(trace_fn(state)))
            ])
            rhat_and_vars = _get_rhat(state[0], *rhat_and_vars)
            return i + 1, state, trace_arrays, rhat_and_vars

        def _cond(i, state, trace_array, rhat_and_vars):
            default_cond = i < length
            delta_rhat, rhat = rhat_and_vars[0], rhat_and_vars[1]
            if not isinstance(rhat, (list, tuple)):
                delta_rhat, rhat = [delta_rhat], [rhat]

            dynamic_cond_A = tf.reduce_any([tf.greater(tf.reduce_mean(r), 1.2) for r in rhat])
            dynamic_cond_B = tf.reduce_any([tf.greater(tf.reduce_mean(tf.abs(dr)/tf.abs(r)), 0.05) for dr, r in zip(delta_rhat,rhat)])
            dynamic_cond_C = tf.reduce_any([tf.greater(tfp.stats.percentile(tf.math.abs(dr),50), 1e-4) for dr in delta_rhat])

            dynamic_cond = dynamic_cond_C#tf.logical_and(dynamic_cond_A, dynamic_cond_B)
            mean_rhat = tf.nest.map_structure(lambda r: tfp.stats.percentile(r,50), rhat)
            mean_drhat = tf.nest.map_structure(lambda r: tfp.stats.percentile(r,50), delta_rhat)
            rel_change = tf.nest.map_structure(lambda dr, r: tfp.stats.percentile(tf.abs(dr)/tf.abs(r), 50), delta_rhat,rhat )
            with tf.control_dependencies([tf.print(i, rel_change, mean_drhat, mean_rhat)]):
                return tf.logical_and(default_cond, dynamic_cond)

        init_rhat_vars = _initial_rhat_variables(initial_state[0])

        _, final_state, trace_arrays, rhat_and_vars = tf.while_loop(
            cond=_cond,
            body=_body,
            loop_vars=(0, initial_state, trace_arrays, init_rhat_vars),
            parallel_iterations=parallel_iterations)

        stacked_trace = tf.nest.map_structure(lambda x: x.stack(), trace_arrays)

        # Restore the static length if we know it.
        def _merge_static_length(x):
            x.set_shape(tf.TensorShape(static_length).concatenate(x.shape[1:]))
            return x

        stacked_trace = tf.nest.map_structure(_merge_static_length, stacked_trace)
        return final_state, stacked_trace

###
# END: Change

# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.filterwarnings("always",
                        module="tensorflow_probability.*sample",
                        append=True)  # Don't override user-set filters.


class StatesAndTrace(
    collections.namedtuple("StatesAndTrace", "all_states, trace")):
    """States and auxiliary trace of an MCMC chain.
    The first dimension of all the `Tensor`s in this structure is the same and
    represents the chain length.
    Attributes:
      all_states: A `Tensor` or a nested collection of `Tensor`s representing the
        MCMC chain state.
      trace: A `Tensor` or a nested collection of `Tensor`s representing the
        auxiliary values traced alongside the chain.
    """
    __slots__ = ()


class CheckpointableStatesAndTrace(
    collections.namedtuple("CheckpointableStatesAndTrace",
                           "all_states, trace, final_kernel_results")):
    """States and auxiliary trace of an MCMC chain.
    The first dimension of all the `Tensor`s in the `all_states` and `trace`
    attributes is the same and represents the chain length.
    Attributes:
      all_states: A `Tensor` or a nested collection of `Tensor`s representing the
        MCMC chain state.
      trace: A `Tensor` or a nested collection of `Tensor`s representing the
        auxiliary values traced alongside the chain.
      final_kernel_results: A `Tensor` or a nested collection of `Tensor`s
        representing the final value of the auxiliary state of the
        `TransitionKernel` that generated this chain.
    """
    __slots__ = ()


def sample_chain(
        num_results,
        current_state,
        previous_kernel_results=None,
        kernel=None,
        num_burnin_steps=0,
        num_steps_between_results=0,
        trace_fn=lambda current_state, kernel_results: kernel_results,
        return_final_kernel_results=False,
        parallel_iterations=10,
        cond_fn=None,
        name=None,
):
    """Implements Markov chain Monte Carlo via repeated `TransitionKernel` steps.
    This function samples from an Markov chain at `current_state` and whose
    stationary distribution is governed by the supplied `TransitionKernel`
    instance (`kernel`).
    This function can sample from multiple chains, in parallel. (Whether or not
    there are multiple chains is dictated by the `kernel`.)
    The `current_state` can be represented as a single `Tensor` or a `list` of
    `Tensors` which collectively represent the current state.
    Since MCMC states are correlated, it is sometimes desirable to produce
    additional intermediate states, and then discard them, ending up with a set of
    states with decreased autocorrelation.  See [Owen (2017)][1]. Such "thinning"
    is made possible by setting `num_steps_between_results > 0`. The chain then
    takes `num_steps_between_results` extra steps between the steps that make it
    into the results. The extra steps are never materialized (in calls to
    `sess.run`), and thus do not increase memory requirements.
    Warning: when setting a `seed` in the `kernel`, ensure that `sample_chain`'s
    `parallel_iterations=1`, otherwise results will not be reproducible.
    In addition to returning the chain state, this function supports tracing of
    auxiliary variables used by the kernel. The traced values are selected by
    specifying `trace_fn`. By default, all kernel results are traced but in the
    future the default will be changed to no results being traced, so plan
    accordingly. See below for some examples of this feature.
    Args:
      num_results: Integer number of Markov chain draws.
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s).
      previous_kernel_results: A `Tensor` or a nested collection of `Tensor`s
        representing internal calculations made within the previous call to this
        function (or as returned by `bootstrap_results`).
      kernel: An instance of `tfp.mcmc.TransitionKernel` which implements one step
        of the Markov chain.
      num_burnin_steps: Integer number of chain steps to take before starting to
        collect results.
        Default value: 0 (i.e., no burn-in).
      num_steps_between_results: Integer number of chain steps between collecting
        a result. Only one out of every `num_steps_between_samples + 1` steps is
        included in the returned results.  The number of returned chain states is
        still equal to `num_results`.  Default value: 0 (i.e., no thinning).
      trace_fn: A callable that takes in the current chain state and the previous
        kernel results and return a `Tensor` or a nested collection of `Tensor`s
        that is then traced along with the chain state.
      return_final_kernel_results: If `True`, then the final kernel results are
        returned alongside the chain state and the trace specified by the
        `trace_fn`.
      parallel_iterations: The number of iterations allowed to run in parallel. It
        must be a positive integer. See `tf.while_loop` for more details.
      cond_fn: callable
        Dynmaic termination condition that returns True if the sampler should continue.
        Call pattern func(i, state, trace)
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "mcmc_sample_chain").
    Returns:
      checkpointable_states_and_trace: if `return_final_kernel_results` is
        `True`. The return value is an instance of
        `CheckpointableStatesAndTrace`.
      all_states: if `return_final_kernel_results` is `False` and `trace_fn` is
        `None`. The return value is a `Tensor` or Python list of `Tensor`s
        representing the state(s) of the Markov chain(s) at each result step. Has
        same shape as input `current_state` but with a prepended
        `num_results`-size dimension.
      states_and_trace: if `return_final_kernel_results` is `False` and
        `trace_fn` is not `None`. The return value is an instance of
        `StatesAndTrace`.
    #### Examples
    ##### Sample from a diagonal-variance Gaussian.
    I.e.,
    ```none
    for i=1..n:
      x[i] ~ MultivariateNormal(loc=0, scale=diag(true_stddev))  # likelihood
    ```
    ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    dims = 10
    true_stddev = np.sqrt(np.linspace(1., 3., dims))
    likelihood = tfd.MultivariateNormalDiag(loc=0., scale_diag=true_stddev)
    states = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=500,
        current_state=tf.zeros(dims),
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=likelihood.log_prob,
          step_size=0.5,
          num_leapfrog_steps=2),
        trace_fn=None)
    sample_mean = tf.reduce_mean(states, axis=0)
    # ==> approx all zeros
    sample_stddev = tf.sqrt(tf.reduce_mean(
        tf.squared_difference(states, sample_mean),
        axis=0))
    # ==> approx equal true_stddev
    ```
    ##### Sampling from factor-analysis posteriors with known factors.
    I.e.,
    ```none
    # prior
    w ~ MultivariateNormal(loc=0, scale=eye(d))
    for i=1..n:
      # likelihood
      x[i] ~ Normal(loc=w^T F[i], scale=1)
    ```
    where `F` denotes factors.
    ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    # Specify model.
    def make_prior(dims):
      return tfd.MultivariateNormalDiag(
          loc=tf.zeros(dims))
    def make_likelihood(weights, factors):
      return tfd.MultivariateNormalDiag(
          loc=tf.matmul(weights, factors, adjoint_b=True))
    def joint_log_prob(num_weights, factors, x, w):
      return (make_prior(num_weights).log_prob(w) +
              make_likelihood(w, factors).log_prob(x))
    def unnormalized_log_posterior(w):
      # Posterior is proportional to: `p(W, X=x | factors)`.
      return joint_log_prob(num_weights, factors, x, w)
    # Setup data.
    num_weights = 10 # == d
    num_factors = 40 # == n
    num_chains = 100
    weights = make_prior(num_weights).sample(1)
    factors = tf.random_normal([num_factors, num_weights])
    x = make_likelihood(weights, factors).sample()
    # Sample from Hamiltonian Monte Carlo Markov Chain.
    # Get `num_results` samples from `num_chains` independent chains.
    chains_states, kernels_results = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=500,
        current_state=tf.zeros([num_chains, num_weights], name='init_weights'),
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=unnormalized_log_posterior,
          step_size=0.1,
          num_leapfrog_steps=2))
    # Compute sample stats.
    sample_mean = tf.reduce_mean(chains_states, axis=[0, 1])
    # ==> approx equal to weights
    sample_var = tf.reduce_mean(
        tf.squared_difference(chains_states, sample_mean),
        axis=[0, 1])
    # ==> less than 1
    ```
    ##### Custom tracing functions.
    ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    likelihood = tfd.Normal(loc=0., scale=1.)
    def sample_chain(trace_fn):
      return tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=500,
        current_state=0.,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=likelihood.log_prob,
          step_size=0.5,
          num_leapfrog_steps=2),
        trace_fn=trace_fn)
    def trace_log_accept_ratio(states, previous_kernel_results):
      return previous_kernel_results.log_accept_ratio
    def trace_everything(states, previous_kernel_results):
      return previous_kernel_results
    _, log_accept_ratio = sample_chain(trace_fn=trace_log_accept_ratio)
    _, kernel_results = sample_chain(trace_fn=trace_everything)
    acceptance_prob = tf.exp(tf.minimum(log_accept_ratio_, 0.))
    # Equivalent to, but more efficient than:
    acceptance_prob = tf.exp(tf.minimum(kernel_results.log_accept_ratio_, 0.))
    ```
    #### References
    [1]: Art B. Owen. Statistically efficient thinning of a Markov chain sampler.
         _Technical Report_, 2017.
         http://statweb.stanford.edu/~owen/reports/bestthinning.pdf
    """

    if not kernel.is_calibrated:
        warnings.warn("supplied `TransitionKernel` is not calibrated. Markov "
                      "chain may not converge to intended target distribution.")
    with tf.compat.v1.name_scope(
            name, "mcmc_sample_chain",
            [num_results, num_burnin_steps, num_steps_between_results]):
        num_results = tf.convert_to_tensor(
            value=num_results, dtype=tf.int32, name="num_results")
        num_burnin_steps = tf.convert_to_tensor(
            value=num_burnin_steps, dtype=tf.int32, name="num_burnin_steps")
        num_steps_between_results = tf.convert_to_tensor(
            value=num_steps_between_results,
            dtype=tf.int32,
            name="num_steps_between_results")
        current_state = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(value=x, name="current_state"),
            current_state)
        if previous_kernel_results is None:
            previous_kernel_results = kernel.bootstrap_results(current_state)

        if trace_fn is None:
            # It simplifies the logic to use a dummy function here.
            trace_fn = lambda *args: ()
            no_trace = True
        else:
            no_trace = False
        if trace_fn is sample_chain.__defaults__[4]:
            warnings.warn("Tracing all kernel results by default is deprecated. Set "
                          "the `trace_fn` argument to None (the future default "
                          "value) or an explicit callback that traces the values "
                          "you are interested in.")

        def _trace_scan_fn(state_and_results, num_steps):
            next_state, current_kernel_results = mcmc_util.smart_for_loop(
                loop_num_iter=num_steps,
                body_fn=kernel.one_step,
                initial_loop_vars=list(state_and_results),
                parallel_iterations=parallel_iterations)
            return next_state, current_kernel_results

        (_, final_kernel_results), (all_states, trace) = trace_scan(
            loop_fn=_trace_scan_fn,
            initial_state=(current_state, previous_kernel_results),
            elems=tf.one_hot(
                indices=0,
                depth=num_results,
                on_value=1 + num_burnin_steps,
                off_value=1 + num_steps_between_results,
                dtype=tf.int32),
            # pylint: disable=g-long-lambda
            trace_fn=lambda state_and_results: (state_and_results[0],
                                                trace_fn(*state_and_results)),
            # pylint: enable=g-long-lambda
            parallel_iterations=parallel_iterations)

        if return_final_kernel_results:
            return CheckpointableStatesAndTrace(
                all_states=all_states,
                trace=trace,
                final_kernel_results=final_kernel_results)
        else:
            if no_trace:
                return all_states
            else:
                return StatesAndTrace(all_states=all_states, trace=trace)