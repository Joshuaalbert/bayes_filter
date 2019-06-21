###
# TF based solver clock and tec or just tec
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bayes_filter.datapack import DataPack
from bayes_filter import logging
from bayes_filter.sgd import adam_stochastic_gradient_descent_with_linesearch_batched
from bayes_filter.misc import maybe_create_posterior_solsets, get_screen_directions, flatten_batch_dims
import pylab as plt
from bayes_filter import float_type, logging


# import pymc3 as pm
#
# def infer_tec_and_clock_pymc3(freqs, Yimag, Yreal, per_dir_clock=False, gain_uncert=0.02, learning_rate=0.2, max_iters=1000, search_size=10, stop_patience=5, patient_percentage=1e-3, num_replicates=32):
#     # Npol, Nd, Na, Nf, Nt -> Nd, Na, Npol, Nt, Nf
#     Yimag = np.transpose(Yimag, (1, 2, 0, 4, 3))
#     Yreal = np.transpose(Yreal, (1, 2, 0, 4, 3))
#     shape = Yimag.shape
#     Nd, Na, Npol, Nt, Nf = shape
#     # # Nd, Nt*Na*Npol, Nf
#     # Yimag = Yimag.reshape((Nd, Nt * Na * Npol, Nf))
#     # Yreal = Yreal.reshape((Nd, Nt * Na * Npol, Nf))
#     tec_conv = -8.4479745e6/freqs
#     clock_conv = 2. * np.pi *   freqs
#     # Nd, Nt*Na*Npol
#     tec_init = np.mean(np.arctan2(Yimag, Yreal) / tec_conv, axis=-1)
#
#     with pm.Model() as model:
#         clock_uncert = pm.Exponential('clock_uncert', 1., shape=(Nd, Na,1, Nt), testval=1.)
#         tec_uncert = pm.Exponential('tec_uncert', 1. / 10., shape=(Nd, Na, 1, Nt), testval=10.)
#
#         clock = pm.Deterministic('clock', clock_uncert * pm.Normal('clock_', shape=(Nd, Na,1, Nt)))
#         tec_ = tec_uncert * pm.Normal('tec_', shape=(Nd, Na,1,Nt))
#
#         tec = pm.Deterministic('tec', tec_ + tec_init)
#
#         phase0 = tec[..., None] * tec_conv + clock[..., None] * clock_conv
#         phase_m = pm.Deterministic('phase', phase0 - pm.math.tt.tile(phase0[:, 0:1, :], (1, 62, 1, 1, 1)))
#
#         data_uncert = pm.Exponential('b', 1 / 0.07, testval=0.1, shape=(Nd, Na, 1,1,1))
#
#         data = pm.Laplace('Yimag', mu=pm.math.sin(phase_m), b=data_uncert, observed=Yimag) + pm.Laplace('Yreal',
#                                                                                                         mu=pm.math.cos(
#                                                                                                             phase_m),
#                                                                                                         b=data_uncert,
#                                                                                                         observed=Yreal)
#
#         state0 = pm.find_MAP(maxeval=10000)
#         clock_perm = (2, 0, 1, 3)
#         tec_perm = (2, 0, 1, 3)
#     return state0['tec'].transpose(tec_perm), state0['clock'].transpose(clock_perm), None

def infer_tec_and_clock(freqs, Yimag, Yreal, gain_uncert=0.02, learning_rate=0.2, max_iters=1000, stop_patience=5,
                        patience_percentage=1e-3, num_replicates=32):
    # Npol, Nd, Na, Nf, Nt -> Nd, Na, Npol, Nt, Nf
    Yimag = np.transpose(Yimag, (1, 2, 0, 4, 3))
    Yreal = np.transpose(Yreal, (1, 2, 0, 4, 3))
    shape = Yimag.shape
    Nd, Na, Npol, Nt, Nf = shape
    tec_conv = -8.4479745e6 / freqs
    clock_conv = 2. * np.pi * freqs * 1e-9
    amplitude = np.sqrt(Yimag ** 2 + Yreal ** 2)

    with tf.Session(graph=tf.Graph()) as sess:
        gain_uncert_pl = tf.placeholder(float_type, shape=(Nd, Na, 1, 1, 1), name='amp')
        amp_pl = tf.placeholder(float_type, shape=(Nd, Na, Npol, Nt, Nf), name='amp')
        Yimag_pl = tf.placeholder(float_type, shape=(Nd, Na, Npol, Nt, Nf), name='Yimag')
        Yreal_pl = tf.placeholder(float_type, shape=(Nd, Na, Npol, Nt, Nf), name='Yreal')

        freqs = tf.constant(freqs, float_type, name='freqs')
        data_uncert = tf.constant(gain_uncert, float_type, name='data_uncert')
        tec_conv = tf.constant(-8.4479745e6, float_type) * tf.math.reciprocal(freqs)
        clock_conv = tf.constant(2. * np.pi * 1e-9, float_type) * freqs

        def neg_log_prob(tec, clock):
            phase_m = tec_conv * tec[..., None] + clock_conv * clock[..., None]
            Yimag_m = amp_pl * tf.sin(phase_m)
            Yreal_m = amp_pl * tf.cos(phase_m)

            # B, Nd, Na, Npol, Nt, Nf
            Yimag_like = tfp.distributions.Laplace(loc=Yimag_pl, scale=gain_uncert_pl).log_prob(Yimag_m)
            # B, Nd, Na, Npol, Nt, Nf
            Yreal_like = tfp.distributions.Laplace(loc=Yreal_pl, scale=gain_uncert_pl).log_prob(Yreal_m)
            # B, 1, Na, Npol, Nt-1
            diff_clock_prior = tfp.distributions.Normal(loc=clock[..., 1:],
                                                        scale=tf.constant(0.02, float_type)).log_prob(clock[..., :-1])
            # B, Nd, Na, Npol, Nt
            tec_prior = tfp.distributions.Normal(loc=tf.constant(0., float_type),
                                                 scale=tf.constant(55., float_type)).log_prob(tec)
            # B, 1, Na, Npol, Nt
            clock_prior = tfp.distributions.Normal(loc=tf.constant(0., float_type),
                                                   scale=tf.constant(1., float_type)).log_prob(clock)

            likelihood = tf.reduce_sum(tf.reduce_mean(Yimag_like, axis=-1) + tf.reduce_mean(Yreal_like, axis=-1),
                                       axis=[1, 2, 3, 4]) \
                         + tf.reduce_sum(clock_prior, axis=[1, 2, 3, 4])
            # \
                         # + tf.reduce_sum(diff_clock_prior, axis=[1, 2, 3, 4]) \

            loss = -likelihood
            return loss

        init_state = [55. * tf.random.normal((num_replicates, Nd, Na, Npol, Nt), dtype=float_type),
                      1. * tf.random.normal((num_replicates, Nd, Na, Npol, Nt), dtype=float_type),
                      ]

        optim_results = tfp.optimizer.differential_evolution_minimize(neg_log_prob,
                                                                      init_state,
                                                                      max_iterations=max_iters,
                                                                      position_tolerance=1e-3,
                                                                      seed=0)

        # (final_tec, final_clock), loss = adam_stochastic_gradient_descent_with_linesearch_batched(num_replicates,
        #                                                                                           neg_log_prob,
        #                                                                                           init_state,
        #                                                                                           learning_rate=learning_rate,
        #                                                                                           iters=max_iters,
        #                                                                                           stop_patience=stop_patience,
        #                                                                                           patience_percentage=patience_percentage)
        #
        #
        #
        # final_loss = neg_log_prob(final_tec, final_clock)
        # argmin = tf.argmin(final_loss)
        # final_tec = final_tec[argmin,...]#tf.gather(final_tec, argmin, axis=0)
        # final_clock = final_clock[argmin,...]#tf.gather(final_clock, argmin, axis=0)


        clock_perm = (2, 0, 1, 3)
        tec_perm = (2, 0, 1, 3)

        (tec, clock), final_loss, converged, num_eval = sess.run([optim_results.position, optim_results.objective_value, optim_results.converged, optim_results.num_iterations],
                                            feed_dict={Yimag_pl: Yimag.astype(np.float64),
                                                       Yreal_pl: Yreal.astype(np.float64),
                                                       amp_pl: amplitude,
                                                       gain_uncert_pl: gain_uncert[:, :, None, None, None]})

        return tec.transpose(tec_perm), clock.transpose(clock_perm), final_loss, converged, num_eval


if __name__ == '__main__':

    input_datapack = '/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v6.h5'
    datapack = DataPack(input_datapack)
    screen_directions = get_screen_directions('/home/albert/ftp/image.pybdsm.srl.fits', max_N=None)
    maybe_create_posterior_solsets(datapack, 'sol000', posterior_name='posterior', screen_directions=screen_directions,
                                   remake_posterior_solsets=False)

    datapack.current_solset = 'sol000'
    axes = datapack.axes_phase
    _, times = datapack.get_times(axes['time'])
    Nt = len(times)

    select = dict(dir=slice(None, None, 1),
                  ant=slice(None, None, 1),
                  time=slice(0, Nt, 1),
                  freq=slice(None, None, 1),
                  pol=slice(0, 1, 1))

    datapack_raw = DataPack(input_datapack, readonly=True)
    datapack_raw.current_solset = 'sol000'
    # Npol, Nd, Na, Nf, Nt
    datapack_raw.select(**select)
    phase_raw, axes = datapack_raw.phase
    amp_raw, axes = datapack_raw.amplitude
    timestamps, times = datapack_raw.get_times(axes['time'])
    _, freqs = datapack_raw.get_freqs(axes['freq'])

    Yimag_full = amp_raw * np.sin(phase_raw)
    Yreal_full = amp_raw * np.cos(phase_raw)
    # Nd,Na
    gain_uncert = np.maximum(
        0.25 * np.mean(np.abs(np.diff(Yimag_full, axis=-1)) + np.abs(np.diff(Yreal_full, axis=-1)), axis=-1).mean(
            -1).mean(0), 0.02)

    block_size = 1
    save_freq = 1

    save_tec = []
    save_clock = []
    save_phase = []

    for i in range(0, Nt, block_size):
        time_slice = slice(i, min(i + block_size, Nt), 1)
        Yimag = Yimag_full[..., time_slice]
        Yreal = Yreal_full[..., time_slice]

        tec, clock, loss, converged, num_eval = infer_tec_and_clock(freqs, Yimag, Yreal,
                                               gain_uncert=gain_uncert,
                                               learning_rate=0.01,
                                               max_iters=2,
                                               stop_patience=50,
                                               patience_percentage=1e-6,
                                               num_replicates=200)

        logging.info("Iteration {}: coverged: {}, num iters: {}, final loss min: {}".format(i, converged, num_eval, loss))

        tec_conv = -8.4479745e6 / freqs[:, None]
        clock_conv = 2 * np.pi * freqs[:, None] * 1e-9

        phase = tec[..., None, :] * tec_conv + clock[..., None, :] * clock_conv

        save_tec.append(tec)
        save_clock.append(clock)
        save_phase.append(phase)

        if i % save_freq == 0:
            _save_tec = np.concatenate(save_tec, axis=-1)
            _save_clock = np.concatenate(save_clock, axis=-1)
            _save_phase = np.concatenate(save_phase, axis=-1)

            select = dict(dir=slice(None, None, 1),
                          ant=slice(None, None, 1),
                          time=slice(0, min(i + block_size, Nt), 1),
                          freq=slice(None, None, 1),
                          pol=slice(0, 1, 1))

            datapack_save = DataPack(input_datapack, readonly=False)
            datapack_save.current_solset = 'data_posterior'
            # Npol, Nd, Na, Nf, Nt
            datapack_save.select(**select)
            datapack_save.phase = _save_phase
            datapack_save.tec = _save_tec
            datapack_save.clock = _save_clock[:, :, :, :]

    _save_tec = np.concatenate(save_tec, axis=-1)
    _save_clock = np.concatenate(save_clock, axis=-1)
    _save_phase = np.concatenate(save_phase, axis=-1)

    select = dict(dir=slice(None, None, 1),
                  ant=slice(None, None, 1),
                  time=slice(0, Nt, 1),
                  freq=slice(None, None, 1),
                  pol=slice(0, 1, 1))

    datapack_save = DataPack(input_datapack, readonly=False)
    datapack_save.current_solset = 'data_posterior'
    # Npol, Nd, Na, Nf, Nt
    datapack_save.select(**select)
    datapack_save.phase = _save_phase
    datapack_save.tec = _save_tec
    datapack_save.clock = _save_clock[:, :, :, :]
