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

        def neg_log_prob(tec):
            phase_m = tec_conv * tec[..., None]
            Yimag_m = amp_pl * tf.sin(phase_m)
            Yreal_m = amp_pl * tf.cos(phase_m)

            # B, Nd, Na, Npol, Nt, Nf
            Yimag_like = tfp.distributions.Laplace(loc=Yimag_pl, scale=gain_uncert_pl).log_prob(Yimag_m)
            # B, Nd, Na, Npol, Nt, Nf
            Yreal_like = tfp.distributions.Laplace(loc=Yreal_pl, scale=gain_uncert_pl).log_prob(Yreal_m)
            # B, Nd, Na, Npol, Nt
            tec_prior = tfp.distributions.Normal(loc=tf.constant(0., float_type),
                                                 scale=tf.constant(55., float_type)).log_prob(tec)


            likelihood = tf.reduce_mean(Yimag_like, axis=-1) + tf.reduce_mean(Yreal_like, axis=-1)
            loss = -likelihood
            return loss

        search_space = tf.cast(tf.linspace(-200., 200., int(400./0.1) + 1), float_type)
        search_space = tf.tile(search_space[:,None,None,None,None],(1, Nd, Na, Npol, Nt))
        search_results = neg_log_prob(search_space)

        search_results = tf.reshape(search_results, (-1, Nd*Na*Npol*Nt))
        search_space = tf.reshape(search_space, (-1, Nd * Na * Npol * Nt))

        argmin = tf.argmin(search_results,axis=0)
        brute_solutions = tf.gather_nd(search_space,
                     tf.stack([argmin, tf.range(tf.size(argmin, out_type=tf.int64), dtype=tf.int64)], axis=1))
        brute_solutions = tf.reshape(brute_solutions, ( Nd, Na, Npol, Nt))
        final_loss = tf.reduce_min(search_results, axis=0)


        tec_perm = (2, 0, 1, 3)

        tec, final_loss = sess.run([brute_solutions, final_loss],
                                            feed_dict={Yimag_pl: Yimag.astype(np.float64),
                                                       Yreal_pl: Yreal.astype(np.float64),
                                                       amp_pl: amplitude,
                                                       gain_uncert_pl: gain_uncert[:, :, None, None, None]})

        return tec.transpose(tec_perm), final_loss


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
    phase_di = phase_raw[:, 14:15,...]# np.mean(phase_raw, axis=1, keepdims=True)
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

        tec, loss = infer_tec_and_clock(freqs, Yimag, Yreal,
                                       gain_uncert=gain_uncert,
                                       learning_rate=0.01,
                                       max_iters=2000,
                                       stop_patience=50,
                                       patience_percentage=1e-6,
                                       num_replicates=200)

        logging.info("Iteration {}: final loss min: {}".format(i,  loss))

        tec_conv = -8.4479745e6 / freqs[:, None]

        phase = tec[..., None, :] * tec_conv + phase_di[..., time_slice]

        save_tec.append(tec)
        save_phase.append(phase)

    _save_tec = np.concatenate(save_tec, axis=-1)
    _save_phase = np.concatenate(save_phase, axis=-1)


    datapack_save = DataPack(input_datapack, readonly=False)
    datapack_save.current_solset = 'data_posterior'
    # Npol, Nd, Na, Nf, Nt
    datapack_save.select(**select)
    datapack_save.phase = _save_phase
    datapack_save.tec = _save_tec
