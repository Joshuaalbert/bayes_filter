###
# TF based solver clock and tec or just tec
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bayes_filter.datapack import DataPack
from bayes_filter.sgd import adam_stochastic_gradient_descent_with_linesearch
from bayes_filter.misc import maybe_create_posterior_solsets, get_screen_directions
import pylab as plt
from bayes_filter import float_type, logging


def infer_tec_and_clock(freqs, Yimag, Yreal, gain_uncert=0.02, learning_rate=0.2, max_iters=1000, search_size=10, stop_patience=5, patient_percentage=1e-3, num_replicates=32):
    # Npol, Nd, Na, Nf, Nt -> Nd, Na, Npol, Nt, Nf
    Yimag = np.transpose(Yimag, (1, 2, 0, 4, 3))
    Yreal = np.transpose(Yreal, (1, 2, 0, 4, 3))
    shape = Yimag.shape
    Nd, Na, Npol, Nt, Nf = shape
    # Nd, Nt*Na*Npol, Nf
    Yimag = Yimag.reshape((Nd, Nt * Na * Npol, Nf))
    Yreal = Yreal.reshape((Nd, Nt * Na * Npol, Nf))
    # Nd, Nt*Na*Npol
    tec_init = np.mean(np.arctan2(Yimag, Yreal) * freqs / -8.4479745e6, axis=-1)

    with tf.Session(graph=tf.Graph()) as sess:
        tec_init_pl = tf.placeholder(float_type, shape=tec_init.shape, name='tec_init')
        Yimag_pl = tf.placeholder(float_type, shape=Yimag.shape, name='Yimag')
        Yreal_pl = tf.placeholder(float_type, shape=Yreal.shape, name='Yreal')
        freqs = tf.constant(freqs, float_type, name='freqs')
        data_uncert = tf.constant(gain_uncert, float_type, name='data_uncert')
        tec_conv = tf.constant(-8.4479745e6, float_type) * tf.math.reciprocal(freqs)
        clock_conv = tf.constant(2. * np.pi * 1e-9, float_type) * freqs

        def log_prob(tec_, clock_):
            tec = tec_init_pl + tec_
            clock = clock_

            phase_m = tec_conv * tec[..., None] + clock_conv * clock[..., None]
            Yimag_m = tf.sin(phase_m)
            Yreal_m = tf.cos(phase_m)

            Yimag_dist = tfp.distributions.Laplace(loc=Yimag_pl, scale=data_uncert)
            Yreal_dist = tfp.distributions.Laplace(loc=Yreal_pl, scale=data_uncert)

            likelihood = [
                tf.reduce_sum(Yimag_dist.log_prob(Yimag_m)),
                tf.reduce_sum(Yreal_dist.log_prob(Yreal_m))]

            log_prob_ = tf.accumulate_n(likelihood, shape=())
            return -tf.identity(log_prob_)

        init_state = [40. * tf.random.normal((num_replicates, Nd, Nt * Na * Npol), dtype=float_type),
                      1. * tf.random.normal((num_replicates, 1, Nt * Na * Npol), dtype=float_type)
                      ]

        (final_tec, final_clock), loss = adam_stochastic_gradient_descent_with_linesearch(log_prob,
                                                                                          init_state,
                                                                                          learning_rate=learning_rate,
                                                                                          iters=max_iters,
                                                                                          search_size=search_size,
                                                                                          stop_patience=stop_patience,
                                                                                          patient_percentage=patient_percentage)

        tec_shape = (Nd, Na, Npol, Nt)
        tec_perm = (2, 0, 1, 3)
        clock_shape = (1, Na, Npol, Nt)
        clock_perm = (2, 0, 1, 3)

        (tec, clock), loss = sess.run([(final_tec + tec_init_pl, final_clock), loss],
                                      feed_dict={Yimag_pl: Yimag.astype(np.float64),
                                                 Yreal_pl: Yreal.astype(np.float64),
                                                 tec_init_pl: tec_init.astype(np.float64)})
        return np.median(tec, axis=0).reshape(tec_shape).transpose(tec_perm), np.median(clock, axis=0).reshape(
            clock_shape).transpose(clock_perm), loss

if __name__ == '__main__':

    input_datapack = '/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v5.h5'
    datapack = DataPack(input_datapack)
    screen_directions = get_screen_directions('/home/albert/ftp/image.pybdsm.srl.fits', max_N=None)
    maybe_create_posterior_solsets(datapack, 'sol000', posterior_name='posterior', screen_directions=screen_directions, remake_posterior_solsets=True)

    datapack.current_solset = 'sol000'
    axes = datapack.axes_phase
    _, times = datapack.get_times(axes['time'])
    Nt = len(times)

    block_size = 1

    for i in range(0, Nt, block_size):
        select = dict(dir=slice(None, None, 1),
                      ant=slice(None, None, 1),
                      time=slice(i, min(i + block_size, Nt), 1),
                      freq=slice(None, None, 1),
                      pol=slice(0, 1, 1))

        datapack_raw = DataPack(input_datapack, readonly=True)
        datapack_raw.current_solset = 'sol000'
        # Npol, Nd, Na, Nf, Nt
        datapack_raw.select(**select)
        phase_raw, axes = datapack_raw.phase
        timestamps, times = datapack_raw.get_times(axes['time'])
        _, freqs = datapack_raw.get_freqs(axes['freq'])

        Yimag = np.sin(phase_raw)
        Yreal = np.cos(phase_raw)

        tec, clock, loss = infer_tec_and_clock(freqs, Yimag, Yreal,
                                               gain_uncert=0.02,
                                               learning_rate=0.2,
                                               max_iters=1,
                                               search_size=10,
                                               stop_patience=5,
                                               patient_percentage=1e-3,
                                               num_replicates=32)

        tec_conv = -8.4479745e6 / freqs[:, None]
        clock_conv = 2 * np.pi * freqs[:, None] * 1e-9

        phase = tec[..., None, :] * tec_conv + clock[..., None, :] * clock_conv

        datapack_save = DataPack(input_datapack, readonly=False)
        datapack_save.current_solset = 'data_posterior'
        # Npol, Nd, Na, Nf, Nt
        datapack_save.select(**select)
        datapack_save.phase = phase
        datapack_save.tec = tec
        datapack_save.clock = clock[:,0,:,:]
