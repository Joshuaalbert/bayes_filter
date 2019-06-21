###
# TF based solver clock and tec or just tec
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bayes_filter.datapack import DataPack
from bayes_filter import logging
from bayes_filter.sgd import adam_stochastic_gradient_descent_with_linesearch_batch
from bayes_filter.misc import maybe_create_posterior_solsets, get_screen_directions
import pylab as plt
from bayes_filter import float_type, logging
import pymc3 as pm

def infer_tec_and_clock_pymc3(freqs, Yimag, Yreal, per_dir_clock=False, gain_uncert=0.02, learning_rate=0.2, max_iters=1000, search_size=10, stop_patience=5, patient_percentage=1e-3, num_replicates=32):
    # Npol, Nd, Na, Nf, Nt -> Nd, Na, Npol, Nt, Nf
    Yimag = np.transpose(Yimag, (1, 2, 0, 4, 3))
    Yreal = np.transpose(Yreal, (1, 2, 0, 4, 3))
    shape = Yimag.shape
    Nd, Na, Npol, Nt, Nf = shape
    # # Nd, Nt*Na*Npol, Nf
    # Yimag = Yimag.reshape((Nd, Nt * Na * Npol, Nf))
    # Yreal = Yreal.reshape((Nd, Nt * Na * Npol, Nf))
    tec_conv = -8.4479745e6/freqs
    clock_conv = 2. * np.pi *   freqs
    # Nd, Nt*Na*Npol
    tec_init = np.mean(np.arctan2(Yimag, Yreal) / tec_conv, axis=-1)

    with pm.Model() as model:
        clock_uncert = pm.Exponential('clock_uncert', 1., shape=(Nd, Na,1, Nt), testval=1.)
        tec_uncert = pm.Exponential('tec_uncert', 1. / 10., shape=(Nd, Na, 1, Nt), testval=10.)

        clock = pm.Deterministic('clock', clock_uncert * pm.Normal('clock_', shape=(Nd, Na,1, Nt)))
        tec_ = tec_uncert * pm.Normal('tec_', shape=(Nd, Na,1,Nt))

        tec = pm.Deterministic('tec', tec_ + tec_init)

        phase0 = tec[..., None] * tec_conv + clock[..., None] * clock_conv
        phase_m = pm.Deterministic('phase', phase0 - pm.math.tt.tile(phase0[:, 0:1, :], (1, 62, 1, 1, 1)))

        data_uncert = pm.Exponential('b', 1 / 0.07, testval=0.1, shape=(Nd, Na, 1,1,1))

        data = pm.Laplace('Yimag', mu=pm.math.sin(phase_m), b=data_uncert, observed=Yimag) + pm.Laplace('Yreal',
                                                                                                        mu=pm.math.cos(
                                                                                                            phase_m),
                                                                                                        b=data_uncert,
                                                                                                        observed=Yreal)

        state0 = pm.find_MAP(maxeval=10000)
        clock_perm = (2, 0, 1, 3)
        tec_perm = (2, 0, 1, 3)
    return state0['tec'].transpose(tec_perm), state0['clock'].transpose(clock_perm), None

def infer_tec_and_clock(freqs, Yimag, Yreal, per_dir_clock=False, gain_uncert=0.02, learning_rate=0.2, max_iters=1000,  stop_patience=5, patience_percentage=1e-3, num_replicates=32):
    # Npol, Nd, Na, Nf, Nt -> Nd, Na, Npol, Nt, Nf
    Yimag = np.transpose(Yimag, (1, 2, 0, 4, 3))
    Yreal = np.transpose(Yreal, (1, 2, 0, 4, 3))
    shape = Yimag.shape
    Nd, Na, Npol, Nt, Nf = shape
    # # Nd, Nt*Na*Npol, Nf
    # Yimag = Yimag.reshape((Nd, Nt * Na * Npol, Nf))
    # Yreal = Yreal.reshape((Nd, Nt * Na * Npol, Nf))
    tec_conv = -8.4479745e6/freqs
    clock_conv = 2. * np.pi *   freqs
    # Nd, Nt*Na*Npol
    amplitude = np.sqrt(Yimag**2 + Yreal**2)
    tec_init = np.mean(np.arctan2(Yimag, Yreal) / tec_conv, axis=-1)

    
    with tf.Session(graph=tf.Graph()) as sess:
        amp_pl = tf.placeholder(float_type, shape=(Nd,Na,Npol,Nt,Nf), name='amp')
        tec_init_pl = tf.placeholder(float_type, shape=(Nd,Na,Npol,Nt), name='tec_init')
        clock_init_pl = tf.placeholder(float_type, shape=(Nd,Na,Npol,Nt), name='clock_init')

        Yimag_pl = tf.placeholder(float_type, shape=Yimag.shape, name='Yimag')
        Yreal_pl = tf.placeholder(float_type, shape=Yreal.shape, name='Yreal')

        freqs = tf.constant(freqs, float_type, name='freqs')
        data_uncert = tf.constant(gain_uncert, float_type, name='data_uncert')
        tec_conv = tf.constant(-8.4479745e6, float_type) * tf.math.reciprocal(freqs)
        clock_conv = tf.constant(2. * np.pi * 1e-9, float_type) * freqs

   
        def neg_log_prob(tec0, tec_, clock0, clock_):
            tec = tec0 + tec_
            clock = clock0 + clock_
   
            phase_m = tec_conv * tec[..., None] + clock_conv * clock[..., None]
            Yimag_m = amp_pl*tf.sin(phase_m)
            Yreal_m = amp_pl*tf.cos(phase_m)

            Yimag_dist = tfp.distributions.Laplace(loc=Yimag_pl, scale=data_uncert[...,None])
            Yreal_dist = tfp.distributions.Laplace(loc=Yreal_pl, scale=data_uncert[...,None])
            diff_clock = clock[...,1:] - clock[...,:-1]
            diff_clock_prior = tfp.distributions.Normal(loc=tf.constant(0.,float_type),scale=tf.constant(0.02,float_type))
            tec_prior = tfp.distributions.Normal(loc=tf.constant(0.,float_type),scale=tf.constant(25.,float_type))
            clock_prior = tfp.distributions.Normal(loc=tf.constant(0.,float_type),scale=tf.constant(2.,float_type))


            likelihood = tf.reduce_mean(Yimag_dist.log_prob(Yimag_m), axis=-1) \
                    + tf.reduce_mean(Yreal_dist.log_prob(Yreal_m), axis=-1) \
                    + clock_prior.log_prob(clock) \
                    + tec_prior.log_prob(tec)
                #+ tf.reduce_mean(diff_clock_prior.log_prob(diff_clock), axis=-1)[...,None] \
            loss = -likelihood
            return loss
   
        init_state = [40. * tf.random.normal((num_replicates, Nd, Na,Npol,Nt), dtype=float_type),
                      2. * tf.random.normal((num_replicates, Nd, Na,Npol,Nt), dtype=float_type),
                      ]
        clock_shape = (Nd, Na, Npol, Nt)
   
        (final_tec, final_clock), loss = adam_stochastic_gradient_descent_with_linesearch_batch(lambda tec, clock: neg_log_prob(tec_init_pl, tec, 0., clock),
                                                                                          init_state,
                                                                                          learning_rate=learning_rate,
                                                                                          iters=max_iters,
                                                                                          stop_patience=stop_patience,
                                                                                          patience_percentage=patience_percentage)

        final_loss = tf.reduce_sum(tf.reshape(neg_log_prob(tec_init_pl, final_tec, 0., final_clock), (-1, Na*Nd*Npol*Nt)),axis=-1)
        argmin = tf.argmin(final_loss)
        final_tec = tf.gather(final_tec,argmin,axis=0)
        final_clock = tf.gather(final_clock,argmin,axis=0)
            
        tec_shape = (Nd, Na, Npol, Nt)

        clock_perm = (2, 0, 1, 3)
        tec_perm = (2, 0, 1, 3)

       
        (tec, clock), final_loss = sess.run([(final_tec + tec_init_pl, final_clock), final_loss],
                                      feed_dict={Yimag_pl: Yimag.astype(np.float64),
                                                 Yreal_pl: Yreal.astype(np.float64),
                                                 tec_init_pl: tec_init.astype(np.float64),
                                                 amp_pl: amplitude})
        return tec.transpose(tec_perm), clock.transpose(clock_perm),  final_loss

if __name__ == '__main__':

    input_datapack = '/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v6.h5'
    datapack = DataPack(input_datapack)
    screen_directions = get_screen_directions('/home/albert/ftp/image.pybdsm.srl.fits', max_N=None)
    maybe_create_posterior_solsets(datapack, 'sol000', posterior_name='posterior', screen_directions=screen_directions, remake_posterior_solsets=False)

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

    Yimag_full = amp_raw*np.sin(phase_raw)
    Yreal_full = amp_raw*np.cos(phase_raw)

    block_size = 1
    save_freq = 1

    save_tec = []
    save_clock = []
    save_phase = []

    for i in range(0, Nt, block_size):
        time_slice = slice(i, min(i+block_size, Nt), 1)
        Yimag = Yimag_full[..., time_slice]
        Yreal = Yreal_full[..., time_slice]


        tec, clock,  loss = infer_tec_and_clock(freqs, Yimag, Yreal,
                                               gain_uncert=0.07,
                                               learning_rate=0.01,
                                               max_iters=4000,
                                               stop_patience=50,
                                               patience_percentage=1e-6,
                                               num_replicates=100)

        logging.info("Iteration {}: final loss min: {}, max: {}, median: {}".format(i, np.min(loss), np.max(loss), np.median(loss)))
        

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
                          time=slice(0,min(i+block_size, Nt),1),
                          freq=slice(None, None, 1),
                          pol=slice(0, 1, 1))

            datapack_save = DataPack(input_datapack, readonly=False)
            datapack_save.current_solset = 'data_posterior'
            # Npol, Nd, Na, Nf, Nt
            datapack_save.select(**select)
            datapack_save.phase = _save_phase
            datapack_save.tec = _save_tec
            datapack_save.clock = _save_clock[:,:,:,:]

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
    datapack_save.clock = _save_clock[:,:,:,:]
