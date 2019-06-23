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
from scipy.optimize import brute, fmin
from bayes_filter.coord_transforms import ITRSToENUWithReferences


import pymc3 as pm
from dask.multiprocessing import get
from functools import partial

import numpy as np


def build_loss(Yreal, Yimag, freqs, gain_uncert=0.02, tec_mean_prior=0., tec_uncert_prior=100.,S=20):
    """
    This function builds the loss function.
    Simple use case:
    # loop over data
    loss_fn = build_loss(Yreal, Yimag, freqs, gain_uncert=0.02, tec_mean_prior=0., tec_uncert_prior=0.1,S=20)
    #brute force
    tec_mean, tec_uncert = brute(loss_fn, (slice(-200, 200,1.), slice(np.log(0.01), np.log(10.), 1.), finish=fmin)
    #The results are Bayesian estimates of tec mean and uncert.

    :param Yreal: np.array shape [Nf]
        The real data (including amplitude)
    :param Yimag: np.array shape [Nf]
        The imag data (including amplitude)
    :param freqs: np.array shape [Nf]
        The freqs in Hz
    :param gain_uncert: float
        The uncertainty of gains.
    :param tec_mean_prior: float
        the prior mean for tec in mTECU
    :param tec_uncert_prior: float
        the prior tec uncert in mTECU
    :param S: int
        Number of hermite terms for Guass-Hermite quadrature
    :return: callable function of the form
        func(params) where params is a tuple or list with:
            params[0] is tec_mean in mTECU
            params[1] is log_tec_uncert in log[mTECU]
        The return of the func is a scalar loss to be minimised.
    """

    x, w = np.polynomial.hermite.hermgauss(S)
    w / np.pi
    tec_conv = -8.4479745e6/freqs

    amp = np.sqrt(Yreal**2 + Yimag**2)

    def loss_func(params):
        tec_mean, log_tec_uncert = params
        tec_uncert = np.exp(log_tec_uncert)
        tec = tec_mean + np.sqrt(2.) * tec_uncert * x
        phase = tec[:, None] * tec_conv
        Yreal_m = amp * np.cos(phase)
        Yimag_m = amp * np.sin(phase)
        log_prob = -np.mean(np.abs(Yreal - Yreal_m) +
                            np.abs(Yimag - Yimag_m), axis=-1) / gain_uncert - np.log(2. * gain_uncert)
        var_exp = np.sum(log_prob * w)
        # Get KL
        q_var = np.square(tec_uncert)
        trace = q_var/tec_uncert_prior**2
        mahalanobis = (tec_mean - tec_mean_prior)**2 /tec_uncert_prior**2
        constant = -1.
        logdet_qcov = np.log(tec_uncert_prior**2 / q_var)
        twoKL = mahalanobis + constant + logdet_qcov + trace
        tec_prior_KL = 0.5 * twoKL
        loss = np.negative(var_exp - tec_prior_KL)
        return loss

    return loss_func


def infer_tec_and_clock(freqs, Yimag, Yreal, gain_uncert=0.02, S=20, ref_dir=14):
    tec_conv = -8.4479745e6/freqs
    Npol, Nd, Na, Nf, Nt = Yimag.shape

    phase = np.arctan2(Yimag, Yreal)
    phase_di = phase[:, ref_dir:ref_dir+1, ...] if ref_dir is not None else np.mean(phase, axis=1, keepdims=True)
    phase_dd = phase - phase_di
    amp = np.sqrt(np.square(Yimag) + np.square(Yreal))
    Yreal_data = amp*np.cos(phase_dd)
    Yimag_data = amp*np.sin(phase_dd)

    def solve_all_time(ant,dir):
        res = []
        tec_mean_prior = 0.
        tec_uncert_prior = 100.
        for time in range(Nt):
            loss_fn = build_loss(Yreal_data[0, dir, ant, :, time], Yimag_data[0, dir, ant, :, time], freqs,
                                 gain_uncert=gain_uncert[dir, ant], tec_mean_prior=tec_mean_prior, tec_uncert_prior=tec_uncert_prior, S=20)

            tec_mean, log_tec_uncert = brute(loss_fn, (slice(-200., 200., 1.), slice(np.log(0.01), np.log(10.), 1.)),
                                             finish=fmin)
            tec_uncert = np.exp(log_tec_uncert)
            tec_mean_prior = tec_mean
            tec_uncert_prior = np.sqrt(tec_uncert**2 + 10.**2)
            logging.info("Soltuion ant: {} dir: {} time: {} tec: {} +- {}".format(ant, dir, time, tec_mean, tec_uncert))
            res.append([tec_mean, tec_uncert])
        return np.array(res)

    logging.info("Constructing the dask of size: {}".format((Nd,Na,Nt)))
    dsk = {}
    get_idx = []
    c = 0
    for d in range(Nd):
        for a in range(Na):
            dsk[str(c)] = (solve_all_time, a, d)
            get_idx.append(str(c))
            c += 1
    logging.info("Running the dask on all cores")

    results = get(dsk, get_idx, num_workers=None)
    logging.info("Completed the dask")
    # print(np.array([p[0] for p in results]))
    tec_mean = np.stack([p[:,0] for p in results], axis=0).reshape((Npol, Nd, Na, Nt))
    tec_std = np.stack([p[:,1] for p in results], axis=0).reshape((Npol, Nd, Na, Nt))
    phase_mean = tec_mean[...,None,:]*tec_conv[:,None] + phase_di
    phase_std = tec_std[..., None, :]*tec_conv[:, None]
    logging.info("Returning results")

    return tec_mean, tec_std, phase_mean, phase_std


def solve_tec_screen(tec_mean, tec_uncert, X, reference_location, reference_direction):
    """
    Solves the tec screen problem using the DTEC kernel model.

    :param tec_mean: tf.Tensor
        Mean of DDTEC shape [Nd, Na] (mTECU)
    :param tec_uncert: tf.Tensor
        Uncertainty of DDTEC shape [Nd, Na] (mTECU)
    :param X: tf.Tensor
        Coordinates shape [Nd, Na, 6] with elements {time, kx, ky, kz, x, y, z}
    :return: tuple of tf.Tensor

    """
    ITRSToENUWithReferences()


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
                  time=slice(None, None, 1),
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

    tec_mean, tec_std, phase_mean, phase_std = infer_tec_and_clock(freqs, Yimag_full, Yreal_full, gain_uncert=gain_uncert)

    logging.info("Storing results")
    datapack_save = DataPack(input_datapack, readonly=False)
    datapack_save.current_solset = 'data_posterior'
    # Npol, Nd, Na, Nf, Nt
    datapack_save.select(**select)
    datapack_save.phase = phase_mean
    datapack_save.weights_phase = phase_std
    datapack_save.tec = tec_mean
    datapack_save.weights_tec = tec_std
    logging.info("Stored results. Done")