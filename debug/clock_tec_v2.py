###
# TF based solver clock and tec or just tec
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bayes_filter.datapack import DataPack
from bayes_filter import logging
from bayes_filter.misc import maybe_create_posterior_solsets, get_screen_directions
import pylab as plt
from bayes_filter import float_type, logging
from scipy.optimize import brute, fmin
from bayes_filter.coord_transforms import ITRSToENUWithReferences

from concurrent.futures import ProcessPoolExecutor
from dask.multiprocessing import get
from functools import partial

from scipy.linalg import cho_solve
import numpy as np
import sys

class TecSolveLoss(object):
    """
    This class builds the loss function.
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
    def __init__(self,Yreal, Yimag, freqs, gain_uncert=0.02, tec_mean_prior=0., tec_uncert_prior=100.,S=20):
        self.x, self.w = np.polynomial.hermite.hermgauss(S)
        self.w /= np.pi
        self.tec_conv = -8.4479745e6/freqs

        self.amp = np.sqrt(Yreal**2 + Yimag**2)
        self.Yreal = Yreal
        self.Yimag = Yimag
        self.gain_uncert = gain_uncert
        self.tec_mean_prior = tec_mean_prior
        self.tec_uncert_prior = tec_uncert_prior
    
    def loss_func(self, params):
        tec_mean, log_tec_uncert = params
        tec_uncert = np.exp(log_tec_uncert)
        tec = tec_mean + np.sqrt(2.) * tec_uncert * self.x
        phase = tec[:, None] * self.tec_conv
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        log_prob = -np.mean(np.abs(self.Yreal - Yreal_m) +
                            np.abs(self.Yimag - Yimag_m), axis=-1) / self.gain_uncert - np.log(2. * self.gain_uncert)
        var_exp = np.sum(log_prob * self.w)
        # Get KL
        q_var = np.square(tec_uncert)
        trace = q_var/self.tec_uncert_prior**2
        mahalanobis = (tec_mean - self.tec_mean_prior)**2 /self.tec_uncert_prior**2
        constant = -1.
        logdet_qcov = np.log(self.tec_uncert_prior**2 / q_var)
        twoKL = mahalanobis + constant + logdet_qcov + trace
        tec_prior_KL = 0.5 * twoKL
        loss = np.negative(var_exp - tec_prior_KL)
        return loss

class TecSolve(object):
    def __init__(self, freqs, Yimag, Yreal, gain_uncert=0.02, S=20, ref_dir=14):
        logging.info('Inferred TEC with ref dir: {}'.format(ref_dir))
        self.tec_conv = -8.4479745e6/freqs
        self.shape = Yimag.shape
        self.freqs = freqs
        self.phase = np.arctan2(Yimag, Yreal)
        self.phase_di = self.phase[:, ref_dir:ref_dir+1, ...]
        self.phase_dd = self.phase - self.phase_di
        self.amp = np.sqrt(np.square(Yimag) + np.square(Yreal))
        self.Yreal_data = self.amp*np.cos(self.phase_dd)
        self.Yimag_data = self.amp*np.sin(self.phase_dd)
        self.gain_uncert = gain_uncert
        
    def solve_all_time(self, args):
        ant,dir = args
        res = []
        tec_mean_prior = 0.
        tec_uncert_prior = 55.
#         logging.info("Inferring TEC for ant: {} dir: {} ref: {}".format(ant, dir, ref_dir))
        for time in range(Nt):
            Loss = TecSolveLoss(self.Yreal_data[0, dir, ant, :, time], self.Yimag_data[0, dir, ant, :, time], self.freqs,
                                 gain_uncert=self.gain_uncert[dir, ant], tec_mean_prior=tec_mean_prior, tec_uncert_prior=tec_uncert_prior, S=20)

            tec_mean, log_tec_uncert = brute(Loss.loss_func, (slice(-200., 200., 5.), 
                                                       slice(np.log(0.5), np.log(5.), 1.)),
                                             finish=fmin)
            tec_uncert = np.exp(log_tec_uncert)
            tec_mean_prior = tec_mean
            tec_uncert_prior = np.sqrt(tec_uncert**2 + 50.**2)
#             logging.info("Soltuion ant: {} dir: {} time: {} tec: {} +- {}".format(ant, dir, time, tec_mean, tec_uncert))
            res.append([tec_mean, tec_uncert])
#         logging.info("Finished inferring TEC for ant: {} dir: {} ref: {}".format(ant, dir, ref_dir))
        return np.array(res)

    def run(self):
        Npol, Nd, Na, Nf, Nt = self.shape
        logging.info("Constructing the dask of size: {}".format((Nd,Na,Nt)))
        dsk = {}
        get_idx = []
        args = []
        c = 0
        for d in range(Nd):
            for a in range(Na):
#                 dsk[str(c)] = (self.solve_all_time, a, d)
                args.append((a,d))
                get_idx.append(str(c))
                c += 1
        logging.info("Running the dask on all cores")
        
        with ProcessPoolExecutor(max_workers=1) as exe:
            results = list(exe.map(self.solve_all_time, args))

#         results = get(dsk, get_idx, num_workers=None)
        logging.info("Completed the dask")
        # print(np.array([p[0] for p in results]))
        
        tec_mean = np.stack([p[:,0] for p in results], axis=0).reshape((Npol, Nd, Na, Nt))
        tec_std = np.stack([p[:,1] for p in results], axis=0).reshape((Npol, Nd, Na, Nt))
        phase_mean = tec_mean[...,None,:]*self.tec_conv[:,None] + self.phase_di
        phase_std = tec_std[..., None, :]*self.tec_conv[:, None]
        logging.info("Returning results")

        return tec_mean, tec_std, phase_mean, phase_std

class ResidualSmoothLoss(object):
    def __init__(self, phase_res, freqs):
        """
        This function builds the loss function.
        Simple use case:
        # loop over data
        loss_fn = build_loss(Yreal, Yimag, freqs, gain_uncert=0.02, tec_mean_prior=0., tec_uncert_prior=0.1,S=20)
        #brute force
        tec_mean, tec_uncert = brute(loss_fn, (slice(-200, 200,1.), slice(np.log(0.01), np.log(10.), 1.), finish=fmin)
        #The results are Bayesian estimates of tec mean and uncert.

        :param phase_res: np.array shape [Nf]
            The phase residual data 
        :param freqs: np.array shape [Nf]
            The freqs in Hz
        :return: callable function of the form
            func(params) where params is a tuple or list with:
                params[0] is log_phase_noise in radians
                params[1] is log_freq_lengthscales in MHz
                params[2] is log_sigma in radians
                params[3] is mean
            The return of the func is a scalar loss to be minimised.
        """
        self.freqs = freqs/1e6
        self.emp_mean = phase_res.mean()
        self.phase_res = phase_res - self.emp_mean
        dfreqs = freqs[:, None]  - freqs[None, :]
        self.neg_chi = -0.5*np.square(dfreqs)
        self.I = np.eye(freqs.size)
        self.Nf = freqs.size

    def loss_func(self, params):
        phase_noise, sigma, freq_lengthscale = np.exp(params[:3])
        freq_lengthscale += 10.
        mean = params[3]
        K = sigma**2 * np.exp(self.neg_chi/freq_lengthscale**2)
        Kf = K + phase_noise**2 * self.I
        L = np.linalg.cholesky(Kf)
        dy = self.phase_res - mean
        A = cho_solve((L, True), dy, overwrite_b=True )
        maha = -0.5*np.square(A)
        com = -0.5*self.Nf*np.log(2*np.pi) - np.sum(np.log(np.diag(L)))
        marginal_log = np.sum(maha + com)
        return -marginal_log

    def smooth_func(self, params):
        #
        phase_noise, sigma, freq_lengthscale = np.exp(params[:3])
        freq_lengthscale += 10.
        mean = params[3]
        K = sigma**2 * np.exp(self.neg_chi/freq_lengthscale**2)
        Kf = K + phase_noise**2 * self.I
        L = np.linalg.cholesky(Kf)
        A = cho_solve((L, True), K, overwrite_b=True)
        dy = self.phase_res - mean
        post_mean = A.T.dot(dy) + mean + self.emp_mean
        post_var = sigma**2 - np.sum(np.square(A), axis=0)
        #TODO: fix variance!?
        return post_mean, np.sqrt(np.abs(post_var))


class ResidualSmooth(object):
    def __init__(self, freqs, phase_res):
        logging.info("Smoothing the phase residuals optimally, minimum bandfilter of 10MHz.")
    
        Npol, Nd, Na, Nf, Nt = phase_res.shape
        self.phase_res = phase_res
        self.freqs = freqs

    def solve_all_time(self, args):
        ant,dir = args
        mean_res = []
        uncert_res = []
#             logging.info("Starting residual smooth for ant: {} dir: {}".format(ant, dir))
        for time in range(Nt):
            LossAndSmooth = ResidualSmoothLoss(self.phase_res[0,dir,ant,:,time], self.freqs)
            params = brute(LossAndSmooth.loss_func, (
                slice(np.log(0.01), np.log(0.1), 1.), 
                slice(np.log(0.01), np.log(0.1), 1.),
                slice(np.log(5.), np.log(25.), 1.), 
                slice(-0.1, 0.1, 0.05)
                ),
                           finish=fmin)
            smoothed_phase_mean, smoothed_phase_uncert = LossAndSmooth.smooth_func(params)
            mean_res.append(smoothed_phase_mean)
            uncert_res.append(smoothed_phase_uncert)
#             logging.info("Finsihed residual smooth for ant: {} dir: {}".format(ant, dir))
        #Nf, Nt
        return np.stack(mean_res, axis=1), np.stack(uncert_res, axis=1)
    
    def run(self):

        logging.info("Constructing the dask of size: {}".format((Nd,Na,Nt)))
        dsk = {}
        get_idx = []
        args = []
        c = 0
        for d in range(Nd):
            for a in range(Na):
#                 dsk[str(c)] = (self.solve_all_time, a, d)
                args.append((a,d))
                get_idx.append(str(c))
                c += 1
        logging.info("Running the dask on all cores")
        
        with ProcessPoolExecutor(max_workers=1) as exe:
            results = list(exe.map(self.solve_all_time, args))

#         results = get(dsk, get_idx, num_workers=None)
        logging.info("Completed the dask")
        # print(np.array([p[0] for p in results]))
        smooth_residual_mean = np.stack([r[0] for r in results], axis=0).reshape((Npol, Nd, Na, Nf, Nt))
        smooth_residual_uncert = np.stack([r[1] for r in results], axis=0).reshape((Npol, Nd, Na, Nf, Nt))
        logging.info("Returning results")

        return smooth_residual_mean, smooth_residual_uncert
    


if __name__ == '__main__':

    input_datapack = '/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v5.h5'
    datapack = DataPack(input_datapack)
    screen_directions = get_screen_directions('/home/albert/ftp/image.pybdsm.srl.fits', max_N=None)
    maybe_create_posterior_solsets(datapack, 'sol000', posterior_name='posterior', screen_directions=screen_directions,
                                   remake_posterior_solsets=False)

    datapack.current_solset = 'sol000'
    axes = datapack.axes_phase
    _, times = datapack.get_times(axes['time'])
    Nt = len(times)
    
    if len(sys.argv) != 4:
        raise ValueError("{} ant from_time to_time".format(sys.argv[0]))

    ant = int(sys.argv[1])
    from_time, to_time = [int(l) for l in sys.argv[1:3]]
    select = dict(dir=slice(None, None, 1),
                  ant=slice(ant, ant+1, 1),
                  time=slice(from_time, to_time, 1),
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
    
    Npol, Nd, Na, Nf, Nt = phase_raw.shape

    Yimag_full = amp_raw * np.sin(phase_raw)
    Yreal_full = amp_raw * np.cos(phase_raw)
    # Nd,Na
    gain_uncert = np.maximum(
        0.25 * np.mean(np.abs(np.diff(Yimag_full, axis=-1)) + np.abs(np.diff(Yreal_full, axis=-1)), axis=-1).mean(
            -1).mean(0), 0.02)
    
#     gain_uncert = 0.07*np.ones((Nd,Na))


    
    tec_means = []
    tec_stds = []
    phase_means = []
    phase_stds = []
    for d in range(Nd):
        Solver = TecSolve(freqs, Yimag_full, Yreal_full, gain_uncert=gain_uncert, ref_dir=d)
        tec_mean, tec_std, phase_mean, phase_std = Solver.run()
        tec_means.append(tec_mean)
        tec_stds.append(tec_std)
        phase_means.append(phase_mean)
        phase_stds.append(phase_std)
        
    logging.info("Solving tec per direction problem")
        
    tec_dir_mean = []
    tec_dir_uncert = []
    lhs = []
    for d in range(Nd):
        A = np.eye(Nd)
        A[d, :] = 0.
        A[:, d] = -1.
        lhs.append(A)
    lhs = np.concatenate(lhs, axis=0)

    for a in range(Na):
        for t in range(Nt):
            rhs = np.concatenate([T[0, :, a, t] for T in tec_means])
            rhs_std = np.concatenate([T[0, :, a, t] for T in tec_stds])
            tec_dir_mean.append(np.linalg.lstsq(lhs, rhs)[0])
            u = np.linalg.lstsq(lhs, np.diag(rhs_std))[0]
            tec_dir_uncert.append(np.sqrt(np.diag(u.dot(u.T))))

    tec_dir_mean = np.stack(tec_dir_mean, axis=1).reshape((Nd, Na, Nt))[None, ...]
    tec_dir_uncert = np.stack(tec_dir_uncert, axis=1).reshape((Nd,Na,Nt))[None, ...]

    tec_conv = -8.4479745e6/freqs

    phase_dd_mean = tec_dir_mean[...,None, :]*tec_conv[:,None]
    def w(p):
        return np.angle(np.exp(1j*p))
    logging.info("Getting phase residual per direction.")
    phase_res = w(w(phase_raw) - w(phase_dd_mean))
    phase_dd_uncert = tec_dir_uncert[...,None, :]*tec_conv[:,None]
    
        
    
    
    Smoother = ResidualSmooth(freqs, phase_res)
    smooth_residual_mean, smooth_residual_uncert = Smoother.run()
    
    logging.info("Constructing final phase")
    
    final_phase_mean = phase_dd_mean + smooth_residual_mean
    final_phase_uncert = np.sqrt(np.square(phase_dd_uncert) + np.square(smooth_residual_uncert))


    logging.info("Storing results")
    datapack_save = DataPack(input_datapack, readonly=False)
    datapack_save.current_solset = 'data_posterior'
    # Npol, Nd, Na, Nf, Nt
    datapack_save.select(**select)
    datapack_save.phase = final_phase_mean
    datapack_save.weights_phase = final_phase_uncert # TODO: Add the uncert from smoothing
    datapack_save.tec = tec_dir_mean
    datapack_save.weights_tec = tec_dir_uncert
    logging.info("Stored results. Done")
