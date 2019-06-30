###
# TF based solver clock and tec or just tec
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.parallel_for.gradients import jacobian
import numpy as np
from bayes_filter.datapack import DataPack
from bayes_filter.sgd import adam_stochastic_gradient_descent_with_linesearch
from bayes_filter import logging, float_type
from bayes_filter.misc import maybe_create_posterior_solsets, get_screen_directions, safe_cholesky
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


def tf_slice(start, stop, step):
    num = tf.cast((stop - start)/step, tf.int32) + 1
    return tf.cast(tf.linspace(start, stop, num), float_type)

def tf_brute(fn, ranges, finish=None, vectorized=True):
    M = len(ranges)
    #N, M
    grid = tf.stack([tf.reshape(t, (-1,)) for t in tf.meshgrid(*ranges, indexing='ij')], axis=1)
    if vectorized:
        result = fn(grid)
    else:
        result = tf.map_fn(fn, grid)
    argmin = tf.argmin(result)
    best_res = result[argmin]
    best_point = grid[argmin, :]
    best_point.set_shape([M])
    best_point = tf.unstack(best_point)
    #finish
    if finish is None:
        return best_point
    if not callable(finish):
        raise ValueError("finish must be callable")
    return finish(fn, best_point)

def build_brute_finish(radius, steps, finish=None, vectorized=True):
    def recursion(fn, points):
        points = [tf.convert_to_tensor(p, float_type) for p in points]
        if len(points) != len(radius):
            raise ValueError("Radius length must match init point length")
        if len(points) != len(steps):
            raise ValueError("Steps length must match init point length")
        ranges = [tf_slice(p-r, p+r, s) for p,r,s in zip(points, radius, steps)]
        return tf_brute(fn, ranges, finish=finish, vectorized=vectorized)
    return recursion

def helper_brute_recursion(levels, init_radius, shrinkage=0.1, vectorized=True):
    finishers = [None]
    #radius_0*shrinkage^(t-1), radius_0*shrinkage^t
    if not isinstance(shrinkage, (list,tuple)):
        shrinkage = [shrinkage for _ in init_radius]

    for l in range(1,levels+1, 1)[::-1]:
        solve_fn = build_brute_finish([tf.convert_to_tensor(2.*r*s**(l-1), float_type) for r,s in zip(init_radius, shrinkage)],
                                      [tf.convert_to_tensor(2.*r*s**l, float_type) for r,s in zip(init_radius, shrinkage)],
                                      finish=finishers[-1],
                                      vectorized=vectorized)
        finishers.append(solve_fn)
    return finishers[-1]

def help_brute_bfgs(radius, steps, vectorized=True):
    def bfgs(fn, points):
        def val_and_grad(points):
            val = fn(points)
            g = tf.gradients(val, [points])[0]
            return val, g
        M = len(points)
        bfgs_res = tfp.optimizer.bfgs_minimize(val_and_grad, tf.stack(points, axis=0)[None, :])
        best_point = bfgs_res.position[0,:]
        best_point.set_shape([M])
        best_point = tf.unstack(best_point)
        return best_point

    return build_brute_finish(radius, steps, finish=bfgs, vectorized=vectorized)

def help_brute_adam(radius, steps, vectorized=True, **kwargs):
    def adam(fn, points):
        best_point, _ = adam_stochastic_gradient_descent_with_linesearch(lambda *points: fn(tf.stack(points, axis=0)[None, :]),
                                                                     points,
                                                                     iters=kwargs.get('iters', 100),
                                                                     learning_rate=kwargs.get('learning_rate', 0.1),
                                                                     stop_patience=kwargs.get('stop_patience', 5),
                                                                     patience_percentage=kwargs.get('patience_percentage', 1e-4),
                                                                     log_step=kwargs.get('log_step', 0.05))

        # best_point.set_shape([len(points)])
        # best_point = tf.unstack(best_point)
        return best_point

    return build_brute_finish(radius, steps, finish=adam, vectorized=vectorized)

def help_brute_newton(radius, steps, num_steps=3, vectorized=True, **kwargs):
    def newton(fn, points):
        next_point = tf.stack(points, axis=0)
        for i in range(num_steps):
            loss = fn(next_point[None, :])
            gradient = tf.gradients(loss, [next_point])[0]
            hessian = jacobian(gradient, next_point)
            hessian += tf.linalg.diag(tf.reduce_mean(0.1*tf.linalg.diag_part(hessian))*tf.ones((len(points),), float_type))
            grad = tf.linalg.lstsq(hessian, gradient[:, None], fast=False)[:, 0]
            next_point = next_point - 0.5*grad
            next_point.set_shape([len(points)])
        return tf.unstack(next_point)

    return build_brute_finish(radius, steps, finish=newton, vectorized=vectorized)


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
    def __init__(self,Yreal, Yimag, freqs, gain_uncert=0.02, tec_mean_prior=0., tec_uncert_prior=100.,S=20, log_uncert=True):
        x, w = np.polynomial.hermite.hermgauss(S)
        w /= np.pi
        self.x = tf.convert_to_tensor(x, float_type)
        self.w = tf.convert_to_tensor(w, float_type)
        self.log_uncert = log_uncert

        self.tec_conv = -8.4479745e6/freqs
        # Nf
        self.amp = tf.math.sqrt(tf.math.square(Yreal) + tf.math.square(Yimag))
        self.Yreal = Yreal
        self.Yimag = Yimag
        # scalar
        self.gain_uncert = gain_uncert
        self.tec_mean_prior = tec_mean_prior
        self.tec_uncert_prior = tec_uncert_prior
    
    def loss_func(self, params):
        """
        VI loss
        :param params: tf.Tensor
            shape [B, D]
        :return: tf.Tensor
            shape [B] The loss
        """
        # B
        tec_mean, log_tec_uncert = params[:, 0], params[:, 1]
        if self.log_uncert:
            tec_uncert = tf.math.exp(log_tec_uncert)
        else:
            tec_uncert = log_tec_uncert
        # B, S
        tec = tec_mean[:, None] + np.sqrt(2.) * tec_uncert[:, None] * self.x
        # B, S, Nf
        phase = tec[:, :, None] * self.tec_conv
        Yreal_m = self.amp * tf.math.cos(phase)
        Yimag_m = self.amp * tf.math.sin(phase)
        # B, S
        log_prob = -tf.reduce_mean(tf.math.abs(self.Yreal - Yreal_m) +
                            tf.math.abs(self.Yimag - Yimag_m), axis=-1) / self.gain_uncert - np.log(2.) - tf.math.log(self.gain_uncert)
        # B
        var_exp = tf.reduce_sum(log_prob * self.w, axis=1)
        # Get KL
        # B
        q_var = tf.math.square(tec_uncert)
        tec_var_prior = tf.math.square(self.tec_uncert_prior)
        trace = q_var/tec_var_prior
        mahalanobis = tf.math.square(tec_mean - self.tec_mean_prior) /tec_var_prior
        constant = -1.
        logdet_qcov = tf.math.log(tec_var_prior / q_var)
        twoKL = mahalanobis + constant + logdet_qcov + trace
        tec_prior_KL = 0.5 * twoKL
        loss = tf.math.negative(var_exp - tec_prior_KL)
        # B
        return loss

class TecSolve(object):
    def __init__(self, freqs, Yimag, Yreal, gain_uncert, S=20, ref_dir=14):
        # Nf
        self.tec_conv = -8.4479745e6/freqs
        self.freqs = freqs
        self.ref_dir = ref_dir
        # Npol, Nd, Na, Nf, Nt
        self.phase = tf.math.atan2(Yimag, Yreal)
        self.phase_di = self.phase[:, ref_dir:ref_dir+1, ...]
        self.phase_dd = self.phase - self.phase_di
        self.amp = tf.math.sqrt(tf.math.square(Yimag) + tf.math.square(Yreal))
        self.Yreal_data = self.amp*tf.math.cos(self.phase_dd)
        self.Yimag_data = self.amp*tf.math.sin(self.phase_dd)
        # Nd, Na
        self.gain_uncert = gain_uncert
        self.S = S

    def solve_all_time(self, args):
        dir, ant = args

        tec_mean_prior = 0.
        tec_uncert_prior = 55.
        tec_mean_ta = tf.TensorArray(float_type, size=tf.shape(self.Yimag_data)[-1], element_shape=())
        tec_uncert_ta = tf.TensorArray(float_type, size=tf.shape(self.Yimag_data)[-1], element_shape=())
        def body(time, tec_mean_prior, tec_uncert_prior, tec_mean_ta, tec_uncert_ta):
            loss = TecSolveLoss(self.Yreal_data[0, dir, ant, :, time], self.Yimag_data[0, dir, ant, :, time],
                                self.freqs,
                                gain_uncert=self.gain_uncert[dir, ant], tec_mean_prior=tec_mean_prior,
                                tec_uncert_prior=tec_uncert_prior, S=self.S,
                                log_uncert=True)
            # brute_solver = help_brute_adam([200., 2.], [5., 4./10.], vectorized=True, iters=10)
            # brute_solver = help_brute_bfgs([200., 2.], [5., 0.2], vectorized=True)
            brute_solver = helper_brute_recursion(2, [200., 2.], shrinkage=[1/80., 1/5.], vectorized=True)
            # brute_solver = help_brute_newton([200., 2.], [5., 0.2], num_steps=3, vectorized=True)
            params = brute_solver(loss.loss_func, [0., np.log(1.)])
            tec_mean, tec_uncert = params[0], tf.math.exp(params[1])
            next_tec_mean_prior = tec_mean
            next_tec_uncert_prior = tf.math.sqrt(tf.math.square(tec_uncert) + 50.**2)
            return [time + 1, next_tec_mean_prior, next_tec_uncert_prior, tec_mean_ta.write(time, tec_mean), tec_uncert_ta.write(time, tec_uncert)]

        def cond(time , *args):
            return time < tf.shape(self.Yimag_data)[-1]

        _, _, _, tec_mean_ta, tec_uncert_ta = tf.while_loop(cond,
                                                            body,
                                                            [tf.constant(0),
                                                             tf.constant(tec_mean_prior, float_type),
                                                             tf.constant(tec_uncert_prior, float_type),
                                                             tec_mean_ta,
                                                             tec_uncert_ta],
                                                            back_prop=False)

        tec_mean = tec_mean_ta.stack()
        tec_uncert = tec_uncert_ta.stack()
        return [tec_mean, tec_uncert]

    def run(self, parallel_iterations = 10):
        shape = tf.shape(self.Yimag_data)
        Npol, Nd, Na, Nf, Nt = shape[0], shape[1], shape[2], shape[3], shape[4]
        grid = [tf.reshape(t, (-1,)) for t in tf.meshgrid(tf.range(Nd), tf.range(Na), indexing='ij')]
        with tf.control_dependencies([tf.print('Solving tec with ref dir', self.ref_dir)]):
            # Nd*Na, Nt
            tec_mean, tec_uncert = tf.map_fn(self.solve_all_time, grid, parallel_iterations=parallel_iterations, back_prop=False, dtype=[float_type, float_type])
        tec_mean = tf.reshape(tec_mean, (Npol, Nd, Na, Nt))
        tec_uncert = tf.reshape(tec_uncert, (Npol, Nd, Na, Nt))

        return tec_mean, tec_uncert

class TecSystemSolve(object):
    def __init__(self,freqs, Yimag, Yreal, gain_uncert, S=20, reference_dirs = None, per_ref_parallel_iterations=10):
        self.freqs = freqs
        self.Yimag = Yimag
        self.Yreal = Yreal
        self.gain_uncert = gain_uncert
        self.S = S
        self.reference_dirs = reference_dirs
        self.per_ref_parallel_iterations = per_ref_parallel_iterations

    def solve_per_ref(self, ref_dir):
        tec_solver = TecSolve(self.freqs, self.Yimag, self.Yreal, self.gain_uncert, S = self.S, ref_dir = ref_dir)
        #Npol, Nd, Na, Nt
        tec_mean, tec_uncert = tec_solver.run(parallel_iterations=self.per_ref_parallel_iterations)
        return [tec_mean, tec_uncert]

    def run(self, lstsq_parallel_iterations=10):
        shape = tf.shape(self.Yimag)
        Npol, Nd, Na, Nf, Nt = shape[0], shape[1], shape[2], shape[3], shape[4]
        if self.reference_dirs is None:
            ref_grid = tf.range(Nd)
        else:
            ref_grid = tf.constant(self.reference_dirs, tf.int32)
        # Nd, Npol, Nd, Na, Nt
        tec_mean, tec_uncert = tf.map_fn(self.solve_per_ref, ref_grid, parallel_iterations=1, dtype=[float_type, float_type])

        def _construct_lhs(ref_grid, Nd):
            ref_grid = ref_grid.numpy()
            Nd = Nd.numpy()
            lhs = []
            for d in ref_grid:
                A = np.eye(Nd)
                A[d, :] = 0.
                A[:, d] = -1.
                lhs.append(A)
            lhs = np.concatenate(lhs, axis=0).astype(np.float64)
            return [lhs]

        #Nd*Nd, Nd
        lhs = tf.py_function(_construct_lhs, [ref_grid, Nd], [float_type])[0]
        # todo: all in one go
        def _solve_system(args):
            ant, time = args
            #Nd**2
            rhs_mean = tf.concat(tf.unstack(tec_mean[:, 0, :, ant, time]), axis=0)
            rhs_uncert = tf.concat(tf.unstack(tec_uncert[:, 0, :, ant, time]), axis=0)
            tec_solution = tf.linalg.lstsq(lhs, rhs_mean[:, None])[:, 0]
            #Nd**2, Nd**2
            u = tf.linalg.lstsq(lhs, tf.linalg.diag(rhs_uncert))
            tec_uncert_solution = tf.math.sqrt(tf.linalg.diag_part(tf.matmul(u, u, transpose_b=True)))
            # Nd
            return [tec_solution, tec_uncert_solution]


        grid = [tf.reshape(t, (-1,)) for t in tf.meshgrid(tf.range(Na), tf.range(Nt), indexing='ij')]
        with tf.control_dependencies([tf.print('Solving system of tec')]):
            #Na*Nt, Nd
            tec_solution, tec_uncert_solution = tf.map_fn(_solve_system, grid, back_prop=False, parallel_iterations=lstsq_parallel_iterations, dtype=[float_type, float_type])
        tec_solution = tf.reshape(tf.transpose(tec_solution, (1,0)), (Npol, Nd, Na, Nt))
        tec_uncert_solution = tf.reshape(tf.transpose(tec_uncert_solution, (1,0)), (Npol, Nd, Na, Nt))
        return tec_solution, tec_uncert_solution


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
        self.freqs = freqs/10e6
        self.Nf = tf.size(self.freqs)
        ###
        # lstsq mean
        #Nf, 2
        lhs = tf.stack([self.freqs, tf.ones_like(self.freqs)], axis=1)
        #2, 1
        coeffs = tf.linalg.lstsq(lhs, phase_res[:, None], fast=False)
        #Nf
        self.emp_mean = tf.matmul(lhs, coeffs)[:, 0]
        #Nf
        # self.emp_mean = tf.reduce_mean(phase_res)
        #Nf
        self.phase_res = phase_res - self.emp_mean
        # Nf, Nf
        self.neg_chi = -0.5*tf.math.squared_difference(self.freqs[:, None], self.freqs[None, :])
        self.I = tf.linalg.diag(tf.ones((self.Nf,), float_type))


    def loss_func(self, params):
        # B, 3
        exp_params = tf.math.exp(params)
        # B
        phase_noise, sigma, freq_lengthscale = exp_params[:, 0], exp_params[:, 1], exp_params[:, 2]
        freq_lengthscale += 0.13
        # B, Nf, Nf
        K = tf.math.square(sigma)[:, None, None] * tf.math.exp(self.neg_chi/tf.math.square(freq_lengthscale)[:, None, None])
        # B, Nf, Nf
        Kf = K + tf.math.square(phase_noise)[:, None, None] * self.I
        L = tf.linalg.cholesky(Kf)
        #B, Nf
        dy = tf.tile(self.phase_res[None, :], (tf.size(phase_noise), 1))
        # B, Nf
        A = tf.linalg.triangular_solve(L, dy[:, :, None])[:, :, 0]
        # B
        maha = -0.5*tf.reduce_sum(tf.math.square(A), axis=-1)
        # B
        com = -0.5*np.log(2*np.pi)*tf.cast(self.Nf, float_type) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=-1)
        marginal_log = maha + com + tfp.distributions.Normal(loc=tf.constant(0., float_type), scale=tf.constant(0.1, float_type)).log_prob(
            phase_noise) + tfp.distributions.Normal(loc=tf.constant(0., float_type), scale=tf.constant(2.5, float_type)).log_prob(freq_lengthscale)
        return tf.math.negative(marginal_log)

    def smooth_func(self, params):
        #non-vecotrised list of params
        # scalars
        phase_noise, sigma, freq_lengthscale = [tf.math.exp(p) for p in params]
        freq_lengthscale += 0.13
        # with tf.control_dependencies([tf.print('smooth params', phase_noise, sigma, freq_lengthscale)]):
        # Nf, Nf
        K = tf.math.square(sigma) * tf.math.exp(
            self.neg_chi / tf.math.square(freq_lengthscale))
        # Nf, Nf
        Kf = K + tf.math.square(phase_noise) * self.I
        L = tf.linalg.cholesky(Kf)
        # Nf, Nf
        A = tf.linalg.triangular_solve(L, K)
        # Nf
        dy = self.phase_res
        #Nf
        post_mean = tf.matmul(A, dy[:, None], transpose_a=True)[:, 0] + self.emp_mean
        post_var = tf.math.square(sigma) - tf.reduce_sum(tf.math.square(A), axis=0)
        #TODO: fix variance!?
        return post_mean, tf.math.sqrt(tf.math.abs(post_var))


# class ResidualSmoothLoss(object):
#     def __init__(self, phase_res, freqs):
#         """
#         This function builds the loss function.
#         Simple use case:
#         # loop over data
#         loss_fn = build_loss(Yreal, Yimag, freqs, gain_uncert=0.02, tec_mean_prior=0., tec_uncert_prior=0.1,S=20)
#         #brute force
#         tec_mean, tec_uncert = brute(loss_fn, (slice(-200, 200,1.), slice(np.log(0.01), np.log(10.), 1.), finish=fmin)
#         #The results are Bayesian estimates of tec mean and uncert.
#
#         :param phase_res: np.array shape [Nf]
#             The phase residual data
#         :param freqs: np.array shape [Nf]
#             The freqs in Hz
#         :return: callable function of the form
#             func(params) where params is a tuple or list with:
#                 params[0] is log_phase_noise in radians
#                 params[1] is log_freq_lengthscales in MHz
#                 params[2] is log_sigma in radians
#                 params[3] is mean
#             The return of the func is a scalar loss to be minimised.
#         """
#         self.freqs = freqs/1e7
#         self.Nf = tf.size(self.freqs)
#         #Nt
#         self.emp_mean = tf.reduce_mean(phase_res, axis=0)
#         #Nf, Nt
#         self.phase_res = phase_res - self.emp_mean
#         # Nf, Nf
#         self.neg_chi = -0.5*tf.math.squared_difference(self.freqs[:, None], self.freqs[None, :])
#         self.I = tf.linalg.diag(tf.ones((self.Nf,), float_type))
#
#
#     def loss_func(self, params):
#         # B, 3
#         exp_params = tf.math.exp(params)
#         # B
#         phase_noise, sigma, freq_lengthscale = exp_params[:, 0], exp_params[:, 1], exp_params[:, 2]
#         # freq_lengthscale += 0.5
#         # B, Nf, Nf
#         K = tf.math.square(sigma)[:, None, None] * tf.math.exp(self.neg_chi/tf.math.square(freq_lengthscale)[:, None, None])
#         # B, Nf, Nf
#         Kf = K + tf.math.square(phase_noise)[:, None, None] * self.I
#         L = tf.linalg.cholesky(Kf)
#         #B, Nf, Nt
#         dy = tf.tile(self.phase_res[None, :, :], (tf.size(phase_noise), 1, 1))
#         # B, Nf, Nt
#         A = tf.linalg.triangular_solve(L, dy)
#         # B, Nt
#         maha = -0.5*tf.reduce_sum(tf.math.square(A), axis=-2)
#         # B
#         com = -0.5*np.log(2*np.pi)*tf.cast(self.Nf, float_type) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=-1)
#         # B
#         marginal_log = tf.reduce_sum(maha + com[:, None], axis=1)# + tfp.distributions.Normal(loc=tf.constant(1.5, float_type), scale=tf.constant(0.5, float_type)).log_prob(freq_lengthscale) + tfp.distributions.Normal(loc=tf.constant(0.075, float_type), scale=tf.constant(0.025, float_type)).log_prob(phase_noise)
#         return tf.math.negative(marginal_log)
#
#     def smooth_func(self, params):
#         #non-vecotrised list of params
#         # scalars
#         phase_noise, sigma, freq_lengthscale = [tf.math.exp(p) for p in params]
#         # phase_noise = tf.constant(0.07, float_type)
#         # phase_noise = tf.constant(0.75, float_type) * tf.math.sqrt(tf.reduce_mean(tf.math.square(self.phase_res)))
#         # sigma=tf.constant(0.1, float_type)
#         # freq_lengthscale += 0.5
#         with tf.control_dependencies([tf.print('smooth params', phase_noise, sigma, freq_lengthscale)]):
#             # Nf, Nf
#             K = tf.math.square(sigma) * tf.math.exp(
#                 self.neg_chi / tf.math.square(freq_lengthscale))
#         # Nf, Nf
#         Kf = K + tf.math.square(phase_noise) * self.I
#         L = tf.linalg.cholesky(Kf)
#         # Nf, Nf
#         A = tf.linalg.triangular_solve(L, K)
#         # Nf, Nt
#         dy = self.phase_res
#         #Nf, Nt
#         post_mean = tf.matmul(A, dy, transpose_a=True) + self.emp_mean
#         post_var = tf.math.square(sigma) - tf.linalg.diag_part(tf.matmul(A, A, transpose_b=True))
#         post_var = tf.tile(post_var[:, None], (1, tf.shape(dy)[1]))
#         #TODO: fix variance!?
#         return post_mean, tf.math.sqrt(tf.math.abs(post_var))


class ResidualSmooth(object):
    def __init__(self, freqs, phase_res):
        self.phase_res = phase_res
        self.freqs = freqs

    def solve_all_time(self, args):
        dir, ant = args

        ###
        # all time learn and smooth

        smooth_residual_mean_ta = tf.TensorArray(float_type, size=tf.shape(self.phase_res)[-1])
        smooth_residual_uncert_ta = tf.TensorArray(float_type, size=tf.shape(self.phase_res)[-1])

        def body(time, smooth_residual_mean_ta, smooth_residual_uncert_ta):
            loss = ResidualSmoothLoss(self.phase_res[0, dir, ant, :, time], self.freqs)
            #noise, sigma, lengthscale
            # brute_solver = help_brute_adam([2., 2., 2., 0.1], [0.2, 0.1, 0.2, 0.01], vectorized=True)
            # brute_solver = help_brute_bfgs([2., 2., 2., 0.2], [0.2, 0.2, 0.2, 0.02], vectorized=True)
            brute_solver = helper_brute_recursion(2, [1., 1., 1.], shrinkage=1/15., vectorized=True)
            params = brute_solver(loss.loss_func, [np.log(0.1), np.log(0.1), np.log(1.)])
            smoothed_phase_mean, smoothed_phase_uncert = loss.smooth_func(params)

            return [time + 1, smooth_residual_mean_ta.write(time, smoothed_phase_mean),
                    smooth_residual_uncert_ta.write(time, smoothed_phase_uncert)]

        def cond(time, *args):
            return time < tf.shape(self.phase_res)[-1]

        _, smooth_residual_mean_ta, smooth_residual_uncert_ta = tf.while_loop(cond,
                                                            body,
                                                            [tf.constant(0),
                                                             smooth_residual_mean_ta,
                                                             smooth_residual_uncert_ta],
                                                            back_prop=False)

        smooth_residual_mean = smooth_residual_mean_ta.stack()
        smooth_residual_uncert = smooth_residual_uncert_ta.stack()
        return [smooth_residual_mean, smooth_residual_uncert]

    def run(self, smoother_parallel_iterations=10):
        shape = tf.shape(self.phase_res)
        Npol, Nd, Na, Nf, Nt = shape[0], shape[1], shape[2], shape[3], shape[4]
        grid = [tf.reshape(t, (-1,)) for t in tf.meshgrid(tf.range(Nd), tf.range(Na), indexing='ij')]
        with tf.control_dependencies([tf.print('Smoothing residual phases')]):
            # Nd*Na, Nf, Nt
            residual_mean, residual_uncert = tf.map_fn(self.solve_all_time, grid, parallel_iterations=smoother_parallel_iterations,
                                             back_prop=False, dtype=[float_type, float_type])
        residual_mean = tf.reshape(residual_mean, (Npol, Nd, Na, Nf, Nt))
        residual_uncert = tf.reshape(residual_uncert, (Npol, Nd, Na, Nf, Nt))
        return residual_mean, residual_uncert

# class ResidualSmooth(object):
#     def __init__(self, freqs, phase_res):
#         self.phase_res = phase_res
#         self.freqs = freqs
#
#     def solve_all_time(self, args):
#         dir, ant = args
#
#         loss = ResidualSmoothLoss(self.phase_res[0, dir, ant, :, :], self.freqs)
#         # noise, sigma, lengthscale, mean
#         # brute_solver = help_brute_adam([2., 2., 2., 0.1], [0.2, 0.1, 0.2, 0.01], vectorized=True)
#         # brute_solver = help_brute_bfgs([2., 2., 2., 0.2], [0.2, 0.2, 0.2, 0.02], vectorized=True)
#         brute_solver = helper_brute_recursion(3, [2., 2., 2.], shrinkage=1 / 15., vectorized=True)
#         params = brute_solver(loss.loss_func, [np.log(0.1), np.log(0.1), np.log(1.)])
#         smooth_residual_mean, smooth_residual_uncert = loss.smooth_func(params)
#
#         return [smooth_residual_mean, smooth_residual_uncert]
#
#     def run(self, smoother_parallel_iterations=10):
#         shape = tf.shape(self.phase_res)
#         Npol, Nd, Na, Nf, Nt = shape[0], shape[1], shape[2], shape[3], shape[4]
#         grid = [tf.reshape(t, (-1,)) for t in tf.meshgrid(tf.range(Nd), tf.range(Na), indexing='ij')]
#         with tf.control_dependencies([tf.print('Smoothing residual phases')]):
#             # Nd*Na, Nf, Nt
#             residual_mean, residual_uncert = tf.map_fn(self.solve_all_time, grid,
#                                                        parallel_iterations=smoother_parallel_iterations,
#                                                        back_prop=False, dtype=[float_type, float_type])
#         residual_mean = tf.reshape(residual_mean, (Npol, Nd, Na, Nf, Nt))
#         residual_uncert = tf.reshape(residual_uncert, (Npol, Nd, Na, Nf, Nt))
#         return residual_mean, residual_uncert

def tf_wrap(phi):
    return tf.atan2(tf.math.sin(phi), tf.math.cos(phi))

def tf_unwrap(phi, axis=0):
    n = len(phi.shape)
    s0 = [slice(None, None, 1)] * n
    s0[axis] = slice(0, -1, 1)
    s1 = [slice(None,None,1)]*n
    s1[axis] = slice(1,None,1)
    dphi = phi[s1] - phi[s0]
    mod_phi = tf_wrap(dphi) - dphi
    cor = tf.cumsum(mod_phi, axis=axis, exclusive=False)
    s0[axis] = slice(0, 1, 1)
    cor = tf.concat([tf.zeros_like(phi[s0]), cor], axis=axis)
    return phi + cor

if __name__ == '__main__':

    input_datapack = '/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v6.h5'
    datapack = DataPack(input_datapack)
    screen_directions = get_screen_directions('/home/albert/ftp/image.pybdsm.srl.fits', max_N=None)
    maybe_create_posterior_solsets(datapack, 'sol000', posterior_name='posterior', screen_directions=screen_directions,
                                   remake_posterior_solsets=False)

    datapack.current_solset = 'sol000'
    axes = datapack.axes_phase
    _, times = datapack.get_times(axes['time'])

    # if len(sys.argv) != 4:
    #     raise ValueError("{} ant from_time to_time".format(sys.argv[0]))
    #
    # ant = int(sys.argv[1])
    # from_time, to_time = [int(l) for l in sys.argv[1:3]]
    num_ref_dirs = 2
    if num_ref_dirs is not None:
        if num_ref_dirs < 2:
            raise ValueError("Number of reference directions must be >= 2. Got {}.".format(num_ref_dirs))

    select = dict(dir=slice(None, None, 1),
                  ant=slice(None, None, 1),
                  time=slice(None, None, 1),
                  freq=slice(None, None, 1),
                  pol=slice(0, 1, 1))

    datapack_raw = DataPack(input_datapack, readonly=True)
    datapack_raw.current_solset = 'sol000'
    # Npol, Nd, Na, Nf, Nt
    datapack_raw.select(**select)
    phase_raw, _ = datapack_raw.phase
    amp_raw, axes = datapack_raw.amplitude
    timestamps, times = datapack_raw.get_times(axes['time'])
    _, freqs = datapack_raw.get_freqs(axes['freq'])
    
    Npol, Nd, Na, Nf, Nt = phase_raw.shape

    Yimag_full = amp_raw * np.sin(phase_raw)
    Yreal_full = amp_raw * np.cos(phase_raw)
    # Nd,Na
    gain_uncert = 0.25 * np.mean(np.abs(np.diff(Yimag_full, axis=-1)) + np.abs(np.diff(Yreal_full, axis=-1)), axis=-1).mean(
            -1).mean(0)
    ref_dir_selection = np.argsort(np.mean(gain_uncert, axis=1))[:num_ref_dirs]

    gain_uncert = np.maximum(gain_uncert, 0.02)

    # gain_uncert = 0.07*np.ones((Nd,Na))



    logging.info('Constructing graph')
    with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.device('/device:CPU:0'):
            Yreal_pl = tf.placeholder(float_type, shape=(Npol, Nd, Na, Nf, Nt))
            Yimag_pl = tf.placeholder(float_type, shape=(Npol, Nd, Na, Nf, Nt))
            freqs_pl = tf.placeholder(float_type, shape=(Nf,))
            gain_uncert_pl = tf.placeholder(float_type, shape=(Nd, Na))
            tec_conv = -8.4479745e6 / freqs_pl

            phase_raw_tf = tf.atan2(Yimag_pl, Yreal_pl)

            tec_system_solver = TecSystemSolve(freqs_pl, Yimag_pl, Yreal_pl, gain_uncert_pl, S=20, reference_dirs = ref_dir_selection, per_ref_parallel_iterations=6)
            tec_solution, tec_uncert_solution = tec_system_solver.run(lstsq_parallel_iterations=6)

            #Npol, Nd, Na, Nf, Nt
            phase_dd_mean = tec_solution[..., None, :] * tec_conv[:, None]
            phase_dd_uncert = tec_uncert_solution[..., None, :] * tec_conv[:, None]

            ###
            # smooth residual gains



            phase_res = tf_wrap(tf_wrap(phase_raw_tf) - tf_wrap(phase_dd_mean))
            phase_res = tf_unwrap(phase_res, axis=-1)

            residual_smoother = ResidualSmooth(freqs_pl, phase_res)
            #Npol, Nd, Na, Nf, Nt
            residual_mean, residual_uncert = residual_smoother.run(smoother_parallel_iterations=6)

            final_phase_mean = phase_dd_mean + residual_mean
            final_phase_res = tf_wrap(tf_wrap(phase_raw_tf) - tf_wrap(final_phase_mean))
            eff_tec = final_phase_res / tec_conv[:, None]
            final_phase_uncert = tf.math.sqrt(tf.math.square(phase_dd_uncert) + tf.math.square(residual_uncert))
        logging.info("Calculating...")
        tec_mean, tec_uncert, phase_mean, phase_uncert, eff_tec, residual_mean, phase_res = sess.run([tec_solution, tec_uncert_solution, final_phase_mean, final_phase_uncert, eff_tec, residual_mean, phase_res],
                                                                  feed_dict={Yreal_pl:Yreal_full, Yimag_pl: Yimag_full, freqs_pl:freqs, gain_uncert_pl:gain_uncert})
    import pylab as plt
    for d in range(phase_res.shape[1]):
        plt.close('all')
        plt.imshow(eff_tec[0,d,-1,:,:],cmap='coolwarm', vmin=-1., vmax=1.)
        plt.savefig('/home/albert/ftp/eff_tec_plots/ref_dirs_{:02d}_dir_{}.png'.format(num_ref_dirs, d))
        plt.close('all')
        plt.plot(phase_res[0,d, -1, :, 0])
        plt.plot(residual_mean[0, d, -1, :, 0])
        plt.savefig('/home/albert/ftp/eff_tec_plots/phase_res_{:02d}_dir_{}.png'.format(num_ref_dirs, d))

    logging.info("Storing results")
    datapack_save = DataPack(input_datapack, readonly=False)
    datapack_save.current_solset = 'data_posterior'
    # Npol, Nd, Na, Nf, Nt
    datapack_save.select(**select)
    datapack_save.phase = phase_mean
    datapack_save.weights_phase = phase_uncert
    datapack_save.tec = tec_mean
    datapack_save.weights_tec = tec_uncert
    logging.info("Stored results. Done")
