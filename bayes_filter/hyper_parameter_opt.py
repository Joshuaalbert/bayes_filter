"""
This file contains the code to use gpflow to optimise the kernel hyperparameters of D(D)TEC data.
We use a global concensus model to speed things up.
"""
from bayes_filter import float_type
from bayes_filter.callbacks import Callback

from .kernels import DTECIsotropicTimeGeneral
from .misc import safe_cholesky

import tensorflow as tf
import gpflow as gp

from gpflow import transforms
from gpflow import settings
from gpflow import DataHolder
from gpflow.logdensities import multivariate_normal
from . import logging

from gpflow.params import Parameter, Parameterized, ParamList
from gpflow.decors import params_as_tensors, autoflow

float_type = settings.float_type

class GPRCustom(gp.models.GPR):

    def __init__(self, X, Y, Y_var, kern, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        Y_var = DataHolder(Y_var)
        super(GPRCustom,self).__init__( X, Y, kern, mean_function, **kwargs)
        self.Y_var = Y_var


    @gp.params_as_tensors
    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    def predict_density_full_cov(self, Xnew, Ynew, ground=False):
        """
        Compute the (log) density of the data Ynew at the points Xnew
        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew,full_cov=True)
        #Knn + sigma^2I + Knm (Kmm + sigma^2I)^-1 Kmn
        if ground:
            K = pred_f_var
            L = safe_cholesky(K[0,:,:])
        else:
            K = pred_f_var + self.likelihood.variance*tf.eye(tf.shape(Xnew)[0],dtype=Ynew.dtype)
            L = tf.cholesky(K)[0,:,:]
        return gp.logdensities.multivariate_normal(Ynew, pred_f_mean, L)

    @gp.params_as_tensors
    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    def predict_density_independent(self, Xnew, Ynew, ground=False):
        """
        Compute the (log) density of the data Ynew at the points Xnew
        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew,full_cov=False)
        if ground:
            var = pred_f_var
        else:
            #diag(Knn + sigma^2I + Knm (Kmm + sigma^2I)^-1 Kmn)
            var = pred_f_var + self.likelihood.variance
        return gp.logdensities.gaussian(Ynew, pred_f_mean, var)[:,0]

    @gp.name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        r"""
        Construct a tensorflow function to compute the likelihood.
            \log p(Y | theta).
        """
        K = self.kern.K(self.X) + tf.matrix_diag(self.Y_var[:,0])
        #tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        L = safe_cholesky(K)
        m = self.mean_function(self.X)
        logpdf = multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y
        return tf.reduce_sum(logpdf)

class DTECKernel(gp.kernels.Kernel):
    def __init__(self, input_dim, variance=1., lengthscales=10.0,
                 velocity=[0.,0.,0.], a = 250., b = 50., timescale=30.,resolution=10,
                 active_dims=None, fed_kernel='RBF', obs_type='DTEC',name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        """
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance, transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)
        # (3,)
        self.lengthscales = Parameter(lengthscales, transform=transforms.positiveRescale(lengthscales),
                                      dtype=settings.float_type)
#         # (3,)
#         self.velocity = Parameter(velocity, transform=transforms.positive,
#                                       dtype=settings.float_type)
        self.a = Parameter(a, transform=transforms.positiveRescale(a),
                                      dtype=settings.float_type)
        self.b = Parameter(b, transform=transforms.positiveRescale(b),
                                      dtype=settings.float_type)
        self.timescale = Parameter(timescale, transform=transforms.positiveRescale(timescale),
                                      dtype=settings.float_type)
        self.resolution = resolution
        self.obs_type = obs_type
        self.fed_kernel = fed_kernel

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.diag_part(self.K(X,None))

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):

        if not presliced:
            X, X2 = self._slice(X, X2)

        kern = DTECIsotropicTimeGeneral(variance=self.variance, lengthscales=self.lengthscales,
                                a= self.a, b=self.b, timescale=self.timescale,
                                        fed_kernel=self.fed_kernel, obs_type=self.obs_type,
                               squeeze=True,#ode_type='adaptive',
                               kernel_params={'resolution':self.resolution})
        return kern.K(X,X2)

class KernelHyperparameterSolveCallback(Callback):
    def __init__(self,resolution=8, maxiter=100,obs_type='DDTEC', fed_kernel='RBF'):
        """
            Create the python function for use with tf.py_function that will solve the hyperparameter inference problem
            using GPFlow's L-BFGS solver.

            :param resolution: int
                The resolution of the DTECKernel
            :param maxiter: int
                The maximum number of iterations in L-BFGS
            :param obs_type: str
                The type of data TEC, DTEC, DDTEC
            :param fed_kernel: str
                The type of FED kernel: RBF, M12, M32, M52
                TODO: add Histogram spectrum
            :return: callable
                The function that solves hyperparameters
            """
        super(KernelHyperparameterSolveCallback, self).__init__(resolution=resolution,
                                                                maxiter=maxiter,
                                                                obs_type=obs_type,
                                                                fed_kernel=fed_kernel)
    def generate(self, resolution, maxiter,obs_type, fed_kernel):
        self.output_dtypes = [float_type]*5
        self.name = 'KernelHyperparameterSolveCallback'


        def optimise_hyperparams(X, mean_dtec, var_dtec, init_variance, init_lengthscales, init_a, init_b, init_timescale):
            """
            py_function that solves hyperparameters for DDTEC kernel.

            :param X:
            :param mean_dtec:
            :param var_dtec:
            :param init_variance:
            :param init_lengthscales:
            :param init_a:
            :param init_b:
            :param init_timescale:
            :return:
            """

            with tf.Session(graph=tf.Graph()) as sess:
                kern = DTECKernel(13,
                                variance=init_variance,
                                lengthscales=init_lengthscales,
                                a = init_a,
                                b = init_b,
                                timescale = init_timescale,
                                resolution=resolution,
                                fed_kernel=fed_kernel,
                                obs_type=obs_type)
                                # **states)

                m = GPRCustom(X, mean_dtec, var_dtec, kern)

                m.likelihood.variance.trainable = False

                gp.train.ScipyOptimizer().minimize(m,maxiter=maxiter)
                logging.info("Final loglikelihood: {}".format(m.compute_log_likelihood()))

                variance = kern.variance.value
                lengthscales = kern.lengthscales.value
                a = kern.a.value
                b = kern.b.value
                timescale = kern.timescale.value

            return [variance, lengthscales, a, b, timescale]

        return optimise_hyperparams