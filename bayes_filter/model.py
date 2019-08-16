from gpflow.models import GPModel
from gpflow.likelihoods import Gaussian
from gpflow.params import Parameter
from gpflow import DataHolder, name_scope, params_as_tensors, autoflow
from gpflow import settings
float_type = settings.float_type
import tensorflow as tf
import numpy as np
from typing import List
from scipy.special import logsumexp
from gpflow.training import ScipyOptimizer

class HGPR(GPModel):

    def __init__(self, X, Y, Y_var, kern, regularisation_param=1., mean_function=None, parallel_iterations=1,
                 name=None):

        likelihood = Gaussian()
        # M, D
        X = DataHolder(X)
        # T, M
        Y = DataHolder(Y)
        num_latent = Y.shape[0]
        GPModel.__init__(self, X=X, Y=Y, kern=kern, likelihood=likelihood,
                         mean_function=mean_function, num_latent=num_latent, name=name)
        self.Y_var = DataHolder(Y_var)
        self.parallel_iterations = parallel_iterations
        self.regularisation_param = Parameter(regularisation_param, dtype=float_type, trainable=False)

    @name_scope('common')
    @params_as_tensors
    def _build_common(self):
        # M, M
        Kmm = self.kern.K(self.X)
        # B, M
        Y_std = tf.math.sqrt(self.Y_var)

        M = tf.shape(Kmm)[-1]
        # M, M
        eye = tf.linalg.eye(M, dtype=float_type)
        # B, M, M
        K_sigma = Kmm / (Y_std[..., :, None] * Y_std[..., None, :]) + eye
        # B, M, M
        L = tf.linalg.cholesky(K_sigma)
        # B, M
        Y = self.Y / Y_std
        # B, M, 1
        Ly = tf.linalg.triangular_solve(L, Y[..., :, None])

        return L, Ly, Y_std

    @name_scope('batched_likelihood')
    @params_as_tensors
    def _build_batched_likelihood(self, L, Ly, Y_std):


        M = tf.shape(L)[-1]

        # B
        maha = -0.5 * tf.reduce_sum(tf.math.square(Ly), axis=[-1, -2])

        # B, M
        x_ok = tf.not_equal(Y_std, tf.constant(np.inf, float_type))
        safe_x = tf.where(x_ok, Y_std, tf.ones_like(Y_std))
        # B
        logdetL = tf.reduce_sum(tf.where(x_ok, tf.math.log(tf.linalg.diag_part(L) * safe_x), tf.zeros_like(Y_std)),
                                axis=-1)

        constant = 0.5 * np.log(np.sqrt(2. * np.pi)) * tf.cast(M, float_type)
        # B
        log_marginal_likelihood = maha - self.regularisation_param * logdetL - constant
        return log_marginal_likelihood

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        r"""
        Construct a tensorflow function to compute the likelihood.
            \log p(Y | theta).
        """
        L, Ly, Y_std = self._build_common()
        log_marginal_likelihood = self._build_batched_likelihood(L, Ly, Y_std)
        return tf.reduce_sum(log_marginal_likelihood)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, the points at which we want to predict.
        This method computes
            p(F* | Y)
        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """

        L, Ly, Y_std = self._build_common()

        # M, N
        Kmn = self.kern.K(self.X, Xnew)
        # N, N
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)

        # B, M, N
        A = tf.linalg.triangular_solve(L, Kmn / Y_std[..., :, None])

        # B, N, 1
        post_mean = tf.matmul(A, Ly, transpose_a=True)[..., :, 0]
        if full_cov:
            # B, N, N
            post_cov = Knn - tf.matmul(A, A, transpose_a=True)
        else:
            # sum_k A[k,i]A[k,j]
            # B, N
            post_cov = Knn - tf.reduce_sum(tf.math.square(A), axis=-2)
        return post_mean, post_cov

    @autoflow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self._build_predict(Xnew)

    @autoflow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @autoflow((float_type, [None, None]))
    def log_marginal_likelihood_and_predict_f(self, Xnew, full_cov = False, only_mean = False):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        L, Ly, Y_std = self._build_common()
        log_marginal_likelihood = self._build_batched_likelihood(L, Ly, Y_std)

        # M, N
        Kmn = self.kern.K(self.X, Xnew)
        # N, N
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)

        # B, M, N
        A = tf.linalg.triangular_solve(L, Kmn / Y_std[..., :, None])

        # B, N, 1
        post_mean = tf.matmul(A, Ly, transpose_a=True)[..., :, 0]

        if full_cov:
            # B, N, N
            post_cov = Knn - tf.matmul(A, A, transpose_a=True)
        else:
            # sum_k A[k,i]A[k,j]
            # B, N
            post_cov = Knn - tf.reduce_sum(tf.math.square(A), axis=-2)

        if only_mean:
            return log_marginal_likelihood, post_mean
        return log_marginal_likelihood, post_mean, post_cov


class AverageModel(object):
    def __init__(self, models: List[HGPR]):
        self.models = models

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        if not isinstance(value, (tuple, list)):
            value = [value]

        for m in value:
            if m.num_latent is not value[0].num_latent:
                raise ValueError("num_latent must be the same")
        self._models = value

    def optimise(self):
        opt = ScipyOptimizer()
        for model in self.models:
            opt.minimize(model)

    def predict_f(self, X, only_mean=True):
        """
        Predict over all batches
        :param X:
        :return:
        """
        post_means = []
        post_vars = []
        log_marginal_likelihoods = []
        for model in self.models:
            if only_mean:
                # [B], [B, N]
                log_marginal_likelihood, post_mean = model.log_marginal_likelihood_and_predict_f(X, full_cov=False,
                                                                                                 only_mean=True)
            else:
                # [B], [B, N], [B, N]
                log_marginal_likelihood, post_mean, post_var = model.log_marginal_likelihood_and_predict_f(X, full_cov=False,
                                                                                                 only_mean=False)
                post_vars.append(post_var)
            post_means.append(post_mean)
            log_marginal_likelihoods.append(log_marginal_likelihood)
        # num_models, batch_size
        log_marginal_likelihoods = np.stack(log_marginal_likelihoods, axis=0)
        # batch_size
        # num_models, batch_size, N
        post_means = np.stack(post_means, axis = 0)
        # num_models, batch_size
        weights = np.exp(log_marginal_likelihoods - logsumexp(log_marginal_likelihoods, axis=0))
        # batch_size, N
        post_mean = np.sum(weights[:,:,None]*post_means, axis=0)
        if not only_mean:
            # num_models, batch_size, N
            post_vars = np.stack(post_vars, axis=0)
            # batch_size, N
            post_var = np.sum(weights[:,:,None]*(post_vars + np.square(post_means)), axis=0) - np.square(post_mean)
            return (weights, log_marginal_likelihoods), post_mean, post_var
        return (weights, log_marginal_likelihoods), post_mean

