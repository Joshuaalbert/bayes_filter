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
