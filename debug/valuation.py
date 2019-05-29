import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
from collections import OrderedDict


class ODE(object):
    def __init__(self, num_cities=1, num_clinic_sizes=1, num_clinic_specs=1):
        self.shapes = OrderedDict(V=(1,),
                                  discounted_cashflow=(1,),
                                  discount_rate=(1,),
                                  n_join=(num_cities, num_clinic_sizes, num_clinic_specs),
                                  n_churn=(num_cities, num_clinic_sizes, num_clinic_specs),
                                  gamma=(num_cities,),
                                  s=(num_clinic_sizes,),
                                  alpha=(num_cities, num_clinic_specs),
                                  P_join=(num_cities, num_clinic_sizes, num_clinic_specs),
                                  P_churn=(num_cities, num_clinic_sizes, num_clinic_specs),
                                  N_total=(num_cities, num_clinic_sizes, num_clinic_specs),
                                  burn_rate=(1,),
                                  retension=(num_cities, num_clinic_sizes, num_clinic_specs),
                                  frac_accessible=(num_cities, num_clinic_sizes, num_clinic_specs),
                                  invested=(1,),
                                  burned=(1,))
        self._boosts = []

    def boost(self, state, t_start, t_end, amount):
        if state not in self.shapes.keys():
            raise ValueError('{} invalid key'.format(state))
        self._boosts.append((state, t_start, t_end, amount))

    def _diff(self, t, V, discounted_cashflow, discount_rate, n_join, n_churn, gamma, s, alpha, P_join, P_churn,
              N_total, burn_rate, retension, frac_accessible, invested, burned):

        def _boost(tstart, tend, P_boost):
            window = tend - tstart
            mean = (tend + tstart) / 2.
            return P_boost * (2 * np.pi * window ** 2) ** (-0.5) * tf.exp(-0.5 * (t - mean) ** 2 / window ** 2)

        n = n_join - n_churn
        clinic_revenue = tf.einsum("ijk,j,ik->ijk", n, s, alpha)
        dVdt = tf.einsum("i,ijk->", gamma, clinic_revenue)

        ddiscounted_cashflowdt = (dVdt - burn_rate) / (1. + discount_rate) ** t
        ddiscount_ratedt = tf.zeros_like(discount_rate)

        dburneddt = burn_rate

        dinvesteddt = tf.zeros_like(invested)

        N_untaken = N_total * frac_accessible - n_join
        dn_joindt = P_join * N_untaken

        n_perm = retension * n_join
        n_non_perm = n - n_perm
        dn_churndt = P_churn * n_non_perm

        dsdt = tf.zeros_like(s)
        dalphadt = tf.zeros_like(alpha)
        dgammadt = tf.zeros_like(gamma)
        dP_joindt = tf.zeros_like(P_join)
        dP_churndt = tf.zeros_like(P_churn)
        dN_totaldt = (0.024 / 12.) * N_total
        dburn_ratedt = tf.zeros_like(burn_rate)
        dretensiondt = tf.zeros_like(retension)
        dfrac_accessibledt = tf.zeros_like(frac_accessible)

        augmented_out = OrderedDict(V=dVdt,
                                    discounted_cashflow=ddiscounted_cashflowdt,
                                    discount_rate=ddiscount_ratedt,
                                    n_join=dn_joindt,
                                    n_churn=dn_churndt,
                                    gamma=dgammadt,
                                    s=dsdt,
                                    alpha=dalphadt,
                                    P_join=dP_joindt,
                                    P_churn=dP_churndt,
                                    N_total=dN_totaldt,
                                    burn_rate=dburn_ratedt,
                                    retension=dretensiondt,
                                    frac_accessible=dfrac_accessibledt,
                                    invested=dinvesteddt,
                                    burned=dburneddt
                                    )
        for b in self._boosts:
            augmented_out[b[0]] = augmented_out[b[0]] + _boost(b[1], b[2], b[3])

        return list(augmented_out.values())

    @property
    def state_size(self):
        return np.sum([np.prod(shape) for shape in self.shapes.values()])

    @property
    def state_names(self):
        return list(self.shapes.keys())

    def get_derivative_and_jacobian_func(self, sess):
        state_pl = tf.placeholder(tf.float64, shape=self.state_size, name='state_pl')
        t_pl = tf.placeholder(tf.float64, shape=(), name='t_pl')

        out_derivative = self.derivative(t_pl, state_pl)
        out_jacobian = jacobian(out_derivative, state_pl, use_pfor=True, parallel_iterations=10)

        def diff_func(t_np, state_np):
            return sess.run(out_derivative, feed_dict={t_pl: t_np, state_pl: state_np})

        def jac_func(t_np, state_np):
            return sess.run(out_jacobian, feed_dict={t_pl: t_np, state_pl: state_np})

        return diff_func, jac_func

    def derivative(self, t, state):
        with tf.variable_scope("derivative") as scope:
            idx = 0
            split = {}
            for key, shape in self.shapes.items():
                m = np.prod(shape)
                split[key] = tf.cast(tf.reshape(state[idx:idx + m], shape), tf.float64)
                idx += m

            def _merge(*D):
                res = tf.concat([tf.reshape(d, (-1,)) for d in D], axis=0)
                return res

            return _merge(*self._diff(t=t, **split))

    def odeint(self, init_state, time_array, sess):
        diff_func, jac_func = self.get_derivative_and_jacobian_func(sess)
        out = odeint(diff_func, init_state, time_array, Dfun=jac_func, tfirst=True)
        out_dict = OrderedDict()
        jdx = 0
        for i, (k, v) in enumerate(self.shapes.items()):
            size = np.prod(v)
            out_dict[k] = np.reshape(out[:, jdx:jdx + size], (-1,) + v)
            jdx += size
        return out_dict


def run():
    with tf.Session(graph=tf.Graph()) as sess:
        ode = ODE(num_cities=1, num_clinic_sizes=1, num_clinic_specs=1)

        V = 0.
        discounted_cashflow = 0.
        discount_rate = 0.05
        n_join = 0.
        n_churn = 0.
        gamma = 0.01  # proportion of clinic revenue
        s = 3  # physios per clinic
        alpha = 6.5  # 1000$ / month
        P_join = np.random.uniform(0.05, 0.15)  # prob of someone new joining in a month
        P_churn = np.random.uniform(0.05, 0.15)  # prob of someone with churning in a month
        N_total = 1.2 / 37. * 20e3  # total number of physios
        burn_rate = 49.5
        retension = np.random.uniform(0.25, 0.65)
        frac_accessible = 0.
        invested = 335.
        burned = 299.

        init_state = np.array(
            [V, discounted_cashflow, discount_rate, n_join, n_churn, gamma, s, alpha, P_join, P_churn, N_total,
             burn_rate, retension, frac_accessible, invested, burned])

        ode.boost('frac_accessible', 4., 6., np.random.uniform(0.25, 0.5))
        ode.boost('invested', 0., 2., 300.)
        ode.boost('N_total', 6., 10., np.random.uniform(10e3, 20e3))

        out = ode.odeint(init_state, time_array, sess)
    return out


import pylab as plt
from IPython import display
from dask.multiprocessing import get

if __name__ == '__main__':

    ode = ODE()
    fig, ax = plt.subplots(1, 1)

    dsk = {}

    time_array = np.linspace(0, 12 * 1, 200)

    N = 1000
    labels = False
    fig, axs = plt.subplots(18, 1, figsize=(6, 20 * 3))
    for j in range(N):
        dsk[j] = (run,)

    results = get(dsk, list(range(N)), num_workers=64)

    for j in range(N):
        out = results[j]
        c = 'blue'

        ax = axs[0]
        ax.plot(time_array, out['V'] + out['invested'] - out['burned'], label='Total Value', alpha=0.1, c=c)
        if j == 0:
            ax.legend()

        ax = axs[1]
        ax.plot(time_array, out['V'], label='Value', ls='-', alpha=0.1, c=c)
        ax.plot(time_array, out['invested'], label='invested', ls='dashed', alpha=0.1, c=c)
        ax.plot(time_array, out['burned'], label='burned', alpha=0.1, ls='dotted', c=c)
        if j == 0:
            ax.legend()

        for i, name in enumerate(ode.state_names):
            ax = axs[i + 2]
            ax.plot(time_array, out[name].flatten(), label=name, alpha=0.1, c=c)
            if j == 0:
                ax.legend()

        # display.clear_output(wait=True)
        # display.display(plt.gcf())

    plt.show()


# import pylab as plt
# from IPython import display
#
# fig, ax = plt.subplots(1, 1)
#
# results = []
#
# time_array = np.linspace(0, 12 * 1, 200)
#
# N = 100
# labels = False
# fig, axs = plt.subplots(18, 1, figsize=(6, 20 * 3))
# for j in range(N):
#     with tf.Session(graph=tf.Graph()) as sess:
#         ode = ODE(num_cities=1, num_clinic_sizes=1, num_clinic_specs=1)
#
#         V = 0.
#         discounted_cashflow = 0.
#         discount_rate = 0.05
#         n_join = 0.
#         n_churn = 0.
#         gamma = 0.01  # proportion of clinic revenue
#         s = 3  # physios per clinic
#         alpha = 6.5  # 1000$ / month
#         P_join = np.random.uniform(0.05, 0.15)  # prob of someone new joining in a month
#         P_churn = np.random.uniform(0.05, 0.15)  # prob of someone with churning in a month
#         N_total = 1.2 / 37. * 20e3  # total number of physios
#         burn_rate = 49.5
#         retension = np.random.uniform(0.25, 0.65)
#         frac_accessible = 0.
#         invested = 335.
#         burned = 299.
#
#         init_state = np.array(
#             [V, discounted_cashflow, discount_rate, n_join, n_churn, gamma, s, alpha, P_join, P_churn, N_total,
#              burn_rate, retension, frac_accessible, invested, burned])
#
#         ode.boost('frac_accessible', 4., 6., np.random.uniform(0.25, 0.5))
#         ode.boost('invested', 0., 2., 300.)
#         ode.boost('N_total', 6., 10., np.random.uniform(10e3, 20e3))
#
#         out = ode.odeint(init_state, time_array, sess)
#         results.append(out)
#
#         c = 'blue'
#
#         ax = axs[0]
#         ax.plot(time_array, out['V'] + out['invested'] - out['burned'], label='Total Value', alpha=0.1, c=c)
#         if j == 0:
#             ax.legend()
#
#         ax = axs[1]
#         ax.plot(time_array, out['V'], label='Value', ls='-', alpha=0.1, c=c)
#         ax.plot(time_array, out['invested'], label='invested', ls='dashed', alpha=0.1, c=c)
#         ax.plot(time_array, out['burned'], label='burned', alpha=0.1, ls='dotted', c=c)
#         if j == 0:
#             ax.legend()
#
#         for i, name in enumerate(ode.state_names):
#             ax = axs[i + 2]
#             ax.plot(time_array, out[name].flatten(), label=name, alpha=0.1, c=c)
#             if j == 0:
#                 ax.legend()
#
#         display.clear_output(wait=True)
#         display.display(plt.gcf())
#
# # plt.show()