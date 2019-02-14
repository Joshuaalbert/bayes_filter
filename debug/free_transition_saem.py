from bayes_filter.filters import FreeTransitionSAEM
import tensorflow as tf
import tensorflow_probability as tfp
import os
from bayes_filter.misc import load_array_file
from bayes_filter import float_type
import sys
from bayes_filter.data_feed import IndexFeed,TimeFeed,CoordinateFeed, DataFeed, init_feed, ContinueFeed
from bayes_filter.coord_transforms import tf_coord_transform, itrs_to_enu_with_references
from bayes_filter.kernels import DTECIsotropicTimeGeneral
import astropy.time as at
import numpy as np
import pylab as plt
import seaborn as sns
from timeit import default_timer
from bayes_filter.settings import angle_type, dist_type


def arrays():
    return os.path.dirname(sys.modules["bayes_filter"].__file__)

def lofar_array(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    return load_array_file(lofar_array)

def lofar_array2(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    res = load_array_file(lofar_array)
    return res[0][[0,50, 51]], res[1][[0,50,51],:]


def simulated_ddtec(tf_session, lofar_array):
    class Simulated:
        def __init__(self):

            Nt, Nd, Na, Nf = 30, 2, len(lofar_array[0]), 6

            with tf_session.graph.as_default():
                index_feed = IndexFeed(Nt)
                obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
                times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., Nt*30., Nt)[:, None],float_type)
                time_feed = TimeFeed(index_feed, times)
                cont_feed = ContinueFeed(time_feed)
                ra  = np.pi / 4. + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
                dec = np.pi / 4. + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
                Xd  = tf.concat([ra, dec], axis=1)
                Xa  = tf.constant(lofar_array[1], dtype=float_type)
                coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                            coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
                ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(Nd+1, 1))
                dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(Nd+1, 1))
                Xd = tf.concat([ra, dec], axis=1)
                star_coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                                 coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))

                init, next = init_feed(coord_feed)
                init_cont, cont = init_feed(cont_feed)
                tf_session.run([init, init_cont])
                kern = DTECIsotropicTimeGeneral(variance=1e-4,timescale=45.,lengthscales=10., a=200., b=60.,
                                         fed_kernel='RBF',obs_type='DDTEC',resolution=5, squeeze=True)
                # kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(tf.convert_to_tensor(0.04,float_type), tf.convert_to_tensor(10.,float_type))

                from timeit import default_timer
                t0 = default_timer()
                Y_real, Y_imag = [],[]
                while True:
                    K = tf_session.run(kern.K(next))
                    # K = tf_session.run(kern.matrix(next[:,4:7],next[:,4:7]))
                    L = np.linalg.cholesky(K+1e-6*np.eye(K.shape[-1]))
                    ddtec = np.einsum('ab,b->a',L, np.random.normal(size=L.shape[1]))
                    freqs = np.linspace(110.e6, 160.e6, Nf)
                    Y_real.append(np.cos(-8.448e9*ddtec[:,None]/freqs))
                    Y_imag.append(np.sin(-8.448e9 * ddtec[:, None] / freqs))
                    if not tf_session.run(cont):
                        break
                Y_real_true = np.concatenate(Y_real,axis=0).reshape((Nt, Nd, Na, Nf))
                Y_real = Y_real_true + 0.5*np.random.laplace(size=Y_real_true.shape)
                Y_real[Nt//2:Nt//2 + 5, ...] *= 0.5
                Y_imag_true = np.concatenate(Y_imag, axis=0).reshape((Nt, Nd, Na, Nf))
                Y_imag = Y_imag_true + 0.5 * np.random.laplace(size=Y_imag_true.shape)
                Y_imag[Nt // 2:Nt // 2 + 5, ...] *= 0.5
                self.freqs = freqs
                self.ddtec_true = ddtec.reshape((Nt, Nd, Na))
                self.Y_real = Y_real
                self.Y_imag = Y_imag
                self.Y_real_true = Y_real_true
                self.Y_imag_true = Y_imag_true
                # self.np_freqs = tf_session.run(freqs)
                self.np_times = tf_session.run(times)
                self.ddtec = ddtec
                self.coord_feed = coord_feed
                self.star_coord_feed = star_coord_feed
                self.data_feed = DataFeed(index_feed, Y_real, Y_imag, event_size=1)
    return Simulated()

if __name__ == '__main__':
    from tensorflow.python import debug as tf_debug
    sess = tf.Session(graph=tf.Graph())
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess.graph.as_default():
        simulated_ddtec = simulated_ddtec(sess, lofar_array2(arrays()))

        free_transition = FreeTransitionSAEM(
            simulated_ddtec.freqs,
            simulated_ddtec.data_feed,
            simulated_ddtec.coord_feed,
            simulated_ddtec.star_coord_feed)

        filtered_res, inits = free_transition.filter_step(
            num_samples=1000, parallel_iterations=10, num_leapfrog_steps=3,target_rate=0.6,
            num_burnin_steps=100,num_saem_samples=100,saem_bfgs_maxsteps=1,initial_stepsize=7e-3,
            init_kern_params={'variance':1e-4,'y_sigma':0.2,'lengthscales':15.,'timescale':50.})
        sess.run(inits)
        cont = True
        while cont:
            res = sess.run(filtered_res)
            # plt.plot(res.step_sizes)
            # plt.show()
            # plt.hist(res.ess.flatten(),bins=100)
            # plt.show()
            times = simulated_ddtec.np_times[:,0]
            ddtec_true = simulated_ddtec.ddtec_true

            # plt.plot(times, res.Y_imag[1,:,0,1,0],c='black',lw=2.)
            # plt.fill_between(times, res.Y_imag[0,:,0,1,0], res.Y_imag[2,:,0,1,0],alpha=0.5)
            # plt.plot(times, res.extra.Y_imag_data[:, 0, 1, 0], c='red', lw=1.)
            # plt.plot(times, simulated_ddtec.Y_imag_true[:, 0, 1, 0], c='green', lw=1.)
            # plt.show()

            plt.style.use('ggplot')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            ax1.plot(times, ddtec_true[:, 0, 1])
            ax1.plot(times, res.dtec[1,:,0,1])
            ax1.fill_between(times, res.dtec[0,:,0,1], res.dtec[2,:,0,1], alpha=0.5)
            ax1.legend()
            ax1.set_title("Model space solution vs true")

            ax2.plot(times, simulated_ddtec.Y_imag_true[:, 0, 1, :], c='black', alpha=0.5, ls='dotted', label='Ytrue')
            ax2.plot(times, res.Y_imag[1,:,0,1,:], label='posterior')
            [ax2.fill_between(times, res.Y_imag[0,:,0,1,i], res.Y_imag[2,:,0,1,i],
                              alpha=0.5) for i, f in enumerate(simulated_ddtec.freqs)]
            ax2.set_title("Data space solution vs true")
            ax2.legend()

            ax3.plot(times, res.extra.Y_imag_data[:, 0, 1, :], c='black', alpha=0.5, ls='dotted', label='Ydata')
            ax3.plot(times, res.Y_imag[1,:,0,1,:], label='posterior')
            [ax3.fill_between(times, res.Y_imag[0,:,0,1,i], res.Y_imag[2,:,0,1,i],
                              alpha=0.5) for i,f in enumerate(simulated_ddtec.freqs)]
            ax3.legend()
            ax3.set_title("Data space solution vs data")

            sns.kdeplot(res.ess.flatten(), ax=ax4, shade=True, alpha=0.5)
            # ax4.plot(step_sizes[0], label='y_sigma stepsize')
            # ax4.plot(step_sizes[1], label='dtec stepsize')
            # ax4.set_yscale("log")
            # ax4.set_title("stepsizes")
            ax4.legend()
            ax4.set_title('Effective sample size')
            plt.show()


            # print(res)
            cont = res.cont