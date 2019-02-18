from bayes_filter.filters import FreeTransitionSAEM
import tensorflow as tf
import tensorflow_probability as tfp
import os
from bayes_filter.misc import load_array_file
from bayes_filter import float_type
import sys
from bayes_filter.data_feed import IndexFeed,TimeFeed,CoordinateFeed, DataFeed, init_feed, ContinueFeed
from bayes_filter.coord_transforms import tf_coord_transform, itrs_to_enu_with_references
from bayes_filter.kernels import DTECIsotropicTimeGeneralODE, DTECIsotropicTimeGeneral
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

            Nt, Nd, Na, Nf = 30, 4, len(lofar_array[0]), 6

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
                # ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
                # dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
                ra = ra - 1.*np.pi/180.
                dec = dec - 1.* np.pi / 180.
                Xd = tf.concat([ra, dec], axis=1)
                star_coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                                 coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))

                init, next = init_feed(coord_feed)
                init_star, next_star = init_feed(star_coord_feed)
                init_cont, cont = init_feed(cont_feed)
                tf_session.run([init, init_cont, init_star])
                kern = DTECIsotropicTimeGeneral(variance=0.5e-4,timescale=45.,lengthscales=15., a=200., b=60.,
                                         fed_kernel='RBF',obs_type='DDTEC', squeeze=True)
                # kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(tf.convert_to_tensor(0.04,float_type), tf.convert_to_tensor(10.,float_type))

                from timeit import default_timer
                t0 = default_timer()
                Y_real, Y_imag = [],[]
                Y_real_star, Y_imag_star = [], []
                ddtec_true, ddtec_star = [],[]
                while True:
                    K,N = tf_session.run([kern.K(tf.concat([next,next_star],axis=0)),tf.shape(next)[0]])
                    L = np.linalg.cholesky(K+1e-6*np.eye(K.shape[-1]))
                    np.random.seed(0)
                    ddtec = np.einsum('ab,b->a',L, np.random.normal(size=L.shape[1]))
                    ddtec_true.append(ddtec[:N])
                    ddtec_star.append(ddtec[N:])
                    freqs = np.linspace(110.e6, 160.e6, Nf)
                    Y_real.append(np.cos(-8.448e9 * ddtec[:N,None]/freqs))
                    Y_imag.append(np.sin(-8.448e9 * ddtec[:N, None] / freqs))
                    Y_real_star.append(np.cos(-8.448e9 * ddtec[N:, None] / freqs))
                    Y_imag_star.append(np.sin(-8.448e9 * ddtec[N:, None] / freqs))
                    if not tf_session.run(cont):
                        break
                self.Y_real_star = np.concatenate(Y_real_star,axis=0).reshape((Nt, Nd, Na, Nf))
                self.Y_imag_star = np.concatenate(Y_imag_star, axis=0).reshape((Nt, Nd, Na, Nf))
                Y_real_true = np.concatenate(Y_real,axis=0).reshape((Nt, Nd, Na, Nf))
                Y_real = Y_real_true + 0.05*np.random.normal(size=Y_real_true.shape)
                # Y_real[Nt//2:Nt//2 + 5, ...] *= 0.5
                Y_imag_true = np.concatenate(Y_imag, axis=0).reshape((Nt, Nd, Na, Nf))
                Y_imag = Y_imag_true + 0.05 * np.random.normal(size=Y_imag_true.shape)
                # Y_imag[Nt // 2:Nt // 2 + 5, ...] *= 0.5
                self.freqs = freqs
                self.ddtec_true = np.concatenate(ddtec_true,axis=0).reshape((Nt, Nd, Na))
                self.ddtec_star = np.concatenate(ddtec_star, axis=0).reshape((Nt, Nd, Na))
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
            num_samples=1000, num_chains=1,parallel_iterations=10, num_leapfrog_steps=3,target_rate=0.6,
            num_burnin_steps=100,num_saem_samples=500,saem_maxsteps=3,initial_stepsize=7e-3,
            init_kern_params={'y_sigma':0.5,'variance':0.5e-4,'timescale':45.,'lengthscales':15., 'a':200., 'b':60.},
            which_kernel=0, kernel_params={'resolution':3})
        sess.run(inits)
        cont = True
        while cont:
            res = sess.run(filtered_res)
            print("post_logp", res.post_logp,"test_logp", res.test_logp)
            # plt.plot(res.step_sizes)
            # plt.show()
            # plt.hist(res.ess.flatten(),bins=100)
            # plt.show()
            times = simulated_ddtec.np_times[:,0]
            ddtec_true = simulated_ddtec.ddtec_true
            ddtec_star = simulated_ddtec.ddtec_star
            Y_real_star = simulated_ddtec.Y_real_star
            Y_imag_star = simulated_ddtec.Y_imag_star


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
            ax1.set_title("Model space solution")

            ax2.plot(times, simulated_ddtec.Y_imag_true[:, 0, 1, :], c='black', alpha=0.5, ls='dotted', label='Ytrue')
            ax2.plot(times, res.extra.Y_imag_data[:, 0, 1, :], c='red', alpha=0.5, ls='dotted', label='Ydata')
            ax2.plot(times, res.Y_imag[1,:,0,1,:], label='posterior')
            [ax2.fill_between(times, res.Y_imag[0,:,0,1,i], res.Y_imag[2,:,0,1,i],
                              alpha=0.5) for i, f in enumerate(simulated_ddtec.freqs)]
            ax2.set_title("Data space solution")
            ax2.legend()

            ax3.plot(times, ddtec_star[:, 0, 1])
            ax3.plot(times, res.dtec_star[1, :, 0, 1])
            ax3.fill_between(times, res.dtec_star[0, :, 0, 1], res.dtec_star[2, :, 0, 1], alpha=0.5)
            ax3.legend()
            ax3.set_title("Model space solution*")

            ax4.plot(times, simulated_ddtec.Y_imag_star[:, 0, 1, :], c='black', alpha=0.5, ls='dotted', label='Ytrue')
            # ax4.plot(times, res.extra.Y_imag_star[:, 0, 1, :], c='red', alpha=0.5, ls='dotted', label='Ydata')
            ax4.plot(times, res.Y_imag_star[1, :, 0, 1, :], label='posterior')
            [ax4.fill_between(times, res.Y_imag_star[0, :, 0, 1, i], res.Y_imag_star[2, :, 0, 1, i],
                              alpha=0.5) for i, f in enumerate(simulated_ddtec.freqs)]
            ax4.set_title("Data space solution*")
            ax4.legend()
            plt.show()


            # print(res)
            cont = res.cont