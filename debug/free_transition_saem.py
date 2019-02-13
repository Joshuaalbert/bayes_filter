from bayes_filter.filters import FreeTransitionSAEM
import tensorflow as tf
import os
from bayes_filter.misc import load_array_file
from bayes_filter import float_type
import sys
from bayes_filter.data_feed import IndexFeed,TimeFeed,CoordinateFeed, DataFeed, init_feed, ContinueFeed
from bayes_filter.coord_transforms import tf_coord_transform, itrs_to_enu_with_references
from bayes_filter.kernels import DTECIsotropicTimeGeneral
import astropy.time as at
import numpy as np


def arrays():
    return os.path.dirname(sys.modules["bayes_filter"].__file__)

def lofar_array(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    return load_array_file(lofar_array)

def lofar_array2(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    res = load_array_file(lofar_array)
    return res[0][::2], res[1][::2,:]


def simulated_ddtec(tf_session, lofar_array):
    class Simulated:
        def __init__(self):
            with tf_session.graph.as_default():
                index_feed = IndexFeed(3)
                obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
                times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 9*30., 3)[:, None],float_type)
                time_feed = TimeFeed(index_feed, times)
                cont_feed = ContinueFeed(time_feed)
                ra  = np.pi / 4. + 2. * np.pi / 180. * tf.random_normal(shape=(2, 1))
                dec = np.pi / 4. + 2. * np.pi / 180. * tf.random_normal(shape=(2, 1))
                Xd  = tf.concat([ra, dec], axis=1)
                Xa  = tf.constant(lofar_array[1], dtype=float_type)
                coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                            coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
                ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
                dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
                Xd = tf.concat([ra, dec], axis=1)
                star_coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                                 coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))

                init, next = init_feed(coord_feed)
                init_cont, cont = init_feed(cont_feed)
                tf_session.run([init, init_cont])
                kern = DTECIsotropicTimeGeneral(variance=0.07,timescale=30.,lengthscales=15., a=250., b=50.,
                                         fed_kernel='RBF',obs_type='DDTEC',resolution=5, squeeze=True)
                from timeit import default_timer
                t0 = default_timer()
                Y_real, Y_imag = [],[]
                while True:
                    K = tf_session.run(kern.K(next))
                    L = np.linalg.cholesky(K+1e-6*np.eye(K.shape[-1]))
                    ddtec = np.einsum('ab,b->a',L, np.random.normal(size=L.shape[1]))
                    freqs = np.linspace(110.e6, 160.e6, 6)
                    Y_real.append(np.cos(-8.448e9*ddtec[:,None]/freqs))
                    Y_imag.append(np.sin(-8.448e9 * ddtec[:, None] / freqs))
                    if not tf_session.run(cont):
                        break
                Y_real = np.concatenate(Y_real,axis=0).reshape((3, 2, 31,len(freqs)))
                Y_imag = np.concatenate(Y_imag, axis=0).reshape((3, 2, 31,len(freqs)))
                self.freqs = freqs
                self.Y_real = Y_real
                self.Y_imag = Y_imag
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

        filtered_res, inits = free_transition.filter_step(num_samples=20, num_chains=1, parallel_iterations=1, num_leapfrog_steps=5,
                               target_rate=0.5, num_burnin_steps=0,num_saem_samples=12,saem_steps=13,saem_learning_rate=0.1)
        sess.run(inits)
        cont = True
        while cont:
            res = sess.run(filtered_res)
            # print(res)
            cont = res.cont

