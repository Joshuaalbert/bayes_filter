from bayes_filter.filters import FreeTransitionSAEM
import tensorflow as tf
import tensorflow_probability as tfp
import os
from bayes_filter.misc import load_array_file, get_screen_directions
from bayes_filter import float_type
from bayes_filter.datapack import DataPack
import sys
from bayes_filter.feeds import IndexFeed,TimeFeed,CoordinateFeed, DataFeed, init_feed, ContinueFeed
from bayes_filter.coord_transforms import tf_coord_transform, itrs_to_enu_with_references
from bayes_filter.kernels import DTECIsotropicTimeGeneral, DTECIsotropicTimeGeneralODE
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au
from bayes_filter.frames import ENU
import numpy as np
import pylab as plt
import seaborn as sns
from timeit import default_timer
from bayes_filter import logging
from bayes_filter.settings import angle_type, dist_type
from timeit import default_timer


def arrays():
    return os.path.dirname(sys.modules["bayes_filter"].__file__)

def lofar_array(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    return load_array_file(lofar_array)

def lofar_array2(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    res = load_array_file(lofar_array)
    return res[0], res[1]
    return res[0][[0,50, 51]], res[1][[0,50,51],:]


def arrays():
    return os.path.dirname(sys.modules["bayes_filter"].__file__)

def lofar_array(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    return load_array_file(lofar_array)

def lofar_array2(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    res = load_array_file(lofar_array)
    return res[0][[0,50, 51]], res[1][[0,50,51],:]


def simulated_ddtec(tf_session, lofar_array, Nt=4, Nd=4, kern_type=0):
    class Simulated:
        def __init__(self, Nt=Nt, Nd=Nd, kern_type=kern_type):

            Na = len(lofar_array[0])

            with tf_session.graph.as_default():
                index_feed = IndexFeed(Nt//2)
                obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
                times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., Nt*30., Nt)[:, None],float_type)
                time_feed = TimeFeed(index_feed, times)
                cont_feed = ContinueFeed(time_feed)
                enu = ENU(location = ac.ITRS(*lofar_array[1][0,:]*au.m), obstime=obstime_init)
                up = ac.SkyCoord(east=0., north=0.,up=1., frame=enu).transform_to('icrs')
                ra  = up.ra.rad + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
                dec = up.dec.rad + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
                Xd  = tf.concat([ra, dec], axis=1)
                Xa  = tf.constant(lofar_array[1], dtype=float_type)
                coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                            coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [up.ra.rad, up.dec.rad], lofar_array[1][0,:])))
                init, next = init_feed(coord_feed)
                init_cont, cont = init_feed(cont_feed)
                tf_session.run([init, init_cont])
                if kern_type == 0:
                    kern = DTECIsotropicTimeGeneral(variance=0.5e-4,timescale=45.,lengthscales=10., a=200., b=60.,
                                             fed_kernel='RBF',obs_type='DDTEC', squeeze=True,
                                                    kernel_params={'resolution':3})
                elif kern_type == 1:
                    kern = DTECIsotropicTimeGeneralODE(variance=0.5e-4, timescale=45., lengthscales=15., a=200., b=60.,
                                                    fed_kernel='RBF', obs_type='DDTEC', squeeze=True, ode_type='fixed',
                                                       kernel_params={'resolution':3})
                elif kern_type == 2:
                    kern = DTECIsotropicTimeGeneralODE(variance=0.5e-4, timescale=45., lengthscales=15., a=200., b=60.,
                                                    fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                                       ode_type='adaptive', kernel_params={'rtol':1e-2})
                # kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(tf.convert_to_tensor(0.04,float_type), tf.convert_to_tensor(10.,float_type))

                size, time, rate = [],[],[]
                while True:
                    t0 = default_timer()
                    K,N = tf_session.run([kern.K(next),tf.shape(next)[0]])
                    dt = default_timer() - t0
                    print(kern_type, N,dt,N/dt)
                    size.append(N)
                    time.append(dt)
                    rate.append(N/dt)
                    if not tf_session.run(cont):
                        break
                    # plt.scatter(size, time)
                    # plt.show()
                    # plt.scatter(size,rate)
                    # plt.show()
    return Simulated()

if __name__ == '__main__':
    from tensorflow.python import debug as tf_debug
    sess = tf.Session(graph=tf.Graph())
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess.graph.as_default():
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=4, kern_type=0)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=4, kern_type=1)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=4, kern_type=2)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=8, kern_type=0)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=8, kern_type=1)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=8, kern_type=2)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=16, kern_type=0)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=16, kern_type=1)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=16, kern_type=2)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=32, kern_type=0)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=32, kern_type=1)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=32, kern_type=2)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=64, kern_type=0)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=64, kern_type=1)
        simulated_ddtec(sess, lofar_array(arrays()), Nt=4, Nd=64, kern_type=2)