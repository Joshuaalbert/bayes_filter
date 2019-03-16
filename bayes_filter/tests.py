import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys, os
from .data_feed import CoordinateFeed, TimeFeed, IndexFeed, DataFeed, init_feed, ContinueFeed, CoordinateDimFeed
from .misc import flatten_batch_dims, load_array_file, make_coord_array, timer, diagonal_jitter, log_normal_solve_fwhm, K_parts, random_sample, make_example_datapack, safe_cholesky
from .coord_transforms import itrs_to_enu_6D, tf_coord_transform, itrs_to_enu_with_references
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at
from .frames import ENU
from .settings import dist_type, float_type, jitter
from .parameters import SphericalToCartesianBijector, ConstrainedBijector, ScaledPositiveBijector, ScaledBijector, Parameter, ScaledLowerBoundedBijector
from .kernels import DTECIsotropicTimeGeneral, DTECIsotropicTimeGeneralODE, Histogram
from .logprobabilities import DTECToGains
from .dblquad import dblquad
from .plotting import DatapackPlotter, animate_datapack,plot_vornoi_map
import pylab as plt

@pytest.fixture
def tf_graph():
    return tf.Graph()


@pytest.fixture
def tf_session(tf_graph):
    sess = tf.Session(graph=tf_graph)
    return sess

@pytest.fixture
def arrays():
    return os.path.dirname(sys.modules["bayes_filter"].__file__)

@pytest.fixture
def lofar_array(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    return load_array_file(lofar_array)

@pytest.fixture
def lofar_array2(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    res = load_array_file(lofar_array)
    return res[0][::2], res[1][::2,:]


###
# Feed tests

@pytest.fixture
def index_feed(tf_graph):
    with tf_graph.as_default():
        return IndexFeed(2)


@pytest.fixture
def time_feed(tf_graph, index_feed):
    with tf_graph.as_default():
        times = tf.linspace(0.,100.,9)[:,None]
        return TimeFeed(index_feed,times)


@pytest.fixture
def coord_feed(tf_graph, time_feed, lofar_array):
    with tf_graph.as_default():
        ra = np.pi/4 + 2.*np.pi/180. * tf.random_normal(shape=(4,1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        Xd = tf.concat([ra,dec],axis=1)
        Xa = tf.constant(lofar_array[1],dtype=float_type)
        return CoordinateFeed(time_feed, Xd, Xa, coord_map = tf_coord_transform(itrs_to_enu_6D))

@pytest.fixture
def data_feed(tf_graph, index_feed):
    with tf_graph.as_default():
        shape1 = (1,2,3,4)
        shape2 = (1, 2, 3, 4)
        data1 = tf.ones(shape1)
        data2 = tf.ones(shape2)
        return DataFeed(index_feed, data1, data2)


def test_index_feed(tf_session):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(2)
        init, next = init_feed(index_feed)
        tf_session.run(init)
        out = tf_session.run(next)
        assert out[1] - out[0] == index_feed._step


def test_time_feed(tf_session, index_feed):
    with tf_session.graph.as_default():
        times = tf.linspace(0., 10., 9)[:, None]
        time_feed = TimeFeed(index_feed, times)
        init, next = init_feed(time_feed)
        tf_session.run(init)
        out,slice_size = tf_session.run([next,time_feed.slice_size])
        assert out.shape == (slice_size, 1)


def test_continue_feed(tf_session):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(2)
        times = tf.linspace(0., 10., 3)[:, None]
        time_feed = TimeFeed(index_feed, times)
        cont_feed = ContinueFeed(time_feed)
        init, next = init_feed(cont_feed)
        tf_session.run(init)
        # out = tf_session.run([next])
        assert tf_session.run(next) == True
        assert tf_session.run(next) == False
        assert tf_session.run(next) == False
        assert tf_session.run(next) == False

        index_feed = IndexFeed(3)
        times = tf.linspace(0., 10., 3)[:, None]
        time_feed = TimeFeed(index_feed, times)
        cont_feed = ContinueFeed(time_feed)
        init, next = init_feed(cont_feed)
        tf_session.run(init)
        # out = tf_session.run([next])
        assert tf_session.run(next) == False
        assert tf_session.run(next) == False
        assert tf_session.run(next) == False
        assert tf_session.run(next) == False

def test_coord_feed(tf_session, time_feed):
    with tf_session.graph.as_default():
        X1 = tf.linspace(0., 1., 50)[:, None]
        X2 = tf.linspace(0., 1., 5)[:, None]
        coord_feed = CoordinateFeed(time_feed, X1, X2)
        init, next = init_feed(coord_feed)
        tf_session.run(init)
        out,N,slice_size = tf_session.run([next, coord_feed.N, coord_feed.slice_size])
        assert out.shape[0] == slice_size*50*5


def test_data_feed(tf_session, index_feed):
    with tf_session.graph.as_default():
        shape1 = (5, 2, 3, 4)
        shape2 = (5, 2, 3, 4)
        data1 = tf.ones(shape1)
        data2 = tf.ones(shape2)
        data_feed = DataFeed(index_feed, data1, data2)
        init, next = init_feed(data_feed)
        tf_session.run(init)
        out,N_slice,D, slice_size = tf_session.run([next, data_feed.N_slice, data_feed.D, data_feed.slice_size])
        for o in out:
            # assert o.shape[-1] == D
            # assert o.shape[0] == N_slice
            assert o.shape == (slice_size*6, 4)

        shape1 = (5, 2, 3, 4)
        shape2 = (5, 2, 3, 4)
        data1 = tf.ones(shape1)
        data2 = tf.ones(shape2)
        data_feed = DataFeed(index_feed, data1, data2, event_size=2)
        init, next = init_feed(data_feed)
        tf_session.run(init)
        out, N_slice, D, slice_size = tf_session.run([next, data_feed.N_slice, data_feed.D, data_feed.slice_size])
        for o in out:
            # assert o.shape[-1] == D
            # assert o.shape[0] == N_slice
            assert o.shape == (slice_size * 2, 3, 4)

def test_coord_dim_feed(tf_session):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(2)
        times = tf.linspace(0., 10., 3)[:, None]
        time_feed = TimeFeed(index_feed, times)
        X1 = tf.linspace(0., 1., 50)[:, None]
        X2 = tf.linspace(0., 1., 5)[:, None]
        coord_feed = CoordinateFeed(time_feed, X1, X2)
        dim_feed = CoordinateDimFeed(coord_feed)
        init, next = init_feed(dim_feed)
        tf_session.run(init)
        assert np.all(tf_session.run(next) == (2, 50, 5))
        assert np.all(tf_session.run(next) == (1, 50, 5))


###
# misc tests

def test_plotdatapack():
    # dp = DatapackPlotter(datapack='/home/albert/git/bayes_tec/scripts/data/P126+65_compact_full_raw.h5')
    # dp.plot(ant_sel=None, pol_sel=slice(0,1,1), time_sel=slice(0,1,1), fignames=['test_fig.png'], solset='sol000', observable='phase', plot_facet_idx=True,
    #         labels_in_radec=True, show=False)


   animate_datapack('/home/albert/git/bayes_tec/scripts/data/P126+65_compact_full_raw.h5',
           '/home/albert/git/bayes_tec/scripts/data/figures',num_processes=12,observable='phase',labels_in_radec=True,
                    solset='sol000', plot_facet_idx=True,)

def test_random_sample(tf_session):
    with tf_session.graph.as_default():
        t = tf.ones((6,5,8))
        assert tf_session.run(random_sample(t)).shape == (6,5,8)
        assert tf_session.run(random_sample(t,3)).shape == (3, 5, 8)
        assert tf_session.run(random_sample(t, 9)).shape == (6, 5, 8)


def test_flatten_batch_dims(tf_session):
    with tf_session.graph.as_default():
        t = tf.ones((1,2,3,4))
        f = flatten_batch_dims(t)
        assert tuple(tf_session.run(tf.shape(f))) == (6,4)

        t = tf.ones((1, 2, 3, 4))
        f = flatten_batch_dims(t,num_batch_dims=2)
        assert tuple(tf_session.run(tf.shape(f))) == (2, 3 , 4)

def test_load_array_file(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    lofar_cycle0_array = os.path.join(arrays, 'arrays/lofar.cycle0.hba.antenna.cfg')
    gmrt_array = os.path.join(arrays, 'arrays/gmrtPos.csv')
    lofar_array = load_array_file(lofar_array)
    lofar_cycle0_array = load_array_file(lofar_cycle0_array)
    gmrt_array = load_array_file(gmrt_array)
    assert (len(lofar_array[0]),3) == lofar_array[1].shape
    assert (len(lofar_cycle0_array[0]), 3) == lofar_cycle0_array[1].shape
    assert (len(gmrt_array[0]), 3) == gmrt_array[1].shape

def test_timer(tf_session):
    with tf_session.graph.as_default():
        t0 = timer()
        with tf.control_dependencies([t0]):
            t1 = timer()
        t0, t1 = tf_session.run([t0,t1])
        assert t1 > t0

def test_diagonal_jitter(tf_session):
    with tf_session.graph.as_default():
        j = diagonal_jitter(5)
        assert np.all(tf_session.run(j) == jitter*np.eye(5))

def test_log_normal_solve_fwhm():
    mu, stddev = log_normal_solve_fwhm(np.exp(1), np.exp(2), np.exp(-1))
    assert np.isclose(stddev**2, 0.5 * (0.5) ** 2)
    assert np.isclose(mu, 3./2. + 0.5 * (0.5) ** 2)


###
# frames

def test_enu(lofar_array):
    antennas = lofar_array[1]
    obstime = at.Time("2018-01-01T00:00:00.000", format='isot')
    location = ac.ITRS(x=antennas[0,0] * dist_type, y=antennas[0,1] * dist_type, z=antennas[0,2] * dist_type)
    enu = ENU(obstime=obstime,location=location.earth_location)
    altaz = ac.AltAz(obstime=obstime,location=location.earth_location)
    lofar_antennas = ac.ITRS(x=antennas[:, 0] * dist_type, y=antennas[:, 1] * dist_type, z=antennas[:, 2] * dist_type, obstime=obstime)
    assert np.all(np.linalg.norm(lofar_antennas.transform_to(enu).cartesian.xyz.to(dist_type).value, axis=0) < 100.)
    assert np.all(np.isclose(lofar_antennas.transform_to(enu).cartesian.xyz.to(dist_type).value,
                             lofar_antennas.transform_to(enu).transform_to(altaz).transform_to(enu).cartesian.xyz.to(
                                 dist_type).value))
    assert np.all(np.isclose(lofar_antennas.transform_to(altaz).cartesian.xyz.to(dist_type).value,
                             lofar_antennas.transform_to(altaz).transform_to(enu).transform_to(altaz).cartesian.xyz.to(
                                 dist_type).value))
    north_enu = ac.SkyCoord(east=0.,north=1., up=0.,frame=enu)
    north_altaz = ac.SkyCoord(az=0*au.deg, alt=0*au.deg, distance=1.,frame=altaz)
    assert np.all(np.isclose(
        north_enu.transform_to(altaz).cartesian.xyz.value, north_altaz.cartesian.xyz.value))
    assert np.all(np.isclose(
        north_enu.cartesian.xyz.value, north_altaz.transform_to(enu).cartesian.xyz.value))
    east_enu = ac.SkyCoord(east=1., north=0., up=0., frame=enu)
    east_altaz = ac.SkyCoord(az=90 * au.deg, alt=0 * au.deg, distance=1., frame=altaz)
    assert np.all(np.isclose(
        east_enu.transform_to(altaz).cartesian.xyz.value, east_altaz.cartesian.xyz.value))
    assert np.all(np.isclose(
        east_enu.cartesian.xyz.value, east_altaz.transform_to(enu).cartesian.xyz.value))
    up_enu = ac.SkyCoord(east=0., north=0., up=1., frame=enu)
    up_altaz = ac.SkyCoord(az=0 * au.deg, alt=90 * au.deg, distance=1., frame=altaz)
    assert np.all(np.isclose(
        up_enu.transform_to(altaz).cartesian.xyz.value, up_altaz.cartesian.xyz.value))
    assert np.all(np.isclose(
        up_enu.cartesian.xyz.value, up_altaz.transform_to(enu).cartesian.xyz.value))
    ###
    # dimensionful
    north_enu = ac.SkyCoord(east=0.*dist_type, north=1.*dist_type, up=0.*dist_type, frame=enu)
    north_altaz = ac.SkyCoord(az=0 * au.deg, alt=0 * au.deg, distance=1.*dist_type, frame=altaz)
    assert np.all(np.isclose(
        north_enu.transform_to(altaz).cartesian.xyz.to(dist_type).value, north_altaz.cartesian.xyz.to(dist_type).value))
    assert np.all(np.isclose(
        north_enu.cartesian.xyz.to(dist_type).value, north_altaz.transform_to(enu).cartesian.xyz.to(dist_type).value))
    east_enu = ac.SkyCoord(east=1.*dist_type, north=0.*dist_type, up=0.*dist_type, frame=enu)
    east_altaz = ac.SkyCoord(az=90 * au.deg, alt=0 * au.deg, distance=1.*dist_type, frame=altaz)
    assert np.all(np.isclose(
        east_enu.transform_to(altaz).cartesian.xyz.to(dist_type).value, east_altaz.cartesian.xyz.to(dist_type).value))
    assert np.all(np.isclose(
        east_enu.cartesian.xyz.to(dist_type).value, east_altaz.transform_to(enu).cartesian.xyz.to(dist_type).value))
    up_enu = ac.SkyCoord(east=0.*dist_type, north=0.*dist_type, up=1.*dist_type, frame=enu)
    up_altaz = ac.SkyCoord(az=0 * au.deg, alt=90 * au.deg, distance=1.*dist_type, frame=altaz)
    assert np.all(np.isclose(
        up_enu.transform_to(altaz).cartesian.xyz.to(dist_type).value, up_altaz.cartesian.xyz.to(dist_type).value))
    assert np.all(np.isclose(
        up_enu.cartesian.xyz.to(dist_type).value, up_altaz.transform_to(enu).cartesian.xyz.to(dist_type).value))


###
# Coord transforms

def test_itrs_to_enu_6D(tf_session, time_feed, lofar_array):
    # python test
    times = np.arange(2)[:,None]
    directions = np.random.normal(0,0.1, size=(10,2))
    antennas = lofar_array[1]
    X = make_coord_array(times, directions, antennas,flat=False)
    out = np.array(list(map(itrs_to_enu_6D,X)))
    assert out.shape == (2,10,antennas.shape[0],7)
    #TF test
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = tf.linspace(obstime_init.mjd * 86400., obstime_init.mjd * 86400. + 100., 9)[:, None]
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(lofar_array[1], dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_6D))
        init, next = init_feed(coord_feed)
        tf_session.run(init)
        out, N, slice_size = tf_session.run([next, coord_feed.N, coord_feed.slice_size])
        assert out.shape[0] == slice_size * 4 * len(lofar_array[0])
        assert out.shape[1] == 7
        assert np.all(np.isclose(np.linalg.norm(out[:,1:4],axis=1), 1.))
        assert np.all(np.isclose(np.linalg.norm(out[:, 4:7], axis=1) < 100., 1.))

def test_itrs_to_enu_with_references(tf_session, time_feed, lofar_array):
    # python test
    times = np.arange(2)[:,None]
    directions = np.random.normal(0,0.1, size=(10,2))
    antennas = lofar_array[1]
    X = make_coord_array(times, directions, antennas,flat=False)
    out = np.array(list(map(itrs_to_enu_6D,X)))
    assert out.shape == (2,10,antennas.shape[0],7)
    #TF test
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = tf.linspace(obstime_init.mjd * 86400., obstime_init.mjd * 86400. + 100., 9)[:, None]
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(lofar_array[1], dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        init, next = init_feed(coord_feed)
        tf_session.run(init)
        out, N, slice_size = tf_session.run([next, coord_feed.N, coord_feed.slice_size])
        print(out[0,:])
        assert out.shape[0] == slice_size * 4 * len(lofar_array[0])
        assert out.shape[1] == 13
        assert np.all(np.isclose(np.linalg.norm(out[:,1:4],axis=1), 1.))
        assert np.all(np.isclose(np.linalg.norm(out[:, 4:7], axis=1) < 100., 1.))
        assert np.all(np.isclose(np.linalg.norm(out[:, 10:13], axis=1), 1.))
        assert np.all(np.isclose(np.linalg.norm(out[:, 7:10], axis=1) < 100., 1.))

###
# Parameters

def test_constrained_bijector(tf_session):
    with tf_session.graph.as_default():
        x = tf.constant(-10., dtype=float_type)
        b = ConstrainedBijector(-1., 2.)
        y = b.forward(x)
        assert -1. < tf_session.run(y) < 2.
        x = tf.constant(10., dtype=float_type)
        b = ConstrainedBijector(-1., 2.)
        y = b.forward(x)
        assert -1. < tf_session.run(y) < 2.

def test_positive_lowerbound_bijector(tf_session):
    with tf_session.graph.as_default():
        x = tf.constant(-10., dtype=float_type)
        b = ScaledLowerBoundedBijector(2., 5.)
        y = b.forward(x)
        assert 2. < tf_session.run(y)

def test_spherical_to_cartesian_bijector(tf_session):
    with tf_session.graph.as_default():
        sph = tf.constant([[1.,np.pi/2.,np.pi/2.]],dtype=float_type)
        car = tf.constant([[0., 1., 0.]],dtype=float_type)
        b = SphericalToCartesianBijector()
        assert np.linalg.norm(tf_session.run(b.forward(sph))) == 1.
        assert np.all(np.isclose(tf_session.run(b.forward(sph)), tf_session.run(car)))
        assert np.all(np.isclose(tf_session.run(b.inverse(tf_session.run(car))), tf_session.run(sph)))

def test_parameter(tf_session):
    with tf_session.graph.as_default():
        p = Parameter(constrained_value=10.)
        assert tf_session.run(p.constrained_value) == tf_session.run(p.unconstrained_value)

def test_scaled_positive(tf_session):
    with tf_session.graph.as_default():
        b = ScaledPositiveBijector(10.)
        assert tf_session.run(b.inverse(tf.constant(100.,float_type))) == np.log(100./10.)

def test_scaled(tf_session):
    with tf_session.graph.as_default():
        b = ScaledBijector(10.)
        assert tf_session.run(b.inverse(tf.constant(100.,float_type))) == 100./10.

###
# Kernels

def test_histogram(tf_session):
    with tf_session.graph.as_default():
        
        heights = tf.constant(np.arange(50)+1,dtype=float_type)
        edgescales = tf.constant(np.arange(51)+1, dtype=float_type)
        kern = Histogram(heights, edgescales)
        x = tf.constant(np.linspace(0.,10.,100)[:,None],dtype=float_type)

        h,e = tf_session.run([heights, edgescales])
        for i in range(50):
            plt.bar(0.5*(1./e[i+1]+1./e[i]),h[i],1./e[i+1]-1./e[i])
        plt.savefig("test_histogram_spectrum.png")
        plt.close('all')

        K = kern.matrix(x,x)
        K,L = tf_session.run([K,safe_cholesky(K)])
        plt.imshow(K)
        plt.colorbar()
        plt.savefig("test_histogram.png")
        plt.close("all")
        y = np.dot(L, np.random.normal(size=(100,3)))
        plt.plot(np.linspace(0,10.,100),y)
        plt.savefig("test_histogram_sample.png")
        plt.close("all")

        x0 = tf.constant([[0.]],dtype=float_type)
        K_line = kern.matrix(x, x0)
        K_line = tf_session.run(K_line)
        plt.plot(np.linspace(0,10,100),K_line[:,0])
        plt.savefig("test_histogram_kline.png")
        plt.close('all')


def test_isotropic_time_general(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 50., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1,:], lofar_array[1]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        init, next = init_feed(coord_feed)

        index_feed = IndexFeed(2)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd * 86400. + tf.cast(tf.linspace(0., 50., 9)[:, None], float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1, :], lofar_array[1]], axis=0), dtype=float_type)
        coord_feed2 = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        dim_feed = CoordinateDimFeed(coord_feed)
        init2, next2 = init_feed(coord_feed2)
        init3, next3 = init_feed(dim_feed)

        kern = DTECIsotropicTimeGeneral(variance=5e-4,timescale=30.,lengthscales=20., a=250., b=100., fed_kernel='RBF',obs_type='DDTEC',resolution=3, squeeze=True)

        tf_session.run([init, init2, init3])
        K1, K2, dims = tf_session.run([kern.K(next), kern.K(next, next2), next3])
        import pylab as plt
        plt.imshow(K1)
        plt.colorbar()
        plt.show()
        plt.imshow(K2)
        plt.colorbar()
        plt.show()

        L = np.linalg.cholesky(K1 + 1e-6*np.eye(K1.shape[-1]))
        ddtecs = np.einsum("ab,bc->ac",L, np.random.normal(size=L.shape)).reshape(list(dims)+[L.shape[0]])
        print(ddtecs[:,:,51].var(), 0.01**2/ddtecs[:,:,51].var())

def test_ddtec_screen(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 50., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        M = 20
        ra_vec = np.linspace(np.pi / 4 - 2.*np.pi/180., np.pi / 4 + 2.*np.pi/180., M)
        dec_vec = np.linspace(np.pi / 4 - 2. * np.pi / 180., np.pi / 4 + 2. * np.pi / 180., M)
        ra, dec = np.meshgrid(ra_vec, dec_vec, indexing='ij')
        ra = ra.flatten()[:,None]
        dec = dec.flatten()[:,None]
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([ lofar_array[1][50:51,:]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        init, next = init_feed(coord_feed)
        kern = DTECIsotropicTimeGeneral(variance=0.17,timescale=30.,lengthscales=15., a=300., b=90.,
                                           fed_kernel='RBF',obs_type='DDTEC',squeeze=True,
                                           kernel_params={'resolution':5})
        K = kern.K(next)
        tf_session.run([init])
        K = tf_session.run(K)

        s = np.mean(np.diag(K))
        K /= s

        L = np.sqrt(s)*np.linalg.cholesky(K + 1e-6*np.eye(K.shape[-1]))
        ddtecs = np.einsum("ab,b->a",L, np.random.normal(size=L.shape[0])).reshape((M,M))
        import pylab as plt
        plt.imshow(np.sin(-8.448e6/140e6*ddtecs))
        plt.colorbar()
        plt.savefig("sim_ddtecs.png")
        plt.show()

def test_isotropic_time_general_ode(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 50., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1,:], lofar_array[1]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        init, next = init_feed(coord_feed)
        kern = DTECIsotropicTimeGeneralODE(variance=5e-2,timescale=30.,lengthscales=18., a=300., b=90.,
                                           fed_kernel='RBF',obs_type='DTEC',squeeze=True,
                                           ode_type='fixed',kernel_params={'rtol':1e-6,'atol':1e-6})
        K, info = kern.K(next, full_output=True)
        tf_session.run([init])
        [K] = tf_session.run([K])
        print(info)
        import pylab as plt
        plt.imshow(K)
        plt.colorbar()
        plt.show()


        # L = np.linalg.cholesky(K1 + 1e-6*np.eye(K1.shape[-1]))
        # ddtecs = np.einsum("ab,bc->ac",L, np.random.normal(size=L.shape)).reshape(list(dims)+[L.shape[0]])
        # print(ddtecs[:,:,51].var(), 0.01**2/ddtecs[:,:,51].var())

def test_kernel_equivalence(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 50., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1,:], lofar_array[1]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        init, next = init_feed(coord_feed)
        kernels = [DTECIsotropicTimeGeneralODE(variance=2e11**2, timescale=30., lengthscales=10., a=300., b=90.,
                   fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                   ode_type='adaptive', kernel_params={'rtol':1e-6, 'atol':1e-6}),
                   DTECIsotropicTimeGeneralODE(variance=1e9**2,timescale=30.,lengthscales=10., a=300., b=90.,
                                           fed_kernel='RBF',obs_type='DDTEC',squeeze=True,
                                           ode_type='fixed',kernel_params={'resolution':5}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 4}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 6}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 8}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 10}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 12}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 14}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 16}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 18})
                   ]
        K = [kern.K(next) for kern in kernels]
        tf_session.run([init])
        K = tf_session.run(K)
        print(np.sqrt(np.mean(np.diag(K[0]))))
        #1/m^3 km
        for k in K:
            print(np.mean(np.abs(K[1] - k)/np.mean(np.diag(K[1]))))
        # plt.imshow(K[0])
        # plt.colorbar()
        # plt.show()
        # plt.imshow(K[1])
        # plt.colorbar()
        # plt.show()
        # plt.imshow(K[2])
        # plt.colorbar()
        # plt.show()

def test_K_parts(tf_session, lofar_array):
    with tf_session.graph.as_default():
        Xt = np.linspace(0., 100., 2)[:, None]
        Xd = np.random.normal(size=(5, 3))
        Xd /= np.linalg.norm(Xd, axis=1, keepdims=True)
        Xa = 10. * np.random.normal(size=(6, 3))
        Xa[0:2, :] = 0.

        kern = DTECIsotropicTimeGeneral(variance=0.07, timescale=30., lengthscales=5., fed_kernel='RBF', obs_type='DDTEC',
                                 resolution=5, squeeze=True)

        X = make_coord_array(Xt, Xd, Xa)
        K = tf_session.run(kern.K(X, (2,5,6), X, (2,5,6)))

        X = make_coord_array(Xt[0:1,:], Xd, Xa)
        X0 = make_coord_array(Xt[1:2, :], Xd, Xa)

        K00 = tf_session.run(kern.K(X, (1, 5, 6), X, (1, 5, 6)))
        K11 = tf_session.run(kern.K(X0, (1, 5, 6), X0, (1, 5, 6)))
        K01 = tf_session.run(kern.K(X, (1, 5, 6), X0, (1, 5, 6)))
        K10 = tf_session.run(kern.K(X0, (1, 5, 6), X, (1, 5, 6)))

        K_np = np.concatenate([np.concatenate([K00, K01],axis=-1), np.concatenate([K10, K11],axis=-1)],axis=-2)

        assert np.all(K_np==K)
        import pylab as plt

        K_ = tf_session.run(K_parts(kern, [X, X0], [(1, 5, 6),(1, 5, 6)]))
        # plt.imshow(K-K_)
        # plt.show()
        # plt.imshow(K - K.T)
        # plt.show()
        # plt.imshow(K_ - K_.T)
        # plt.show()
        assert np.all(np.isclose(K_, K))
        assert K_.shape == (1*4*5*2, 1*4*5*2)

        X = make_coord_array(Xt, Xd, Xa)
        X0 = make_coord_array(Xt, Xd, Xa[1:,:])
        K = tf_session.run(kern.K(X, (2, 5, 6), X0, (2, 5, 5)))

        X = make_coord_array(Xt[0:1, :], Xd, Xa)
        X0 = make_coord_array(Xt[1:2, :], Xd, Xa)

        K00 = tf_session.run(kern.K(make_coord_array(Xt[0:1, :], Xd, Xa), (1, 5, 6), make_coord_array(Xt[0:1, :], Xd, Xa[1:,:]), (1, 5, 5)))
        K11 = tf_session.run(kern.K(make_coord_array(Xt[1:2, :], Xd, Xa), (1, 5, 6), make_coord_array(Xt[1:2, :], Xd, Xa[1:,:]), (1, 5, 5)))
        K01 = tf_session.run(kern.K(make_coord_array(Xt[0:1, :], Xd, Xa), (1, 5, 6), make_coord_array(Xt[1:2, :], Xd, Xa[1:,:]), (1, 5, 5)))
        K10 = tf_session.run(kern.K(make_coord_array(Xt[1:2, :], Xd, Xa), (1, 5, 6), make_coord_array(Xt[0:1, :], Xd, Xa[1:,:]), (1, 5, 5)))

        K_np = np.concatenate([np.concatenate([K00, K01], axis=-1), np.concatenate([K10, K11], axis=-1)], axis=-2)

        assert np.all(K_np == K)

        # X = make_coord_array(Xt[0:1, :], Xd, Xa)
        # X0 = make_coord_array(Xt[1:2, :], Xd, Xa[1:,:])
        #
        # K_ = tf_session.run(K_parts(kern,[X, X0],[(1,5,6),(1,5,5)]))
        # assert np.all(np.isclose(K, K_))

        # kern = DTECIsotropicTimeLong(variance=0.07, timescale=30., lengthscales=5., fed_kernel='RBF', obs_type='DTEC',
        #                          resolution=5, squeeze=True)

        # K00 = tf_session.run(
        #     kern.K(make_coord_array(Xt[0:1, :], Xd, Xa), (1, 5, 6), make_coord_array(Xt[0:1, :], Xd, Xa[1:, :]),
        #            (1, 5, 5)))
        # K11 = tf_session.run(
        #     kern.K(make_coord_array(Xt[1:2, :], Xd, Xa), (1, 5, 6), make_coord_array(Xt[1:2, :], Xd, Xa[1:, :]),
        #            (1, 5, 5)))
        # K01 = tf_session.run(
        #     kern.K(make_coord_array(Xt[0:1, :], Xd, Xa), (1, 5, 6), make_coord_array(Xt[1:2, :], Xd, Xa[1:, :]),
        #            (1, 5, 5)))
        # K10 = tf_session.run(
        #     kern.K(make_coord_array(Xt[1:2, :], Xd, Xa), (1, 5, 6), make_coord_array(Xt[0:1, :], Xd, Xa[1:, :]),
        #            (1, 5, 5)))
        #
        # K_np = np.concatenate([np.concatenate([K00, K01], axis=-1), np.concatenate([K10, K11], axis=-1)], axis=-2)






        # obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        # times = obstime_init.mjd * 86400. + tf.cast(tf.linspace(0., 50., 9)[:, None], float_type)
        # ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(2, 1))
        # dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(2, 1))
        # Xd = tf.concat([ra, dec], axis=1)
        # Xa = tf.constant(np.concatenate([lofar_array[1][0:1, :], lofar_array[1]], axis=0), dtype=float_type)
        #
        # index_feed = IndexFeed(2)
        # time_feed = TimeFeed(index_feed, times)
        # coord_feed = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_6D))
        # init, next = init_feed(coord_feed)
        # index_feed = IndexFeed(3)
        # time_feed = TimeFeed(index_feed, times)
        # coord_feed2 = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_6D))
        # init2, next2 = init_feed(coord_feed2)

        # kern = DTECIsotropicTime(variance=0.07,timescale=30.,lengthscales=5., fed_kernel='RBF',obs_type='DDTEC',resolution=5, squeeze=True)
        # tf_session.run([init, init2])
        # K = tf_session.run(K_parts(kern, [next, next2], [coord_feed.dims, coord_feed2.dims]))
        #
        # tf_session.run([init, init2])
        # K00 = tf_session.run(kern.K(next, coord_feed.dims))
        # K11 = tf_session.run(kern.K(next2, coord_feed2.dims))
        # tf_session.run([init, init2])
        # K01 = tf_session.run(kern.K(next, coord_feed.dims, next2, coord_feed2.dims))
        # K_np = np.concatenate([np.concatenate([K00, K01],axis=-1),np.concatenate([K01.T, K11],axis=-1)],axis=0)
        # print(np.isclose(K,K_np))
        #
        # import pylab as plt
        # plt.imshow(K-K_np)
        # plt.show()
        # # assert np.all(np.isclose(tf_session.run(K_parts(kern, [next, next], [coord_feed.dims, coord_feed.dims])),
        # #                          np.concatenate([np.concatenate([K,K],axis=2), np.concatenate([np.transpose(K,(0,2,1)),K],axis=2)])))

def test_hmc_matrix_stepsizes(tf_session):
    with tf_session.graph.as_default():

        X = tf.cast(tf.linspace(0.,10.,100), float_type)[:,None]

        kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=tf.convert_to_tensor(1.,float_type),
                                                               length_scale=tf.convert_to_tensor(1.,float_type),
                                                               feature_ndims=1)
        K = kern.matrix(X, X)
        L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))
        Ftrue = 0.01*tf.matmul(L, tf.random_normal(shape=[tf.shape(K)[0], 1],dtype=float_type))
        invfreqs = -8.448e9*tf.reciprocal(tf.cast(tf.linspace(100e6, 160e6, 24),float_type))
        Ytrue = tf.sin(Ftrue*invfreqs)
        Y = Ytrue + 0.3*tf.random_normal(shape=tf.shape(Ytrue),dtype=float_type)

        outliers = np.zeros((100, 24))
        outliers[np.random.choice(100, size=10, replace=False),1:-1] = 3.
        Y += tf.constant(outliers, float_type)
        Ytrue_cos = tf.cos(Ftrue * invfreqs)
        Y_cos = Ytrue_cos + 0.3 * tf.random_normal(shape=tf.shape(Ytrue_cos), dtype=float_type)

        a = tf.Variable(0., dtype=float_type)
        l = tf.Variable(0., dtype=float_type)

        def logp(y_sigma, f):
            # l = tf.get_variable('l', initializer=lambda: tf.constant(0., dtype=float_type))
            # a = tf.get_variable('a', initializer=lambda: tf.constant(0., dtype=float_type))
            kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=2. * tf.exp(a),
                                                                   length_scale=2. * tf.exp(l),
                                                                   feature_ndims=1)
            K = kern.matrix(X, X)
            L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))
            # kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=tf.convert_to_tensor(1.,float_type),
            #                                                             length_scale=tf.convert_to_tensor(1.,float_type),feature_ndims=1)
            # K = kern.matrix(X,X)
            # L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))
            prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f), scale_identity_multiplier=1.)
            # prior = tfp.distributions.MultivariateNormalTriL(scale_tril=L[None, :, :])
            Y_model = tf.sin(tf.matmul(L[None, :, :],f[:,:,None])*invfreqs)
            Y_model_cos = tf.cos(tf.matmul(L[None, :, :], f[:, :, None]) * invfreqs)

            likelihood = tfp.distributions.Laplace(loc = Y_model, scale=y_sigma)
            likelihood_cos = tfp.distributions.Laplace(loc=Y_model_cos, scale=y_sigma)
            logp = tf.reduce_sum(likelihood.log_prob(Y[None, :, :]), axis=[1,2]) + tf.reduce_sum(likelihood_cos.log_prob(Y_cos[None, :, :]), axis=[1,2]) +  prior.log_prob(f)
            # logp.set_shape(tf.TensorShape([]))
            return logp

        step_size = tf.get_variable(
                        name='step_size',
                        initializer=lambda: tf.constant(0.001, dtype=float_type),
                        use_resource=True,
                        dtype=float_type,
                        trainable=False),


        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=logp,
            num_leapfrog_steps=3,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
            step_size=step_size,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=None,
                                                                             target_rate=0.75),
            state_gradients_are_stopped=True)
        #                         step_size_update_fn=lambda v, _: v)

        q0 = [tf.constant([0.01],float_type), 0.*Ftrue[None, :, 0]]

        # Run the chain (with burn-in).
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=1000,
            num_burnin_steps=0,
            current_state=q0,
            kernel=hmc,
            parallel_iterations=5)

        avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                                                         name='avg_acc_ratio')
        posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                                       name='marginal_log_likelihood')

        saem_opt = tf.train.AdamOptimizer(1e-3).minimize(-posterior_log_prob,var_list=[a,l])

        tf_session.run(tf.global_variables_initializer())
        for i in range(100):
            _, loss = tf_session.run([saem_opt, posterior_log_prob])
            print(i, loss)





        with tf.control_dependencies([saem_opt]):
            kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=2. * tf.exp(a),
                                                                            length_scale=2. * tf.exp(l),
                                                                            feature_ndims=1)
            K = kern.matrix(X, X)
            L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))

        samples = [samples[0], tf.einsum("ab,snb->sna", L, samples[1])[...,None], 2.*tf.exp(a), 2.*tf.exp(l)]


        invfreqs,times, logp0, Y, Ytrue, Ftrue, samples,step_sizes, ess, chain_logp = tf_session.run(
            [invfreqs, X[:,0], logp(*q0),Y,Ytrue,Ftrue,samples,
              kernel_results.extra.step_size_assign,
              tfp.mcmc.effective_sample_size(samples[0:2]),
              tf.reduce_mean(kernel_results.accepted_results.target_log_prob,name='marginal_log_likelihood')])
        print(logp0)
        print(chain_logp)
        print(ess)
        import pylab as plt
        import seaborn as sns
        plt.style.use('ggplot')
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

        ax1.plot(times, Ftrue[:, 0])
        ax1.plot(times, samples[1][:,0,:,0].mean(0))
        ax1.fill_between(times, samples[1][:,0,:,0].mean(0) - samples[1][:,0,:,0].std(0), samples[1][:,0,:,0].mean(0) + samples[1][:,0,:,0].std(0), alpha=0.5)
        ax1.legend()
        ax1.set_title("Model space solution vs true")

        Ypost = [np.sin(samples[1][:,0,:,:] *invfreqs).mean(0), np.sin(samples[1][:,0,:,:] *invfreqs).std(0)]
        ax2.plot(times, Ytrue, label='Ytrue')
        ax2.plot(times, Ypost[0], label='posterior')
        [ax2.fill_between(times, Ypost[0][...,i] - Ypost[1][...,i] , Ypost[0][...,i]  + Ypost[1][...,i] , alpha=0.5) for i in range(24)]
        ax2.set_title("Data space solution vs true")
        ax2.legend()

        ax3.plot(times, Y, label='Ytrue')
        ax3.plot(times, Ypost[0], label='posterior')
        [ax3.fill_between(times, Ypost[0][...,i] - Ypost[1][...,i] , Ypost[0][...,i]  + Ypost[1][...,i] , alpha=0.5) for i in range(24)]
        ax3.legend()
        ax3.set_title("Data space solution vs data")

        print(samples[2:4])
        sns.kdeplot(samples[0].flatten(), ax=ax4,shade=True, alpha=0.5)
        # ax4.plot(step_sizes[0], label='y_sigma stepsize')
        # ax4.plot(step_sizes[1], label='dtec stepsize')
        # ax4.set_yscale("log")
        # ax4.set_title("stepsizes")
        ax4.legend()
        plt.show()

###
# logprobabilities

def test_DTECToGains(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(9)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd * 86400. + tf.cast(tf.linspace(0., 50., 9)[:, None], float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(2, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(2, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1, :], lofar_array[1]], axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_6D))
        init, next = init_feed(coord_feed)
        tf_session.run(init)
        # DTECToGains()


###
# filters

@pytest.fixture
def simulated_ddtec(tf_session, lofar_array2):
    lofar_array = lofar_array2
    class Simulated:
        def __init__(self):
            with tf_session.graph.as_default():
                index_feed = IndexFeed(2)
                obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
                times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 9*30., 3)[:, None],float_type)
                time_feed = TimeFeed(index_feed, times)
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
                tf_session.run(init)
                kern = DTECIsotropicTimeGeneral(variance=0.07,timescale=30.,lengthscales=15., a=250., b=50.,
                                         fed_kernel='RBF',obs_type='DDTEC',resolution=5, squeeze=False)
                from timeit import default_timer
                t0 = default_timer()
                Y_real, Y_imag = [],[]
                while True:
                    try:
                        K = tf_session.run(kern.K(next))
                    except:
                        break
                    L = np.linalg.cholesky(K+1e-6*np.eye(K.shape[1]))
                    ddtec = np.einsum('ab,b->a',L[0,:,:], np.random.normal(size=L.shape[1]))
                    freqs = np.linspace(110.e6, 160.e6, 10)
                    Y_real.append(np.cos(-8.448e9*ddtec[:,None]/freqs))
                    Y_imag.append(np.sin(-8.448e9 * ddtec[:, None] / freqs))
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



def test_free_transition(tf_session, simulated_ddtec):
    with tf_session.graph.as_default():

        pass

def test_dblquad(tf_session):
    with tf_session.graph.as_default():
        def func(t1,t2):
            return tf.tile(tf.reshape((-tf.square(t1-t2)), (1,)), [50])
        l = tf.constant(0., float_type)
        u = tf.constant(1., float_type)
        I, info = dblquad(func, l, u,lambda t:l, lambda t:u, (50,))
        print(tf_session.run([I,info]))

        assert np.all(np.isclose(tf_session.run(I), -1/6.))


def test_plot_vornoi_map():
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    points = np.random.uniform(size=(5,2))
    colors = np.random.uniform(size=5)
    plot_vornoi_map(points, colors, ax=ax, alpha=1.,radius=100)
    plt.show()