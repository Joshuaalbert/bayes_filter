import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys, os
from .data_feed import CoordinateFeed, TimeFeed, IndexFeed, DataFeed, init_feed
from .misc import flatten_batch_dims, load_array_file, make_coord_array, timer, diagonal_jitter
from .coord_transforms import itrs_to_enu_6D, tf_coord_transform
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at
from .frames import ENU
from .settings import dist_type, float_type, jitter
from .parameters import SphericalToCartesianBijector, ConstrainedBijector, ScaledPositiveBijector, ScaledBijector, Parameter
from .kernels import DTECFrozenFlow


@pytest.fixture
def tf_graph():
    return tf.Graph()


@pytest.fixture
def tf_session(tf_graph):
    return tf.Session(graph=tf_graph)

@pytest.fixture
def arrays():
    return os.path.dirname(sys.modules["bayes_filter"].__file__)

@pytest.fixture
def lofar_array(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    return load_array_file(lofar_array)


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
            assert o.shape == (2*6, 4)

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
            assert o.shape == (2 * 2, 3, 4)



###
# misc tests



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
        with tf.control_dependencies([timer()]):
            t1 = timer()
        t0, t1 = tf_session.run([t0,t1])
        assert t1 > t0

def test_diagonal_jitter(tf_session):
    with tf_session.graph.as_default():
        j = diagonal_jitter(5)
        assert np.all(tf_session.run(j) == jitter*np.eye(5))


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

def test_frozen_flow(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(9)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 100., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(2, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(2, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1,:], lofar_array[1]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_6D))
        init, next = init_feed(coord_feed)
        tf_session.run(init)
        kern = DTECFrozenFlow(variance=0.07,velocity=[0.1, 0., 0.],lengthscales=5., fed_kernel='M32',obs_type='DDTEC',resolution=5)
        from timeit import default_timer
        t0 = default_timer()
        K = tf_session.run(kern.K(next, coord_feed.dims))
        L = np.linalg.cholesky(K+1e-6*np.eye(K.shape[0]))
        assert True #pos def
        y = np.einsum('ab,bc->ac',L,np.random.normal(size=(K.shape[0],50)))
        # assert K.shape[0] == 2*40*63
        print(default_timer() - t0)
        import pylab as plt
        plt.imshow(K,cmap='jet')
        plt.colorbar()
        plt.show()
        y = y.reshape((9, 1,62, 50))
        plt.plot(y[:,0,51,:])
        plt.show()

def test_hmc_matrix_stepsizes(tf_session):
    with tf_session.graph.as_default():

        X = tf.cast(tf.linspace(0.,10.,1000), float_type)[:,None]

        kern = tfp.positive_semidefinite_kernels.MaternOneHalf(amplitude=tf.convert_to_tensor(1.,float_type),
                                                               length_scale=tf.convert_to_tensor(1.,float_type),
                                                               feature_ndims=1)
        K = kern.matrix(X, X)
        L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))
        Ftrue = 0.01*tf.matmul(L, tf.random_normal(shape=[tf.shape(K)[0], 1],dtype=float_type))
        invfreqs = -8.448e9*tf.reciprocal(tf.cast(tf.linspace(100e6, 160e6, 24),float_type))
        Ytrue = tf.sin(Ftrue*invfreqs)
        Y = Ytrue + 0.3*tf.random_normal(shape=tf.shape(Ytrue),dtype=float_type)

        outliers = np.zeros((1000,24))
        outliers[np.random.choice(1000,size=500,replace=False),:3] = 3.
        # Y += tf.constant(outliers, float_type)
        Ytrue_cos = tf.cos(Ftrue * invfreqs)
        Y_cos = Ytrue_cos + 0.3 * tf.random_normal(shape=tf.shape(Ytrue_cos), dtype=float_type)

        def logp(y_sigma, f):
            # kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=tf.convert_to_tensor(1.,float_type),
            #                                                             length_scale=tf.convert_to_tensor(1.,float_type),feature_ndims=1)
            # K = kern.matrix(X,X)
            # L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))
            prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f), scale_identity_multiplier=1.)
            # prior = tfp.distributions.MultivariateNormalTriL(scale_tril=L[None, :, :])
            Y_model = tf.sin(tf.matmul(L[None, :, :],f[:,:,None])*invfreqs)
            Y_model_cos = tf.cos(tf.matmul(L[None, :, :], f[:, :, None]) * invfreqs)

            likelihood = tfp.distributions.Normal(loc = Y_model, scale=y_sigma)
            likelihood_cos = tfp.distributions.Normal(loc=Y_model_cos, scale=y_sigma)
            logp = tf.reduce_sum(likelihood.log_prob(Y[None, :, :]), axis=[1,2]) + tf.reduce_sum(likelihood_cos.log_prob(Y_cos[None, :, :]), axis=[1,2]) +  prior.log_prob(f)
            return logp

        step_size = [tf.get_variable(
                        name='y_sigma_step_size',
                        initializer=lambda: tf.constant(0.0001, dtype=float_type),
                        use_resource=True,
                        dtype=float_type,
                        trainable=False),
                    tf.get_variable(
                        name='L_step_size',
                        initializer=lambda: tf.constant(0.003, dtype=float_type),#tf.linalg.inv(L),
                        use_resource=True,
                        dtype=float_type,
                        trainable=False)]

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
            num_burnin_steps=400,
            current_state=q0,
            kernel=hmc,
            parallel_iterations=5)

        print(samples)
        samples = [samples[0],tf.einsum("ab,scb->sca",L, samples[1])[..., None]]

        tf_session.run(tf.global_variables_initializer())
        invfreqs,times, logp0, Y, Ytrue, Ftrue, samples,step_sizes, ess, chain_logp = tf_session.run(
            [invfreqs, X[:,0], logp(*q0),Y,Ytrue,Ftrue,samples,
              kernel_results.extra.step_size_assign,
              tfp.mcmc.effective_sample_size(samples),
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

        sns.kdeplot(samples[0].flatten(), ax=ax4,shade=True, alpha=0.5)
        # ax4.plot(step_sizes[0], label='y_sigma stepsize')
        # ax4.plot(step_sizes[1], label='dtec stepsize')
        # ax4.set_yscale("log")
        # ax4.set_title("stepsizes")
        ax4.legend()
        plt.show()
