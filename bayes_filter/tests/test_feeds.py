from .common_setup import *
from bayes_filter.datapack import DataPack
from ..kernels import DTECIsotropicTimeGeneral
from .. import TEC_CONV
from bayes_filter.feeds import IndexFeed, init_feed, TimeFeed, ContinueFeed, CoordinateFeed, DataFeed, \
    CoordinateDimFeed, DatapackFeed, FreqFeed
from ..misc import make_example_datapack, maybe_create_posterior_solsets, make_coord_array
from ..coord_transforms import itrs_to_enu_with_references
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au

def test_freq_feed(tf_session):
    with tf_session.graph.as_default():
        freq_feed = FreqFeed([1.,2.])
        init, next = init_feed(freq_feed)
        tf_session.run(init)
        out = tf_session.run(next)
        assert np.all(out == np.array([1.,2.]))

def test_index_feed(tf_session):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(2,3)
        init, next = init_feed(index_feed)
        tf_session.run(init)
        out = tf_session.run(next)
        assert out[1] - out[0] == index_feed._step
        out = tf_session.run(next)
        assert out[1] - out[0] == 1
        try:
            out = tf_session.run(next)
            assert False
        except tf.errors.OutOfRangeError:
            assert True


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
        assert out.shape == (slice_size*50*5, 3)


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
        index_feed = IndexFeed(2,3)
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


def test_datapack_feed(tf_session):

    with tf_session.graph.as_default():
        res = make_example_datapack(4, 2, 2, pols=['XX'],clobber=True, gain_noise=0., name=os.path.join(TEST_FOLDER,'test_feed_data.h5'), return_full=True)
        datapack = res['datapack']
        index_feed = IndexFeed(1)
        patch_names, _ = datapack.directions
        _, screen_directions = datapack.get_directions(patch_names)
        maybe_create_posterior_solsets(datapack, solset='sol000', posterior_name='posterior',
                                       screen_directions=screen_directions)


        datapack_feed = DatapackFeed(datapack, solset='sol000',
                     postieror_name="posterior",
                     selection={'ant':None,'dir':None},
                                     index_n=1)
        init, ((Y_real, Y_imag), freqs, X, Xstar, X_dim, Xstar_dim, cont) = init_feed(datapack_feed)

        tf_session.run(init)
        ((Y_real, Y_imag), freqs, X, Xstar, X_dim, Xstar_dim, cont) = tf_session.run(((Y_real, Y_imag), freqs, X, Xstar, X_dim, Xstar_dim, cont))
        pred_phase = res['dtec'][0,:,:,0,None]*TEC_CONV/res['freqs']
        Y_imag_pred = np.sin(pred_phase)
        assert np.all(np.isclose(Y_imag_pred.reshape(Y_imag.shape), Y_imag))
        Y_real_pred = np.cos(pred_phase)
        assert np.all(np.isclose(Y_real_pred.reshape(Y_real.shape), Y_real))

        datapack.current_solset = 'sol000'
        datapack.select(time=slice(0,1,1))
        antenna_labels, _ = datapack.antennas
        _, antennas = datapack.get_antennas(antenna_labels)
