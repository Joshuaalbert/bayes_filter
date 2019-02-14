import tensorflow as tf
import numpy as np
from . import float_type
from .misc import flatten_batch_dims, make_coord_array
from .datapack import DataPack


def init_feed(feed):
    """
    Initialise a feed
    :param feed: Feed
    :return: tf.data.Iterator.initializer
    :return: tensor slices
    """
    feed = feed.feed
    iterator_tensor = feed.make_initializable_iterator()
    return iterator_tensor.initializer, iterator_tensor.get_next()




class Feed(object):
    def __init__(self):
        self._feed = None

    @property
    def feed(self):
        return self._feed

    @feed.setter
    def feed(self, value):
        if not isinstance(value, tf.data.Dataset):
            raise ValueError("Expected tf.data.Dataset not {}".format(type(value)))
        self._feed = value


class IndexFeed(Feed):
    def __init__(self, step=1):
        self._step = step
        self.step = tf.convert_to_tensor(step, tf.int32)
        self.index_feed = tf.data.Dataset.from_generator(self.index_generator,
                                                         (tf.int32, tf.int32),
                                                         (tf.TensorShape([]), tf.TensorShape([])))
        self.feed = self.index_feed

    def index_generator(self):
        i = 0
        while True:
            i += self._step
            yield i - self._step, i


class DataFeed(Feed):
    def __init__(self, index_feed:IndexFeed, *data, event_size=1, num_parallel_calls=10):
        """
        Create a time feed
        :param index_feed: IndexFeed
            Pulse of this feed
        :param data: list of float_type, Tensor, [Nt, ..., D]
            Data to grab unflattened
        :param num_parallel_calls:
        """
        self.num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, tf.int32)
        self.event_size = tf.convert_to_tensor(event_size, tf.int32)
        self.data = [tf.cast(c, float_type) if c.dtype is not float_type else c for c in data]
        self.index_feed = index_feed
        self.slice_size = self.index_feed.step
        with tf.control_dependencies([tf.assert_equal(tf.shape(d), tf.shape(self.data[0])) for d in self.data]):
            self.Nt = tf.shape(self.data[0])[0]
            self.D = tf.shape(self.data[0])[-1]
            self.N_slice = self.slice_size * tf.reduce_prod(tf.shape(self.data[0])[1:-1])
            self.N = tf.reduce_prod(tf.shape(self.data[0])[:-1])
        self.data_feed = self.index_feed.feed.map(self.get_data_block, num_parallel_calls=self.num_parallel_calls)
        self.feed = self.data_feed

    def get_data_block(self, index, next_index):
        """
        Get the time slice from index to index + step
        :param index: tf.int32, Tensor, scalar
            Index to start slice at
        :return: float_type, Tensor, [N, D]
            The returned data block
        """
        next_index = tf.minimum(next_index, self.Nt)
        indices = tf.range(index, next_index)
        data = [flatten_batch_dims(tf.gather(d, indices, axis=0),num_batch_dims=-self.event_size) for d in self.data]
        return data

class DatapackFeed(Feed):
    def __init__(self, index_feed:IndexFeed, datapack, *addresses, selection = {}, time_axis=2, perm=(2,1,0), event_size=1, num_parallel_calls=10):
        """
        Create a time feed
        :param index_feed: IndexFeed
            Pulse of this feed
        :param data: list of strings
            Addresses to pytable, [Nt, ..., D]
        :param time_axis: int
            The axis at addresses that correspond to time
        :param perm: tuple of int
            The permutation to perform before flattening
        :param num_parallel_calls: int
            How many threads can map.
        """
        self.num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, tf.int32)
        self.event_size = tf.convert_to_tensor(event_size, tf.int32)
        self.datapack = datapack
        self.selection = selection
        self.selection.pop('time', None)
        self.addresses = addresses# = [tf.cast(c, float_type) if c.dtype is not float_type else c for c in data]
        self.time_axis = time_axis
        self.perm = perm
        self.index_feed = index_feed
        self.slice_size = self.index_feed.step
        # with tf.control_dependencies([tf.assert_equal(tf.shape(d), tf.shape(self.data[0])) for d in self.data]):
        #     self.Nt = tf.shape(self.data[0])[0]
        #     self.D = tf.shape(self.data[0])[-1]
        #     self.N_slice = self.slice_size * tf.reduce_prod(tf.shape(self.data[0])[1:-1])
        #     self.N = tf.reduce_prod(tf.shape(self.data[0])[:-1])
        self.data_feed = self.index_feed.feed.map(self.get_data_block, num_parallel_calls=self.num_parallel_calls)
        self.feed = self.data_feed

    def get_data_block(self, index, next_index):
        """
        Get the time slice from index to index + step
        :param index: tf.int32, Tensor, scalar
            Index to start slice at
        :return: float_type, Tensor, [N, D]
            The returned data block
        """
        next_index = tf.minimum(next_index, self.Nt)
        indices = tf.range(index, next_index)
        data = [flatten_batch_dims(tf.gather(d, indices, axis=0),num_batch_dims=-self.event_size) for d in self.data]
        return data

    def _get_block(self, index, next_index):
        selection = {'time':slice(index,next_index)}
        selection.update(self.selection)
        with DataPack as datapack:
            for (solset,soltab) in self.addresses:
                datapack.switch_solset(solset)
                datapack.select(**self.selection)
                val, axes = getattr(soltab)

        # X = tf.py_function(lambda *X: make_coord_array(*[x.numpy() for x in X], flat=False),
        #                       X, float_type)


class TimeFeed(Feed):
    def __init__(self, index_feed: IndexFeed, times, num_parallel_calls=10):
        """
        Create a time feed
        :param index_feed: IndexFeed
            Pulse of this feed
        :param times: float_type, Tensor, [Nt, 1]
            Times to slice
        :param num_parallel_calls:
        """
        self.num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, tf.int32)
        if times.dtype is not float_type:
            times = tf.cast(times, float_type)
        self.times = times
        self.slice_size = index_feed.step
        self.Nt = tf.shape(self.times)[0]
        self.index_feed = index_feed
        self.time_feed = index_feed.feed.map(self.get_times_slice, num_parallel_calls=self.num_parallel_calls)
        self.feed = self.time_feed

    def get_times_slice(self, index, next_index):
        """
        Get the time slice from index to index + step
        :param index: tf.int32, Tensor, scalar
            Index to start slice at
        :return: float_type, Tensor, [step, D]
            The returned times coordinate slice
        """
        next_index = tf.minimum(next_index, self.Nt)
        indices = tf.range(index, next_index)
        return tf.gather(self.times, indices, axis=0)

class ContinueFeed(Feed):
    def __init__(self, time_feed: TimeFeed, num_parallel_calls = 10):
        self.num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, tf.int32)
        self.time_feed = time_feed
        self.continue_feed = time_feed.index_feed.feed.map(self.still_active, num_parallel_calls=self.num_parallel_calls)
        self.feed = self.continue_feed
        self.num_blocks = tf.cast(tf.ceil(tf.div(tf.cast(self.time_feed.Nt, tf.float32), tf.cast(self.time_feed.slice_size, tf.float32))), tf.int32)

    def still_active(self,index, next_index):
        return tf.less(next_index, self.time_feed.Nt)

class CoordinateFeed(object):
    def __init__(self, time_feed:TimeFeed, *coordinates, coord_map=None, num_parallel_calls=10):
        """
        Creates  coordinate feed, that broadcasts time slices with stacked coordinates.

        :param time_feed: TimeFeed
            The pulse of the coordinate feed, provides time slices
        :param coordinates: list of float_type, Tensor
            If the i-th coordinate is [Ni, Di], the then coordinate block is of shape [N0,..., Np, D0+...+Dp]
        """
        self.num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, tf.int32)
        self.coord_map = coord_map
        self.N = np.prod([tf.shape(c)[0] for c in coordinates])
        self.coordinates = [tf.cast(c, float_type) if c.dtype is not float_type else c for c in coordinates]
        self.time_feed = time_feed
        self.slice_size = self.time_feed.slice_size
        self.dims = tf.stack([self.slice_size]+[tf.shape(c)[0] for c in coordinates], axis=0)
        self.coord_feed = self.time_feed.feed.map(self.get_coord_block, num_parallel_calls=self.num_parallel_calls)
        self.feed = self.coord_feed

    def get_coord_block(self, time_slice):
        """
        Create the coordinate blob by stack the time slice with stacked coordinates.
        :param time_slice: float_type, Tensor, [Nt, 1]
            The time slice
        :return: float_type, Tensor, [Nt*N0*...*Np, D]
            The returned flattened coordinates
        """
        return self._make_coord_array(time_slice, *self.coordinates, flat=True)

    def _make_coord_array(self, *X, flat=True):
        X = tf.py_function(lambda *X: make_coord_array(*[x.numpy() for x in X], flat=False),
                              X, float_type)
        if self.coord_map is not None:
            X = tf.map_fn(self.coord_map, X, back_prop=False)
        if flat:
            return flatten_batch_dims(X)
        return X

class CoordinateDimFeed(object):
    def __init__(self, coord_feed:CoordinateFeed, num_parallel_calls=10):
        """
        Create a coordinate dimension feed that correctly incorperates partial batches.

        :param coord_feed: CoordinateFeed
        :param num_parallel_calls: int
        """
        self.num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, tf.int32)
        self.coord_feed = coord_feed
        self.dim_feed = self.coord_feed.feed.map(self.correct_dims, num_parallel_calls=self.num_parallel_calls)
        self.feed = self.dim_feed

    def correct_dims(self, X):
        """Correct for partial batch dims."""
        N = tf.shape(X)[0]
        N_slice = tf.reduce_prod(self.coord_feed.dims[1:])
        return tf.concat([[tf.floordiv(N, N_slice)], self.coord_feed.dims[1:]], axis=0)
