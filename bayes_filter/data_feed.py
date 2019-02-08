import tensorflow as tf
import numpy as np
from . import float_type
from .misc import flatten_batch_dims, make_coord_array


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
    def __init__(self, index_feed, *data, event_size=1, num_parallel_calls=10):
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

class TimeFeed(Feed):
    def __init__(self, index_feed, times, num_parallel_calls=10):
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

class CoordinateFeed(object):
    def __init__(self, time_feed, *coordinates, coord_map=None, num_parallel_calls=10):
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
