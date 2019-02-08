import tensorflow as tf
import numpy as np
import astropy.coordinates as ac
import astropy.units as au
from .settings import dist_type, float_type, jitter
from timeit import default_timer


def diagonal_jitter(N):
    return tf.diag(tf.fill([N],tf.convert_to_tensor(jitter,float_type)))


def timer():
    """
    Return system time as tensorflow op
    :return:
    """
    return tf.py_function(default_timer, [], float_type)

def flatten_batch_dims(t, num_batch_dims=None):
    """
    Flattens the first `num_batch_dims`
    :param t: Tensor [b0,...bB, n0,...nN]
        Flattening happens for first `B` dimensions
    :param num_batch_dims: int, or tf.int32
        Number of dims in batch to flatten. If None then all but last. If < 0 then count from end.
    :return: Tensor [b0*...*bB, n0,...,nN]
    """
    shape = tf.shape(t)
    if num_batch_dims is None:
        num_batch_dims = tf.size(shape) - 1
    out_shape = tf.concat([[-1], shape[num_batch_dims:]],axis=0)
    return tf.reshape(t,out_shape)

def make_coord_array(*X, flat=True, coord_map=None):
    """
    Create the design matrix from a list of coordinates
    :param X: list of length p of float, array [Ni, D]
        Ni can be different for each coordinate array, but D must be the same.
    :param flat: bool
        Whether to return a flattened representation
    :param coord_map: callable(coordinates), optional
            If not None then get mapped over the coordinates
    :return: float, array [N0,...,Np, D] if flat=False else [N0*...*Np, D]
        The coordinate design matrix
    """

    if coord_map is not None:
        X = [coord_map(x) for x in X]

    def add_dims(x, where, sizes):
        shape = []
        tiles = []
        for i in range(len(sizes)):
            if i not in where:
                shape.append(1)
                tiles.append(sizes[i])
            else:
                shape.append(-1)
                tiles.append(1)
        return np.tile(np.reshape(x, shape), tiles)

    N = [x.shape[0] for x in X]
    X_ = []

    for i, x in enumerate(X):
        for dim in range(x.shape[1]):
            X_.append(add_dims(x[:, dim], [i], N))
    X = np.stack(X_, axis=-1)
    if not flat:
        return X
    return np.reshape(X, (-1, X.shape[-1]))


def load_array_file(array_file):
    '''Loads a csv where each row is x,y,z in geocentric ITRS coords of the antennas'''

    try:
        types = np.dtype({'names': ['X', 'Y', 'Z', 'diameter', 'station_label'],
                          'formats': [np.double, np.double, np.double, np.double, 'S16']})
        d = np.genfromtxt(array_file, comments='#', dtype=types)
        diameters = d['diameter']
        labels = np.array(d['station_label'].astype(str))
        locs = ac.SkyCoord(x=d['X'] * au.m, y=d['Y'] * au.m, z=d['Z'] * au.m, frame='itrs')
        Nantenna = int(np.size(d['X']))
    except:
        d = np.genfromtxt(array_file, comments='#', usecols=(0, 1, 2))
        locs = ac.SkyCoord(x=d[:, 0] * au.m, y=d[:, 1] * au.m, z=d[:, 2] * au.m, frame='itrs')
        Nantenna = d.shape[0]
        labels = np.array(["ant{:02d}".format(i) for i in range(Nantenna)])
        diameters = None
    return np.array(labels).astype(np.str_), locs.cartesian.xyz.to(dist_type).value.transpose()

# def save_array_file(array_file):
#     import time
#     ants = _solset.getAnt()
#     labels = []
#     locs = []
#     for label, pos in ants.items():
#         labels.append(label)
#         locs.append(pos)
#     Na = len(labels)
# with open(array_file, 'w') as f:
#     f.write('# Created on {0} by Joshua G. Albert\n'.format(time.strftime("%a %c", time.localtime())))
#     f.write('# ITRS(m)\n')
#     f.write('# X\tY\tZ\tlabels\n')
#     i = 0
#     while i < Na:
#         f.write(
#             '{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:d}\t{4}'.format(locs[i][0], locs[i][1], locs[i][2], labels[i]))
#         if i < Na - 1:
#             f.write('\n')
#         i += 1
