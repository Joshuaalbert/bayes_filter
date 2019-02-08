import astropy.coordinates as ac
import astropy.time as at
from .frames import ENU
from .settings import dist_type, angle_type
import numpy as np
import tensorflow as tf


def tf_coord_transform(transform):
    def tf_transform(X):
        return tf.py_function(lambda X: transform(X.numpy()), [X], X.dtype)
    return tf_transform


def itrs_to_enu_6D(X, ref_location=None):
    """
    Convert the given coordinates from ITRS to ENU

    :param X: float array [b0,...,bB,6]
        The coordinates are ordered [time, ra, dec, itrs.x, itrs.y, itrs.z]
    :param ref_location: float array [3]
        Point about which to rotate frame.
    :return: float array [b0,...,bB, 7]
        The transforms coordinates.
    """

    if np.unique(X[...,0]).size > 1:
        raise ValueError("Times should be the same.")
    time = np.unique(X[...,0])
    shape = X.shape[:-1]
    X = X.reshape((-1, 6))
    if ref_location is None:
        ref_location = X[0,3:]
    obstime = at.Time(time / 86400., format='mjd')
    location = ac.ITRS(x=ref_location[0] * dist_type, y=ref_location[1] * dist_type, z=ref_location[2] * dist_type)
    enu = ENU(location=location, obstime=obstime)
    antennas = ac.ITRS(x=X[:, 3] * dist_type, y=X[:, 4] * dist_type, z=X[:, 5] * dist_type, obstime=obstime)
    antennas = antennas.transform_to(enu).cartesian.xyz.to(dist_type).value.T
    directions = ac.ICRS(ra=X[:, 1] * angle_type, dec=X[:, 2] * angle_type)
    directions = directions.transform_to(enu).cartesian.xyz.value.T
    return np.concatenate([X[:,0:1], directions, antennas], axis=1).reshape(shape+(7,))
