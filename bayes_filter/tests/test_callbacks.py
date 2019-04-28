from .common_setup import *

import os
from ..callbacks import DatapackStoreCallback, GetLearnIndices, StoreHyperparameters
from ..misc import make_example_datapack
from ..settings import float_type,dist_type

import tensorflow as tf
import numpy as np

def test_datapack_store(tf_session):
    datapack = make_example_datapack(10,2,2,clobber=True, name=os.path.join(TEST_FOLDER,"test_callbacks_data.h5"))
    datapack.switch_solset('sol000')
    phase, axes = datapack.phase
    store_array = np.ones_like(phase)#np.random.normal(size=phase.shape)
    datapack.phase = store_array
    with tf_session.graph.as_default():
        callback = DatapackStoreCallback(datapack,'sol000', 'phase',
                                  dir=None,
                                  ant=None,
                                  freq=None,
                                  pol=None)
        array = tf.convert_to_tensor(store_array, tf.float64)
        store_op = callback(0, 2, array)
        assert tf_session.run(store_op)[0] == store_array.__sizeof__()

    with datapack:
        stored, _ = datapack.phase
        # print(stored, store_array, phase)
        assert np.all(np.isclose(stored, store_array))

def test_get_learn_indices(tf_session):
    datapack = make_example_datapack(10,2,2,clobber=True, name=os.path.join(TEST_FOLDER,"test_callbacks_data.h5"))
    antenna_labels, _ = datapack.antennas
    _, antennas = datapack.get_antennas(antenna_labels)
    Xa = antennas.cartesian.xyz.to(dist_type).value.T.astype(np.float64)
    with tf_session.graph.as_default():
        callback = GetLearnIndices(dist_cutoff=0.3)
        indices = callback(tf.convert_to_tensor(Xa, dtype=float_type))
        indices = tf_session.run(indices)[0]
    Xa = Xa[indices,:]
    for i in range(Xa.shape[0]):
        assert np.all(np.linalg.norm(Xa[i:i+1,:] - Xa[i+1:,:], axis=1) < 0.3)

def test_store_hyperparams(tf_session):
    with tf_session.graph.as_default():
        if os.path.exists(os.path.join(TEST_FOLDER,'test_hyperparam_store.npz')):
            os.unlink(os.path.join(TEST_FOLDER,'test_hyperparam_store.npz'))
        callback = StoreHyperparameters(store_file=os.path.join(TEST_FOLDER,'test_hyperparam_store.npz'))
        size = callback(1000., [0., 1., 2., 3., 4., 5.])
        assert tf_session.run(size)[0] == 1
