from .common_setup import *

import os
from ..callbacks import DatapackStoreCallback
from ..misc import make_example_datapack

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
                                  dir_sel=None,
                                  ant_sel=None,
                                  freq_sel=None,
                                  pol_sel=None)
        array = tf.convert_to_tensor(store_array, tf.float64)
        store_op = callback(0, 2, array)
        assert tf_session.run(store_op)[0] == store_array.__sizeof__()

    with datapack:
        stored, _ = datapack.phase
        # print(stored, store_array, phase)
        assert np.all(np.isclose(stored, store_array))
