
from .common_setup import *
from ..misc import make_example_datapack
from ..hyper_parameter_opt import py_function_optimise_hyperparams
from ..feeds import DatapackFeed, IndexFeed, init_feed
import os

def test_py_function_optimise_hyperparams(tf_session):
    datapack = make_example_datapack(5,2,2,gain_noise=0.,clobber=True,obs_type='DDTEC',name=os.path.join(TEST_FOLDER,"test_hp_opt_data.h5"))
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        datapack_feed = DatapackFeed(index_feed, datapack, {'sol000':'phase'},screen_solset='sol000')
        init, ((Y_real, Y_imag), freqs, X, Xstar) = init_feed(datapack_feed)
        ddtec = tf.atan2(Y_imag, Y_real)*freqs/-8.448e6
        mean_ddtec = tf.reduce_mean(ddtec,axis=-1,keepdims=True)
        var_ddtec = tf.reduce_mean(tf.square(ddtec),axis=-1,keepdims=True) - tf.square(mean_ddtec)
        tf_session.run(init)
        learned_hyperparams = tf.py_function(py_function_optimise_hyperparams(resolution=5, obs_type='DTEC'),[X, mean_ddtec, var_ddtec],[float_type]*4)
        res = tf_session.run(learned_hyperparams)
        print(res)

        tf_session.run(init)
        learned_hyperparams = tf.py_function(py_function_optimise_hyperparams(resolution=5, obs_type='DDTEC'),
                                             [X, mean_ddtec, var_ddtec], [float_type] * 4)
        res = tf_session.run(learned_hyperparams)
        print(res)