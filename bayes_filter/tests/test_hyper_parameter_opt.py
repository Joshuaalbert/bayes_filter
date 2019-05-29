
from .common_setup import *
from .. import TEC_CONV
from ..misc import make_example_datapack,maybe_create_posterior_solsets
from ..hyper_parameter_opt import KernelHyperparameterSolveCallback
from ..feeds import DatapackFeed, IndexFeed, init_feed
import os

def test_py_function_optimise_hyperparams(tf_session):
    datapack = make_example_datapack(5,2,2,gain_noise=0.,index_n=2,clobber=True,obs_type='DDTEC',
                                     name=os.path.join(TEST_FOLDER,"test_hp_opt_data.h5"))
    patch_names, _ = datapack.directions
    _, screen_directions = datapack.get_directions(patch_names)
    maybe_create_posterior_solsets(datapack, 'sol000', 'posterior',
    screen_directions = screen_directions,remake_posterior_solsets = False)
    with tf_session.graph.as_default():
        index_feed = IndexFeed(2)
        datapack_feed = DatapackFeed(datapack, {'sol000':'phase'}, index_feed=index_feed)
        init, ((Y_real, Y_imag), freqs, X, Xstar, _, _, _) = init_feed(datapack_feed)
        ddtec = tf.atan2(Y_imag, Y_real)*freqs/TEC_CONV
        mean_ddtec = tf.reduce_mean(ddtec,axis=-1,keepdims=True)
        var_ddtec = tf.reduce_mean(tf.square(ddtec),axis=-1,keepdims=True) - tf.square(mean_ddtec)
        tf_session.run(init)
        learned_hyperparams = KernelHyperparameterSolveCallback(resolution=3, obs_type='DTEC')(X, mean_ddtec, var_ddtec,1.0, 15., 250., 100., 50.)
        res = tf_session.run(learned_hyperparams)
        print(res)

        tf_session.run(init)
        learned_hyperparams = KernelHyperparameterSolveCallback(resolution=3, obs_type='DDTEC')(X, mean_ddtec, var_ddtec,1.0, 15., 250., 100., 50.)
        res = tf_session.run(learned_hyperparams)
        print(res)