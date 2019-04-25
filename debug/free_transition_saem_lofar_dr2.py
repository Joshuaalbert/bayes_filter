
import tensorflow as tf
from bayes_filter import logging
from bayes_filter.filters import FreeTransitionSAEM
from bayes_filter.feeds import DatapackFeed, IndexFeed
from bayes_filter.misc import make_example_datapack, maybe_create_posterior_solsets




if __name__ == '__main__':

    datapack = make_example_datapack(4,2,2,name='test_data.h5',obs_type='DDTEC',clobber=True)
    patch_names, _ = datapack.directions
    _, screen_directions = datapack.get_directions(patch_names)
    maybe_create_posterior_solsets(datapack,'sol000',posterior_name='posterior', screen_directions=screen_directions)


    sess = tf.Session(graph=tf.Graph())
    # from tensorflow.python import debug as tf_debug
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess:

        logging.info("Setting up the index and datapack feeds.")
        index_feed = IndexFeed(1)
        datapack_feed = DatapackFeed(index_feed,
                                     datapack,
                                     selection={'ant':'RS*'},
                                     data_map={'sol000':'phase'},
                                     postieror_name='posterior')

        logging.info("Setting up the filter.")
        free_transition = FreeTransitionSAEM(datapack_feed=datapack_feed)
        free_transition.init_filter(init_kern_hyperparams={'variance':0.1,'lengthscales':15., 'a':250., 'b':100., 'timescale':50.},
                                    initial_stepsize=5e-3)

        filter_op = free_transition.filter(num_chains=2,
                                                     num_samples=1e3,
                                                     parallel_iterations=10,
                                                     num_leapfrog_steps=2,
                                                     target_rate=0.65,
                                                     num_burnin_steps=200,
                                                     which_kernel=0,
                                                     kernel_params={'resolution':5, 'fed_kernel':'RBF', 'obs_type':'DDTEC'},
                                                     num_adapation_steps=200,
                                                     num_parallel_filters=2,
                                                     tf_seed=0)
        logging.info("Initializing the filter")
        sess.run(free_transition.initializer)
        logging.info("Running the filter")
        sess.run(filter_op)
