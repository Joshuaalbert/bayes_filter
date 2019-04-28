
import tensorflow as tf
import os
from bayes_filter import logging
from bayes_filter.filters import FreeTransitionSAEM
from bayes_filter.feeds import DatapackFeed, IndexFeed
from bayes_filter.misc import make_example_datapack, maybe_create_posterior_solsets




if __name__ == '__main__':

    output_folder = os.path.abspath('test_output')
    os.makedirs(output_folder,exist_ok=True)
    datapack = make_example_datapack(10,2,3,name=os.path.join(output_folder,'test_data.h5'),gain_noise=0.,
                                     index_n=1,obs_type='DTEC',clobber=True)
    patch_names, _ = datapack.directions
    _, screen_directions = datapack.get_directions(patch_names)
    maybe_create_posterior_solsets(datapack,'sol000',posterior_name='posterior', screen_directions=screen_directions)


    sess = tf.Session(graph=tf.Graph())
    # from tensorflow.python import debug as tf_debug
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess:

        logging.info("Setting up the index and datapack feeds.")
        datapack_feed = DatapackFeed(datapack,
                                     selection={'ant':slice(45,62,1)},
                                     solset='sol000',
                                     postieror_name='posterior',
                                     index_n=1)


        logging.info("Setting up the filter.")
        free_transition = FreeTransitionSAEM(datapack_feed=datapack_feed)
        free_transition.init_filter(init_kern_hyperparams={'y_sigma':0.1,'variance':0.09,'lengthscales':15., 'a':250., 'b':100., 'timescale':50.},
                                    initial_stepsize=5e-3)

        filter_op = free_transition.filter(num_chains=1,
                                         num_samples=10e3,
                                         parallel_iterations=10,
                                         num_leapfrog_steps=2,
                                         target_rate=0.6,
                                         num_burnin_steps=2000,
                                         which_kernel=0,
                                         kernel_params={'resolution':5, 'fed_kernel':'RBF', 'obs_type':'DTEC'},
                                         num_adapation_steps=1600,
                                         num_parallel_filters=2,
                                         tf_seed=0)
        logging.info("Initializing the filter")
        sess.run(free_transition.initializer)
        # print(sess.run([free_transition.full_block_size, free_transition.datapack_feed.time_feed.slice_size, free_transition.datapack_feed.index_feed.step]))
        logging.info("Running the filter")
        sess.run(filter_op)
