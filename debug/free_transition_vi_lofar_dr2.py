import tensorflow as tf
import os
from bayes_filter import logging
from bayes_filter.filters import FreeTransitionVariationalBayes
from bayes_filter.feeds import DatapackFeed, IndexFeed
from bayes_filter.misc import make_example_datapack, maybe_create_posterior_solsets

if __name__ == '__main__':
    output_folder = os.path.join(os.path.abspath('test_filter_vi'), 'run11')
    os.makedirs(output_folder, exist_ok=True)
    datapack = make_example_datapack(5, 10, 2, name=os.path.join(output_folder, 'test_data.h5'), gain_noise=0.3,
                                     index_n=1, obs_type='DTEC', clobber=True,
                                     kernel_hyperparams={'variance': 3.5 ** 2, 'lengthscales': 15., 'a': 250.,
                                                         'b': 100., 'timescale': 50.})
    patch_names, _ = datapack.directions
    _, screen_directions = datapack.get_directions(patch_names)
    maybe_create_posterior_solsets(datapack, 'sol000', posterior_name='posterior', screen_directions=screen_directions)

    sess = tf.Session(graph=tf.Graph())
    # from tensorflow.python import debug as tf_debug
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess:
        logging.info("Setting up the index and datapack feeds.")
        datapack_feed = DatapackFeed(datapack,
                                     selection={'ant': slice(0, 62, 1)},
                                     solset='sol000',
                                     postieror_name='posterior',
                                     index_n=1)

        logging.info("Setting up the filter.")
        free_transition = FreeTransitionVariationalBayes(datapack_feed=datapack_feed, output_folder=output_folder)
        free_transition.init_filter()

        filter_op = free_transition.filter(
            parallel_iterations=10,
            kernel_params={'resolution': 5, 'fed_kernel': 'RBF', 'obs_type': 'DTEC'},
            num_parallel_filters=10,
        solve_iters=100,
        learning_rate=0.01,
        gamma=0.5,
        num_mcmc_param_samples_learn=50,
        num_mcmc_param_samples_infer=50)

        logging.info("Initializing the filter")
        sess.run(free_transition.initializer)
        # print(sess.run([free_transition.full_block_size, free_transition.datapack_feed.time_feed.slice_size, free_transition.datapack_feed.index_feed.step]))
        logging.info("Running the filter")
        sess.run(filter_op)
