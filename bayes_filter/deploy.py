from .model import AverageModel
from .datapack import DataPack
from typing import List, Union
from .coord_transforms import ITRSToENUWithReferences_v2
from . import logging, angle_type, dist_type, float_type
from .misc import get_screen_directions, maybe_create_posterior_solsets
import numpy as np
import tensorflow as tf
import os, glob

class Deployment(object):
    def __init__(self, datapack: Union[DataPack, str], ref_dir_idx = 14, solset='sol000',
                 flux_limit=0.05, max_N=250, min_spacing_arcmin=1.,
                 srl_file:str='/home/albert/ftp/image.pybdsm.srl.fits',
                 ant=None, dir=None, time=None, freq=None, pol=slice(0,1,1),
                 directional_deploy=True, block_size=1, working_dir = './deployment'):
        cwd = os.path.abspath(working_dir)
        os.makedirs(cwd, exist_ok=True)
        run_idx = len(glob.glob(os.path.join(cwd, 'run_*')))
        cwd = os.path.join(cwd, 'run_{:03d}'.format(run_idx))
        os.makedirs(cwd, exist_ok=True)
        self.cwd = cwd
        logging.info("Using working direction {}".format(cwd))

        if isinstance(datapack, str):
            datapack = DataPack(datapack)
        screen_directions = get_screen_directions(srl_file, flux_limit=flux_limit, max_N=max_N,
                                                  min_spacing_arcmin=min_spacing_arcmin)
        datapack = DataPack(datapack, readonly=False)
        maybe_create_posterior_solsets(datapack, solset, posterior_name='posterior',
                                   screen_directions=screen_directions,
                                       make_soltabs=['phase000', 'tec000'])
        logging.info("Preparing data")
        datapack.current_solset = solset
        self.select = dict(ant=ant, dir=dir, time=time, freq=freq, pol=pol)
        datapack.select(**self.select)
        tec, axes = datapack.tec
        tec_uncert, _ = datapack.weights_tec
        self.directional_deploy = directional_deploy
        if self.directional_deploy:
            # Nd, Na, Nt -> Nt, Na, Nd
            tec = tec[0, ...].transpose((2, 1, 0))
            self.Nt, self.Na, self.Nd = tec.shape
            tec_uncert = tec_uncert[0, ...].transpose((2, 1, 0))
        else:
            # Nd, Na, Nt
            tec = tec[0, ...]
            self.Nd, self.Na, self.Nt = tec.shape
            tec_uncert = tec_uncert[0, ...]
        _, times = datapack.get_times(axes['time'])
        Xt = (times.mjd * 86400.)[:, None]
        _, directions = datapack.get_directions(axes['dir'])
        Xd = np.stack([directions.ra.to(angle_type).value, directions.dec.to(angle_type).value], axis=1)
        self.ref_dir = Xd[ref_dir_idx, :]
        _, antennas = datapack.get_antennas(axes['ant'])
        Xa = antennas.cartesian.xyz.to(dist_type).value.T
        self.ref_ant = Xa[0, :]

        datapack.current_solset = 'screen_posterior'
        datapack.select(**self.select)
        axes = datapack.axes_phase
        _, screen_directions = datapack.get_directions(axes['dir'])
        Xd_screen = np.stack([screen_directions.ra.to(angle_type).value, screen_directions.dec.to(angle_type).value],
                             axis=1)
        self.Nd_screen = Xd_screen.shape[0]

        self.datapack = datapack
        self.Xa = Xa
        self.Xd = Xd
        self.Xd_screen = Xd_screen
        self.Xt = Xt
        self.tec = tec
        self.tec_uncert = tec_uncert
        self.block_size = block_size


    def run(self, model_generator):
        with tf.Session(graph=tf.Graph()) as tf_session:
            logging.info("Creating coordinate transform")
            coord_transform = ITRSToENUWithReferences_v2(self.ref_ant, self.ref_dir, self.ref_ant)
            Xt_pl = tf.placeholder(float_type, shape=[1, 1])
            Xd_pl = tf.placeholder(float_type, shape=self.Xd.shape)
            Xd_screen_pl = tf.placeholder(float_type, shape=self.Xd_screen.shape, )
            Xa_pl = tf.placeholder(float_type, shape=self.Xa.shape)
            tf_X, tf_ref_ant, tf_ref_dir = coord_transform(Xt_pl, Xd_pl, Xa_pl)
            tf_X_screen, _, _ = coord_transform(Xt_pl, Xd_screen_pl, Xa_pl)

            if self.directional_deploy:
                # Nd, 3
                tf_X = tf_X[0, :, 0, :3]
                # Nd_screen, 3
                tf_X_screen = tf_X_screen[0, :, 0, :3]
            else:
                # Nt * Nd * Na, 6
                tf_X = tf.reshape(tf_X, (-1, 6))
                # Nt * Nd_screen * Na, 6
                tf_X_screen = tf.reshape(tf_X_screen, (-1, 6))

            post_mean_array = []
            post_std_array = []
            weights_array = []

            for t in range(0, self.Nt, self.block_size):
                start = t
                stop = min(self.Nt, t + self.block_size)
                mid_time = start + (stop - start) // 2
                logging.info("Beginning time block {} to {}".format(start, stop))

                feed_dict = {}
                # T*Na, N
                feed_dict[Xa_pl] = self.Xa
                feed_dict[Xd_pl] = self.Xd
                feed_dict[Xd_screen_pl] = self.Xd_screen
                feed_dict[Xt_pl] = self.Xt[start:stop, :]

                X, X_screen, ref_direction = tf_session.run([tf_X, tf_X_screen, tf_ref_dir],
                                                 feed_dict)

                if self.directional_deploy:
                    # Nt, Na, Nd
                    Y = np.transpose(self.tec[start:stop,:,:], (0, 2,1))
                    # block_size*Na, Nd
                    batch_size = Y.shape
                    Y = Y.reshape((-1, self.Nd))
                    Y_var = np.square(np.reshape(np.transpose(self.tec_uncert[start:stop,:,:], (0, 2,1)), (-1, self.Nd)))
                else:
                    #block_size, Nd, Na
                    Y = self.tec[start:stop, :, :]
                    batch_shape = Y.shape
                    Y = Y.reshape((-1, self.Nd*self.Na))
                    # block_size, Nd, Na
                    Y_var = np.square(self.tec_uncert[start:stop, :, :]).reshape((-1, self.Nd*self.Na))

                models = model_generator(X, Y, Y_var, ref_direction, reg_param = 1., parallel_iterations = 10)

                model = AverageModel(models)
                logging.info("Optimising models")
                model.optimise()
                logging.info("Predicting posteriors and averaging")
                # batch_size, N
                (weights, log_marginal_likelihoods), post_mean, post_var = model.predict_f(X_screen, only_mean=False)
                if self.directional_deploy:
                    # num_models, block_size, Na
                    weights = np.reshape(weights, (-1,) + batch_shape[:2])
                    log_marginal_likelihoods = np.reshape(log_marginal_likelihoods, (-1,) + batch_shape[:-2])
                    #block_size, Na, Nd -> block_size, Nd, Na
                    post_mean = post_mean.reshape(batch_shape).transpose((0,2,1))
                    post_var = post_var.reshape(batch_shape).transpose((0,2,1))
                else:
                    # num_models, block_size
                    weights = np.reshape(weights, (-1,) + batch_shape[:1])
                    log_marginal_likelihoods = np.reshape(log_marginal_likelihoods, (-1,) + batch_shape[:1])
                    # block_size, Nd, Na -> block_size, Nd, Na
                    post_mean = post_mean.reshape(batch_shape)
                    post_var = post_var.reshape(batch_shape)
                weights_array.append(weights)
                post_mean_array.append(post_mean)
                post_std_array.append(np.sqrt(post_var))

            if self.directional_deploy:
                #num_models, Nt, Na
                weights = np.concatenate(weights, axis=1)
            else:
                #num_models, Nt
                weights = np.concatenate(weights, axis=1)
            #Nt, Nd, Na -> Nd, Na, Nt
            post_mean_array = np.concatenate(post_mean_array, axis=0).transpose((1,2,0))
            post_std_array = np.concatenate(post_std_array, axis=0).transpose((1,2,0))
            self.datapack.current_solset = 'screen_posterior'
            self.datapack.tec = post_mean_array
            self.datapack.weights_tec = post_std_array

            axes = self.datapack.axes_phase
            _, freqs = self.datapack.get_freqs(axes['freq'])
            tec_conv = -8.4479745e6 / freqs

            post_phase_mean = post_mean_array[..., None, :] * tec_conv
            post_phase_std = np.abs(post_std_array[..., None, :]*tec_conv)

            self.datapack.phase = post_phase_mean
            self.datapack.weights_phase = post_phase_std

            np.save(os.path.join(self.cwd, 'weights.npy'), weights_array)

if __name__ == '__main__':
    datapack = '/home/albert/lofar1_1/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v10.h5'
    deployment = Deployment(datapack,
                            ref_dir_idx=14, solset='sol000',
                            flux_limit=0.05, max_N=250, min_spacing_arcmin=1.,
                            srl_file = '/home/albert/ftp/image.pybdsm.srl.fits',
                            ant = None, dir = None, time = slice(0, 1, 1), freq = None,
                            pol = slice(0, 1, 1),
                            directional_deploy = True, block_size = 1,
                            working_dir = './deployment')
    from bayes_filter.directional_models import generate_models
    deployment.run(generate_models)