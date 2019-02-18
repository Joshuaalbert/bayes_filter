from bayes_filter.filters import FreeTransitionSAEM
import tensorflow as tf
import tensorflow_probability as tfp
import os
from bayes_filter.misc import load_array_file, get_screen_directions
from bayes_filter import float_type
from bayes_filter.datapack import DataPack
import sys
from bayes_filter.data_feed import IndexFeed,TimeFeed,CoordinateFeed, DataFeed, init_feed, ContinueFeed
from bayes_filter.coord_transforms import tf_coord_transform, itrs_to_enu_with_references
from bayes_filter.kernels import DTECIsotropicTimeGeneral
import astropy.time as at
import numpy as np
import pylab as plt
import seaborn as sns
from timeit import default_timer
from bayes_filter import logging
from bayes_filter.settings import angle_type, dist_type


def arrays():
    return os.path.dirname(sys.modules["bayes_filter"].__file__)

def lofar_array(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    return load_array_file(lofar_array)

def lofar_array2(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    res = load_array_file(lofar_array)
    return res[0][[0,50, 51]], res[1][[0,50,51],:]

class LofarDR2:
    def __init__(self, tf_session, datapack, solset, srl_fits, index_feed_N=2, output_solset_base='dr2_saem', max_screen_dirs=None,
                 ant_sel=None, time_sel=None, dir_sel=None, freq_sel=None, pol_sel=None):
        # ant_idx = [0,50,51]
        with DataPack(datapack,readonly=True) as datapack:
            # screen_directions = get_screen_directions(srl_fits)
            # self._maybe_create_posterior_solsets(datapack, output_solset_base, solset, screen_directions)
            # output_solset = self.output_solset(output_solset_base)
            # screen_output_solset = self.output_screen_solset(output_solset_base)
            ###
            # get data
            datapack.switch_solset(solset)
            datapack.select(ant=None, time=time_sel, dir=None, freq=freq_sel, pol=pol_sel)
            axes = datapack.axes_phase
            antenna_labels, antennas = datapack.get_antennas(axes['ant'])
            patch_names, directions = datapack.get_sources(axes['dir'])
            ref_ant = antennas.cartesian.xyz.to(dist_type).value.T[0, :]
            ref_dir = np.mean(np.stack([directions.ra.to(angle_type).value, directions.dec.to(angle_type).value], axis=1), axis=0)
            datapack.select(ant=ant_sel, time=time_sel, dir=dir_sel, freq=freq_sel, pol=pol_sel)
            phase, axes = datapack.phase
            phase = phase
            #Nt, Nd, Na, Nf
            self.Y_real = np.cos(phase[0,...]).transpose((3,0,1,2))
            self.Y_imag = np.sin(phase[0,...]).transpose((3,0,1,2))
            antenna_labels, antennas = datapack.get_antennas(axes['ant'])
            patch_names, directions = datapack.get_sources(axes['dir'])
            timestamps, times = datapack.get_times(axes['time'])
            _, freqs = datapack.get_freqs(axes['freq'])
            #Nt, 1
            self.Xt = (times.mjd[:,None]).astype(np.float64)*86400.
            #Nd, 2
            self.Xd = np.stack([directions.ra.to(angle_type).value, directions.dec.to(angle_type).value],axis=1)
            #Na, 3
            self.Xa = antennas.cartesian.xyz.to(dist_type).value.T
            ###
            # get screen
            # datapack.switch_solset(screen_output_solset)
            # datapack.select(ant=ant_sel, time=time_sel, dir=None, freq=freq_sel, pol=pol_sel)
            # axes = datapack.axes_phase
            # antenna_labels, antennas = datapack.get_antennas(axes['ant'])
            # patch_names, directions = datapack.get_sources(axes['dir'])
            # timestamps, times = datapack.get_times(axes['time'])

            screen_directions = get_screen_directions(srl_fits)
            # Nt, 1
            self.screen_Xt = (times.mjd[:, None]).astype(np.float64) * 86400.
            # Nd, 2
            self.screen_Xd = np.stack([screen_directions.ra.to(angle_type).value, screen_directions.dec.to(angle_type).value], axis=1)
            # Na, 3
            self.screen_Xa = antennas.cartesian.xyz.to(dist_type).value.T

            with tf_session.graph.as_default():
                index_feed = IndexFeed(index_feed_N)
                time_feed = TimeFeed(index_feed, tf.convert_to_tensor(self.Xt, dtype=float_type))
                self.coord_feed = CoordinateFeed(time_feed, tf.convert_to_tensor(self.Xd, dtype=float_type),
                                                 tf.convert_to_tensor(self.Xa, dtype=float_type),
                                            coord_map=tf_coord_transform(itrs_to_enu_with_references(ref_ant, ref_dir, ref_ant)))
                self.star_coord_feed = CoordinateFeed(time_feed, tf.convert_to_tensor(self.screen_Xd, dtype=float_type),
                                                      tf.convert_to_tensor(self.screen_Xa, dtype=float_type),
                                                 coord_map=tf_coord_transform(itrs_to_enu_with_references(ref_ant, ref_dir, ref_ant)))
                self.Y_real_pl = tf.placeholder(float_type, self.Y_real.shape, 'Y_real')
                self.Y_imag_pl = tf.placeholder(float_type, self.Y_imag.shape, 'Y_imag')
                self.feed_dict = {self.Y_real_pl: self.Y_real, self.Y_imag_pl: self.Y_imag}
                self.data_feed = DataFeed(index_feed, self.Y_real_pl, self.Y_imag_pl, event_size=1)
                self.freqs = tf.convert_to_tensor(freqs,float_type)
                self.y_sigma = tf.convert_to_tensor(np.square(np.diff(self.Y_real,axis=0)).mean() + np.square(np.diff(self.Y_imag,axis=0)).mean(), float_type)

    def output_solset(self, base_name):
        return "posterior_{}".format(base_name)
    def output_screen_solset(self, base_name):
        return "screen_posterior_{}".format(base_name)

    def _maybe_create_posterior_solsets(self, datapack:DataPack, base_name, solset, screen_directions,
                                        remake_posterior_solsets=False):
        output_solset = self.output_solset(base_name)
        screen_output_solset = self.output_screen_solset(base_name)
        with datapack:
            if remake_posterior_solsets:
                datapack.delete_solset(output_solset)
                datapack.delete_solset(screen_output_solset)
            if not datapack.is_solset(output_solset) or not datapack.is_solset(
                    screen_output_solset):
                logging.info("Creating posterior solsets")
                datapack.switch_solset(solset)
                datapack.select(ant=None, time=None, dir=None, freq=None, pol=None)
                axes = datapack.axes_phase
                antenna_labels, antennas = datapack.get_antennas(axes['ant'])
                patch_names, directions = datapack.get_sources(axes['dir'])
                timestamps, times = datapack.get_times(axes['time'])
                freq_labels, freqs = datapack.get_freqs(axes['freq'])
                pol_labels, pols = datapack.get_pols(axes['pol'])
                Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
                datapack.switch_solset(output_solset,
                                        array_file=DataPack.lofar_array,
                                        directions=np.stack([directions.ra.to(angle_type).value,
                                                             directions.dec.to(angle_type).value], axis=1),
                                        patch_names=patch_names)
                datapack.add_freq_indep_tab('tec', times.mjd * 86400., pols=pol_labels)
                datapack.add_freq_dep_tab('phase', times.mjd * 86400., freqs=freqs,pols=pol_labels)
                datapack.switch_solset(screen_output_solset,
                                            array_file=DataPack.lofar_array,
                                            directions=np.stack([screen_directions.ra.to(angle_type).value,
                                                             screen_directions.dec.to(angle_type).value], axis=1))
                datapack.add_freq_indep_tab('tec', times.mjd * 86400., pols=pol_labels)
                datapack.add_freq_dep_tab('phase', times.mjd * 86400., freqs=freqs, pols=pol_labels)
                datapack.switch_solset(solset)

if __name__ == '__main__':
    from tensorflow.python import debug as tf_debug
    sess = tf.Session(graph=tf.Graph())
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess.graph.as_default():
        data_obj = LofarDR2(sess, '/home/albert/git/bayes_tec/scripts/data/P126+65_compact_full_raw.h5',
                            'sol000','/home/albert/git/bayes_tec/scripts/data/image.pybdsm.srl.fits',
                            max_screen_dirs=None, time_sel=slice(0,30,1),dir_sel=slice(0,5,1),ant_sel=slice(48,55,1),
                            pol_sel=slice(0,1,1),index_feed_N=2)

        free_transition = FreeTransitionSAEM(
            data_obj.freqs,
            data_obj.data_feed,
            data_obj.coord_feed,
            data_obj.star_coord_feed)

        filtered_res, inits = free_transition.filter_step(
            num_samples=1000, num_chains=1,parallel_iterations=10, num_leapfrog_steps=3,target_rate=0.6,
            num_burnin_steps=100,num_saem_samples=500,saem_maxsteps=10,initial_stepsize=7e-3,
            init_kern_params={'variance':0.5e-4,'y_sigma':data_obj.y_sigma,'lengthscales':5.,'timescale':50., 'a':200, 'b':50.},
            which_kernel=0, kernel_params={'resolution':3})
        sess.run(inits, data_obj.feed_dict)
        cont = True
        iteration = 0
        while cont:
            t0 = default_timer()
            res = sess.run(filtered_res)
            print("time",default_timer() - t0)
            print("post_logp",res.post_logp)
            cont = res.cont

