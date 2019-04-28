from .datapack import DataPack
import tensorflow as tf
import numpy as np
import os
from . import logging
from .plotting import DatapackPlotter
import pylab as plt


class Callback(object):
    def __init__(self, *args, **kwargs):
        self._output_dtypes = None
        self._name = 'Callback'
        self.callback_func = self.generate(*args, **kwargs)


    def generate(self, *args, **kwargs):
        raise NotImplementedError("Must subclass")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def output_dtypes(self):
        if self._output_dtypes is None:
            raise ValueError("Output dtype should be a list of output dtypes.")
        return self._output_dtypes

    @output_dtypes.setter
    def output_dtypes(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("output dtypes must be a list or tuple")
        self._output_dtypes = value

    def __call__(self, *Tin):
        def py_func(*Tin):
            result = self.callback_func(*[t.numpy() for t in Tin])
            if not isinstance(result, (list,tuple)):
                result = [result]

            if len(result) != len(self.output_dtypes):
                raise ValueError("Len of py_function result {} not equal to number of output dtypes {}".format(len(result), len(self.output_dtypes)))

            return result

        return tf.py_function(py_func, Tin, self.output_dtypes, name=self.name)

class StoreHyperparameters(Callback):
    def __init__(self, store_file):
        super(StoreHyperparameters, self).__init__(store_file=store_file)

    def generate(self, store_file):

        if not isinstance(store_file, str):
            raise ValueError("store_file should be str {}".format(type(store_file)))

        store_file=os.path.abspath(store_file)

        np.savez(store_file, times=np.array([]), y_sigma=np.array([]), variance=np.array([]), lengthscales=np.array([]), a=np.array([]), b=np.array([]), timescale=np.array([]))


        self.output_dtypes = [tf.int64]
        self.name = 'StoreHyperparameters'

        def store(time, hyperparams):
            data = np.load(store_file)
            #must match the order in the Target
            y_sigma, variance, lengthscales, a, b, timescale = np.reshape(hyperparams, (-1,))

            times = np.array([time] + list(data['times']))
            y_sigma = np.array([y_sigma] + list(data['y_sigma']))
            variance = np.array([variance] + list(data['variance']))
            lengthscales = np.array([lengthscales] + list(data['lengthscales']))
            a = np.array([a] + list(data['a']))
            b = np.array([b] + list(data['b']))

            timescale = np.array([timescale] + list(data['timescale']))

            np.savez(store_file,
                     times=times,
                     y_sigma=y_sigma,
                     variance=variance,
                     lengthscales=lengthscales,
                     a=a,
                     b=b,
                     timescale=timescale
                     )

            return [np.array(len(times),dtype=np.int64)]

        return store

class DatapackStoreCallback(Callback):
    def __init__(self, datapack, solset, soltab, perm=(0,2,3,1),**selection):
        super(DatapackStoreCallback, self).__init__(datapack=datapack,
                                                    solset=solset,
                                                    soltab=soltab,
                                                    perm=perm,
                                                    **selection)

    def generate(self, datapack, solset, soltab, perm, **selection):

        if not isinstance(datapack, str):
            datapack = datapack.filename

        selection.pop('time',None)

        self.output_dtypes = [tf.int64]
        self.name = 'DatapackStoreCallback'

        def store(time_start, time_stop, array):
            with DataPack(datapack,readonly=False) as dp:
                dp.current_solset = solset
                dp.select(time=slice(time_start, time_stop, 1), **selection)
                dp.__setattr__(soltab, np.transpose(array, perm))#, dir=dir_sel, ant=ant_sel, freq=freq_sel, pol=pol_sel
            return [np.array(array.__sizeof__(),dtype=np.int64)]

        return store


class GetLearnIndices(Callback):
    def __init__(self, dist_cutoff=0.3):
        super(GetLearnIndices, self).__init__(dist_cutoff=dist_cutoff)

    def generate(self, dist_cutoff):
        self.output_dtypes = [tf.int64]
        self.name = 'GetLearnIndices'
        def get_learn_indices(X):
            """Get the indices of non-redundant antennas
            :param X: np.array, float64, [N, 3]
                Antenna locations
            :param cutoff: float
                Mark redundant if antennas within this in km
            :return: np.array, int64
                indices such that all antennas are at least cutoff apart
            """
            N = X.shape[0]
            Xa, inverse = np.unique(X, return_inverse=True, axis=0)
            Na = len(Xa)
            keep = []
            for i in range(Na):
                if np.all(np.linalg.norm(Xa[i:i + 1, :] - Xa[keep, :], axis=1) > dist_cutoff):
                    keep.append(i)
            logging.info("Training on antennas: {}".format(keep))
            return [(np.where(np.isin(inverse, keep, assume_unique=True))[0]).astype(np.int64)]
        return get_learn_indices

class PlotResults(Callback):
    def __init__(self, hyperparam_store, datapack, solset, posterior_name='posterior', plot_directory='./plots', **selection):
        super(PlotResults, self).__init__(hyperparam_store=hyperparam_store,
                                          datapack=datapack,
                                          solset=solset,
                                          posterior_name=posterior_name,
                                          plot_directory=plot_directory,
                                          **selection)

    def generate(self, hyperparam_store, datapack, solset, posterior_name, plot_directory, **selection):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotResults'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.abspath(plot_directory)
        fig_directory = os.path.join(plot_directory,'phase_screens')
        os.makedirs(fig_directory,exist_ok=True)
        dp = DatapackPlotter(datapack)

        def plot_results(index_start, index_end):
            """Get the indices of non-redundant antennas
            :param X: np.array, float64, [N, 3]
                Antenna locations
            :param cutoff: float
                Mark redundant if antennas within this in km
            :return: np.array, int64
                indices such that all antennas are at least cutoff apart
            """

            data = np.load(hyperparam_store)
            keys = ['y_sigma','variance', 'lengthscales', 'a', 'b', 'timescale']

            fig, axs = plt.subplots(6,1,sharex=True, figsize=(6,6*2))
            for i,key in enumerate(keys):
                ax = axs[i]
                ax.scatter(data['times'], data[key], label=key)
                ax.legend()
            plt.savefig(os.path.join(plot_directory,'hyperparameters.pdf'))
            plt.close('all')


            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_phase.png".format(i,solset)) for i in
                        range(index_start, index_end, 1)]

            dp.plot(ant_sel=selection.get('ant',None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq',None),
                    dir_sel=selection.get('dir',None),
                    pol_sel=selection.get('pol', slice(0,1,1)),
                    fignames=fignames,
                    observable='phase',
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=solset)

            plt.close('all')

            data_posterior = "data_{}".format(posterior_name)
            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_phase.png".format(i, data_posterior)) for i in
                        range(index_start, index_end, 1)]

            dp.plot(ant_sel=selection.get('ant', None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq', None),
                    dir_sel=selection.get('dir', None),
                    pol_sel=selection.get('pol', slice(0, 1, 1)),
                    fignames=fignames,
                    observable='tec',
                    tec_eval_freq=160e6,
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=data_posterior)

            plt.close('all')

            screen_posterior = "screen_{}".format(posterior_name)
            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_phase.png".format(i, screen_posterior)) for i in
                        range(index_start, index_end, 1)]

            dp.plot(ant_sel=selection.get('ant', None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq', None),
                    dir_sel=selection.get('dir', None),
                    pol_sel=selection.get('pol', slice(0, 1, 1)),
                    fignames=fignames,
                    observable='tec',
                    tec_eval_freq=160e6,
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=screen_posterior)

            plt.close('all')


            return [np.array(3).astype(np.int64)]

        return plot_results

class PlotPerformance(Callback):
    def __init__(self, plot_directory='./plots'):
        super(PlotPerformance, self).__init__(
                                          plot_directory=plot_directory,
                                          )

    def generate(self, plot_directory):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotPerformance'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.abspath(plot_directory)
        perf_directory = os.path.join(plot_directory,'performance')
        os.makedirs(perf_directory,exist_ok=True)

        def plot_results(index_start, index_end, rhat, ess, log_accept_ratio, step_size):
            """Get the indices of non-redundant antennas
            :param X: np.array, float64, [N, 3]
                Antenna locations
            :param cutoff: float
                Mark redundant if antennas within this in km
            :return: np.array, int64
                indices such that all antennas are at least cutoff apart
            """


            # print(index_start, index_end, rhat, ess, log_accept_ratio, step_size)
            fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(6,6))
            ax1.hist(rhat.flatten(),bins=np.linspace(1.,10.,20), label='rhat')#,bins=max(10,int(np.sqrt(rhat.size)))
            ax1.legend()
            ax2.hist(ess.flatten(), bins=max(10, int(np.sqrt(ess.size))), label='ess')
            ax2.legend()
            ax3.hist(np.mean(np.exp(np.minimum(log_accept_ratio,0.)),axis=-1), bins=max(10, int(np.sqrt(log_accept_ratio.size))), label='log_accept_ratio')
            # ax3.plot(np.mean(np.exp(np.minimum(log_accept_ratio,0.)),axis=-1), label='prob_accept_ratio')
            ax3.legend()
            # ax4.hist(step_size.flatten(), bins=max(10, int(np.sqrt(step_size.size))), label='step_size')
            ax4.plot(step_size, label='step_size')
            ax4.legend()

            plt.savefig(os.path.join(perf_directory,'{}_{}_performance.pdf'.format(index_start,index_end)))
            plt.close('all')


            return [np.array(1).astype(np.int64)]

        return plot_results