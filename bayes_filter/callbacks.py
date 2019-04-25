from .datapack import DataPack
import tensorflow as tf
import numpy as np
from . import logging


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

        np.savez(store_file, times=np.array([]), variance=np.array([]), lengthscales=np.array([]), a=np.array([]), b=np.array([]), timescale=np.array([]))


        self.output_dtypes = [tf.int64]
        self.name = 'StoreHyperparameters'

        def store(time, variance, lengthscales, a, b, timescale):
            data = np.load(store_file)

            times = np.array([time] + list(data['times']))
            variance = np.array([variance] + list(data['variance']))
            lengthscales = np.array([lengthscales] + list(data['lengthscales']))
            a = np.array([a] + list(data['a']))
            b = np.array([b] + list(data['b']))
            timescale = np.array([timescale] + list(data['timescale']))

            np.savez(store_file,
                     times=times,
                     variance=variance,
                     lengthscales=lengthscales,
                     a=a,
                     b=b,
                     timescale=timescale
                     )

            return [np.array(len(times),dtype=np.int64)]

        return store

class DatapackStoreCallback(Callback):
    def __init__(self, datapack, solset, soltab, **selection):
        super(DatapackStoreCallback, self).__init__(datapack=datapack,
                                                    solset=solset,
                                                    soltab=soltab,
                                                    **selection)

    def generate(self, datapack, solset, soltab, **selection):

        if not isinstance(datapack, str):
            datapack = datapack.filename

        selection.pop('time',None)

        self.output_dtypes = [tf.int64]
        self.name = 'DatapackStoreCallback'

        def store(time_start, time_stop, array):
            with DataPack(datapack,readonly=False) as dp:
                print(array.shape)
                dp.current_solset = solset
                dp.select(time=slice(time_start, time_stop, 1), **selection)
                dp.__setattr__(soltab, array)#, dir=dir_sel, ant=ant_sel, freq=freq_sel, pol=pol_sel
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