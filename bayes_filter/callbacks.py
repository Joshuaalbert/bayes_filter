from .datapack import DataPack
import tensorflow as tf
import numpy as np

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

class DatapackStoreCallback(Callback):
    def __init__(self, datapack, solset, soltab, dir_sel=None, ant_sel=None, freq_sel=None, pol_sel=None):
        super(DatapackStoreCallback, self).__init__(datapack=datapack,
                                                    solset=solset,
                                                    soltab=soltab,
                                                    dir_sel=dir_sel,
                                                    ant_sel=ant_sel,
                                                    freq_sel=freq_sel,
                                                    pol_sel=pol_sel)

    def generate(self, datapack, solset, soltab, dir_sel, ant_sel, freq_sel, pol_sel):

        if not isinstance(datapack, str):
            datapack = datapack.filename

        self.output_dtypes = [tf.int64]
        self.name = 'DatapackStoreCallback'

        def store(time_start, time_step, array):
            with DataPack(datapack,readonly=False) as dp:
                print(array.shape)
                dp.current_solset = solset
                dp.select(time=slice(time_start, time_step, 1))
                dp.__setattr__(soltab, array)#, dir=dir_sel, ant=ant_sel, freq=freq_sel, pol=pol_sel
            return [np.array(array.__sizeof__(),dtype=np.int64)]

        return store