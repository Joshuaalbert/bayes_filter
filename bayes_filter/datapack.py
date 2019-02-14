from . import logging
from losoto.h5parm import h5parm
import os
import numpy as np
import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import sys
import itertools


class DataPack(object):
    """
    We use losoto.h5parm as the data holder.
    """
    _arrays = os.path.dirname(sys.modules["bayes_filter"].__file__)
    lofar_array = os.path.join(_arrays, 'arrays/lofar.hba.antenna.cfg')
    lofar_cycle0_array = os.path.join(_arrays, 'arrays/lofar.cycle0.hba.antenna.cfg')
    gmrt_array = os.path.join(_arrays, 'arrays/gmrtPos.csv')

    def __init__(self, filename, readonly=False, solset='sol000'):
        self.filename = os.path.abspath(filename)
        self.readonly = readonly
        self.solset = solset
        self.H = None
        self._contexts_open = 0
        self._selection = None

    def is_solset(self, solset):
        """
        Does solset exist
        """
        with self:
            if solset not in self.H.getSolsetNames():
                return False
            else:
                return True

    def delete_solset(self, solset):
        with self:
            if not self.is_solset(solset):
                logging.warning("{} not a valid solset to delete".format(solset))
                return
            #            self.H.getSolset(solset).obj._f_rename('trash',overwrite=True)
            self.H.getSolset(solset).delete()

    def is_soltab(self, soltab):
        with self:
            return soltab in self._solset.getSoltabNames()

    def delete_soltab(self, soltab):
        with self:
            soltab = "{}000".format(soltab)
            if not self.is_soltab(soltab):
                logging.warning("{} is not a valid soltab in solset {}".format(soltab, self.solset))
                return
            self._solset.getSoltab(soltab).delete()

    def split_solset(self, solset, filename, soltabs=None, new_solset=None, clobber=False):
        """Split off a solset and requested soltabs into a new file.
        :param solset: str the name of the solset to split off
        :param filename: str the nanme of the new file to split into.
        :param soltabs: List(str) the names of soltabs to put into the new file.
            If None (default) then puts all soltabs.
        :param new_solset: str the name of the new solset. None (default) means same as `solset`.
        :param clobber: bool, whether to overwrite `filename`.
        Raises:
        IOError if `filename` exists, and `clobber` not True
        ValueError if `solset` is None
        """
        filename = os.path.abspath(filename)
        if os.path.exists(filename):
            if not clobber:
                raise IOError("{} already exists and clobber False".format(filename))
            logging.info("Overwriting {}".format(filename))
            os.unlink(filename)
        with self:
            self.switch_solset(solset)
            soltabs = soltabs or self.soltabs
            patch_names, directions = self.sources
            antenna_labels, antennas = self.antennas
        if solset is None:
            raise ValueError("solset cannot be None")
        new_solset = new_solset or solset

        new_datapack = DataPack(filename, readonly=False, solset=new_solset)
        with new_datapack:
            new_datapack.switch_solset(new_solset, antenna_labels=antenna_labels, antennas=antennas,
                                       directions=directions, patch_names=patch_names)
            with self:
                for soltab in soltabs:
                    if soltab not in self.allowed_soltabs:
                        logging.info("Skipping {}".format(soltab))
                        continue
                    vals, axes = getattr(self, soltab)
                    weights, _ = getattr(self, "weights_{}".format(soltab))

                    timestamps, times = self.get_times(axes['time'])
                    pol_labels, pols = self.get_pols(axes['pol'])
                    if 'freq' in axes.keys():
                        _, freqs = self.get_freqs(axes['freq'])
                        new_datapack.add_freq_dep_tab(soltab, times.mjd * 86400., ants=axes['ant'], dirs=axes['dir'],
                                                      pols=pol_labels, freqs=freqs)
                    else:
                        new_datapack.add_freq_indep_tab(soltab, times.mjd * 86400., ants=axes['ant'], dirs=axes['dir'],
                                                        pols=pol_labels)
                    setattr(new_datapack, soltab, vals)
                    setattr(new_datapack, "weights_{}".format(soltab), weights)

    def switch_solset(self, solset, antenna_labels=None, antennas=None, array_file=None, directions=None,
                      patch_names=None):
        """
        returns
        True if already existed
        False if it make a new one
        """
        with self:
            if solset is None:
                solset = 'sol000'
            self.solset = solset
            if self.solset not in self.H.getSolsetNames():
                logging.info("Making solset: {}".format(self.solset))
                self.H.makeSolset(solsetName=self.solset, addTables=True)
                if directions is not None:
                    self.add_sources(directions, patch_names=patch_names)
                if array_file is not None:
                    self.add_antennas(labels=antenna_labels, pos=antennas, array_file=array_file)
                return False
            else:
                return True

    def __enter__(self):
        if self._contexts_open == 0:
            self.H = h5parm(self.filename, readonly=self.readonly)
        self._contexts_open += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.H:
            if self._contexts_open == 1:
                self.H.close()
                self.H = None
            self._contexts_open -= 1

    @property
    def solsets(self):
        with self:
            return self.H.getSolsetNames()

    @property
    def soltabs(self):
        with self:
            return self._solset.getSoltabNames()

    def __repr__(self):

        def grouper(n, iterable, fillvalue=None):
            "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return itertools.zip_longest(*args, fillvalue=fillvalue)

        with self:
            info = ""
            solsets = self.H.getSolsetNames()
            for solset in solsets:
                info += "=== solset: {} ===\n".format(solset)
                solset = self.H.getSolset(solset)
                sources = sorted(solset.getSou().keys())
                info += "Directions: "
                for src_name1, src_name2, src_name3 in grouper(3, sources):
                    info += "{0:}\t{1:}\t{2:}\n".format(src_name1, src_name2, src_name3)

                antennas = sorted(solset.getAnt().keys())
                info += "\nStations: "
                for ant1, ant2, ant3, ant4 in grouper(4, antennas):
                    info += "{0:}\t{1:}\t{2:}\t{3:}\n".format(ant1, ant2, ant3, ant4)
                soltabs = solset.getSoltabNames()
                for soltab in soltabs:
                    info += "== soltab: {} ==\n".format(soltab)
                    soltab = solset.getSoltab(soltab)
                    shape = tuple([soltab.getAxisLen(a) for a in soltab.getAxesNames()])
                    info += "shape {}\n".format(shape)

            return info

    @property
    def _solset(self):
        with self:
            try:
                return self.H.getSolset(self.solset)
            except:
                raise ValueError("solset {} does not exist".format(self.solset))

    def _load_array_file(self, array_file):
        '''Loads a csv where each row is x,y,z in geocentric ITRS coords of the antennas'''

        try:
            types = np.dtype({'names': ['X', 'Y', 'Z', 'diameter', 'station_label'],
                              'formats': [np.double, np.double, np.double, np.double, 'S16']})
            d = np.genfromtxt(array_file, comments='#', dtype=types)
            diameters = d['diameter']
            labels = np.array(d['station_label'].astype(str))
            locs = ac.SkyCoord(x=d['X'] * au.m, y=d['Y'] * au.m, z=d['Z'] * au.m, frame='itrs')
            Nantenna = int(np.size(d['X']))
        except:
            d = np.genfromtxt(array_file, comments='#', usecols=(0, 1, 2))
            locs = ac.SkyCoord(x=d[:, 0] * au.m, y=d[:, 1] * au.m, z=d[:, 2] * au.m, frame='itrs')
            Nantenna = d.shape[0]
            labels = np.array([b"ant{:02d}".format(i) for i in range(self.Nantenna)])
            diameters = None
        return np.array(labels).astype(np.str_), locs.cartesian.xyz.to(au.m).value.transpose()

    def save_array_file(self, array_file):
        import time
        with self:
            ants = self._solset.getAnt()
            labels = []
            locs = []
            for label, pos in ants.items():
                labels.append(label)
                locs.append(pos)
            Na = len(labels)
        with open(array_file, 'w') as f:
            f.write('# Created on {0} by Joshua G. Albert\n'.format(time.strftime("%a %c", time.localtime())))
            f.write('# ITRS(m)\n')
            f.write('# X\tY\tZ\tlabels\n')
            i = 0
            while i < Na:
                f.write(
                    '{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:d}\t{4}'.format(locs[i][0], locs[i][1], locs[i][2], labels[i]))
                if i < Na - 1:
                    f.write('\n')
                i += 1

    def add_antennas(self, labels=None, pos=None, array_file=None):
        """Adds antennas to the datapack solset.
        :param labels: array of string of station names
        :param pos: array of positions in ITRF(m) frame of station positions
        :param array_file: array_file to load relevant data if labels and pos are None
        """
        if labels is None and pos is None:
            if array_file is None:
                array_file = self.lofar_array
            labels, pos = self._load_array_file(array_file)
        labels = np.array(labels)
        pos = np.array(pos)

        with self:
            antennaTable = self._solset.obj._f_get_child('antenna')
            for lab, p in zip(labels, pos):
                if lab not in antennaTable.cols.name[:].astype(type(lab)):
                    antennaTable.append([(lab, p)])
                else:
                    idx = np.where(antennaTable.cols.name[:].astype(type(lab)) == lab)[0][0]
                    antennaTable.cols.position[idx] = p

    def add_sources(self, directions, patch_names=None):
        Nd = len(directions)
        if patch_names is None:
            patch_names = []
            for d in range(Nd):
                patch_names.append("patch_{:03d}".format(d))
        with self:
            sourceTable = self._solset.obj._f_get_child('source')
            for lab, p in zip(patch_names, directions):
                if lab not in sourceTable.cols.name[:].astype(type(lab)):
                    sourceTable.append([(lab, p)])
                else:
                    logging.info("{} already in source list {}".format(lab, sourceTable.cols.name[:].astype(type(lab))))

    @property
    def _antennas(self):
        with self:
            return self._solset.obj.antenna

    @property
    def antennas(self):
        with self:
            antenna_labels, pos = [], []
            for a in self._antennas:
                antenna_labels.append(a['name'])
                pos.append(a['position'])
            return np.array(antenna_labels), np.stack(pos, axis=0)

    @property
    def ref_ant(self):
        with self:
            antenna_labels, antennas = self.antennas
            return antenna_labels[0]

    @property
    def array_center(self):
        with self:
            _, antennas = self.get_antennas(None)
            center = np.mean(antennas.cartesian.xyz, axis=1)
            center = ac.SkyCoord(x=center[0], y=center[1], z=center[2], frame='itrs')
            return center

    def get_antennas(self, ants):
        with self:
            antenna_labels, antennas = self.antennas

            if ants is None:
                ant_idx = slice(None)
            else:
                ants = np.array(ants).astype(antenna_labels.dtype)
                sorter = np.argsort(antenna_labels)
                ant_idx = np.searchsorted(antenna_labels, ants, sorter=sorter)
            antennas = antennas[ant_idx]
            return antenna_labels[ant_idx], ac.SkyCoord(antennas[:, 0] * au.m, antennas[:, 1] * au.m,
                                                        antennas[:, 2] * au.m, frame='itrs')

    @property
    def _sources(self):
        with self:
            return self._solset.obj.source

    @property
    def sources(self):
        with self:
            patch_names = []
            dirs = []
            for s in self._sources:
                patch_names.append(s['name'])
                dirs.append(s['dir'])
            return np.array(patch_names), np.stack(dirs, axis=0)

    def get_sources(self, dirs):
        with self:
            patch_names, directions = self.sources
            if dirs is None:
                dir_idx = slice(None)
            else:
                dirs = np.array(dirs).astype(patch_names.dtype)
                sorter = np.argsort(patch_names)
                dir_idx = np.searchsorted(patch_names, dirs, sorter=sorter)
            directions = directions[dir_idx]
            return patch_names[dir_idx], ac.SkyCoord(directions[:, 0] * au.rad, directions[:, 1] * au.rad, frame='icrs')

    @property
    def pointing_center(self):
        with self:
            _, directions = self.get_sources(None)
            ra_mean = np.mean(directions.transform_to('icrs').ra)
            dec_mean = np.mean(directions.transform_to('icrs').dec)
            dir = ac.SkyCoord(ra_mean, dec_mean, frame='icrs')
            return dir

    def get_times(self, times):
        """
        times are stored as mjs
        """
        times = at.Time(times / 86400., format='mjd')
        return times.isot, times

    def get_freqs(self, freqs):
        labs = ['{:.1f}MHz'.format(f / 1e6) for f in freqs]
        return np.array(labs), freqs

    def get_pols(self, pols):
        with self:
            return pols, np.arange(len(pols), dtype=np.int32)

    def add_freq_indep_tab(self, name, times, pols=None, ants=None, dirs=None, vals=None, weight_dtype='f64'):
        with self:
            if "{}000".format(name) in self._solset.getSoltabNames():
                logging.warning("{}000 is already a tab in {}".format(name, self.solset))
                return
            # pols = ['XX','XY','YX','YY']
            if dirs is None:
                dirs, _ = self.sources
            if ants is None:
                ants, _ = self.antennas
            if pols is not None:
                Npol = len(pols)
            Nd = len(dirs)
            Na = len(ants)
            Nt = len(times)
            if pols is not None:
                if vals is None:
                    vals = np.zeros([Npol, Nd, Na, Nt])
                self._solset.makeSoltab(name, axesNames=['pol', 'dir', 'ant', 'time'],
                                        axesVals=[pols, dirs, ants, times], vals=vals, weights=np.ones_like(vals),
                                        weightDtype=weight_dtype)
            else:
                if vals is None:
                    vals = np.zeros([Nd, Na, Nt])
                self._solset.makeSoltab(name, axesNames=['dir', 'ant', 'time'],
                                        axesVals=[dirs, ants, times], vals=vals, weights=np.ones_like(vals),
                                        weightDtype=weight_dtype)

    def add_freq_dep_tab(self, name, times, freqs, pols=None, ants=None, dirs=None, vals=None, weight_dtype='f64'):
        with self:
            if "{}000".format(name) in self._solset.getSoltabNames():
                logging.warning("{}000 is already a tab in {}".format(name, self.solset))
                return
            # pols = ['XX','XY','YX','YY']
            if dirs is None:
                dirs, _ = self.sources
            if ants is None:
                ants, _ = self.antennas
            if pols is not None:
                Npol = len(pols)
            Nd = len(dirs)
            Na = len(ants)
            Nt = len(times)
            Nf = len(freqs)
            if pols is not None:
                if vals is None:
                    vals = np.zeros([Npol, Nd, Na, Nf, Nt])
                self._solset.makeSoltab(name, axesNames=['pol', 'dir', 'ant', 'freq', 'time'],
                                        axesVals=[pols, dirs, ants, freqs, times], vals=vals,
                                        weights=np.ones_like(vals), weightDtype=weight_dtype)
            else:
                if vals is None:
                    vals = np.zeros([Nd, Na, Nf, Nt])
                self._solset.makeSoltab(name, axesNames=['dir', 'ant', 'freq', 'time'],
                                        axesVals=[dirs, ants, freqs, times], vals=vals, weights=np.ones_like(vals),
                                        weightDtype=weight_dtype)

    @property
    def allowed_soltabs(self):
        return ['phase', 'amplitude', 'tec', 'scalarphase', 'coords']

    def __getattr__(self, tab):
        """
        Links any attribute with an "axis name" to getValuesAxis("axis name")
        also links val and weight to the relative arrays.
        Parameter
        ----------
        axis : str
            The axis name.
        """
        #        with self:
        #            tabs = self._solset.getSoltabNames()
        tabs = self.allowed_soltabs
        tabs = ["weights_{}".format(t) for t in tabs] + ["axes_{}".format(t) for t in tabs] + tabs
        weight = False
        axes = False
        if tab in tabs:
            if tab.startswith("weights_"):
                tab = "".join(tab.split('weights_')[1:])
                weight = True
            if tab.startswith("axes_"):
                tab = "".join(tab.split('axes_')[1:])
                axes = True
            with self:
                soltab = self._solset.getSoltab("{}000".format(tab))
                if self._selection is None:
                    soltab.clearSelection()
                else:
                    soltab.setSelection(**self._selection)
                if not axes:
                    dtype = type(soltab.getAxisValues('ant', ignoreSelection=True).tolist()[0])
                    return soltab.getValues(reference=np.array(self.ref_ant).astype(dtype),
                                            weight=weight)
                else:
                    axisVals = {}
                    for axis in soltab.getAxesNames():
                        axisVals[axis] = soltab.getAxisValues(axis)
                    return axisVals
        else:
            return object.__getattribute__(self, tab)

    def __setattr__(self, tab, value):
        """
        Links any attribute with an "axis name" to getValuesAxis("axis name")
        also links val and weight to the relative arrays.
        Parameter
        ----------
        axis : str
            The axis name.
        value : array
            The array of the right shape for selection.
        """

        #        with self:
        #            tabs = self._solset.getSoltabNames()
        tabs = self.allowed_soltabs
        tabs = ["weights_{}".format(t) for t in tabs] + ["axes_{}".format(t) for t in tabs] + tabs
        weight = False
        axes = False
        if tab in tabs:
            if tab.startswith("weights_"):
                tab = "".join(tab.split('weights_')[1:])
                weight = True
            if tab.startswith("axes_"):
                tab = "".join(tab.split('axes_')[1:])
                axes = True
            with self:
                soltab = self._solset.getSoltab("{}000".format(tab))
                if self._selection is None:
                    soltab.clearSelection()
                else:
                    soltab.setSelection(**self._selection)
                if not axes:
                    return soltab.setValues(value, weight=weight)
                else:
                    assert isinstance(value, dict), "Axes must come in dict of 'name':vals"
                    for k, v in value.items():
                        soltab.setAxisValues(k, v)
        else:
            object.__setattr__(self, tab, value)

    def select(self, **selection):
        #        for key,item in selection.items():
        #            if key.startswith('not_'):
        #                flags = item.split('|')
        #                selection[key.split("not_")[1]] = "^(" + "".join(["((?!{}))".format(f) for f in flags]) + ".)*$"
        #                del selection[key]
        self._selection = selection

    def select_all(self):
        self._selection = None