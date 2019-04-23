import tensorflow as tf
import numpy as np
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
from .settings import dist_type, float_type, jitter
from timeit import default_timer
from .odeint import odeint
from astropy.io import fits
from matplotlib.patches import Circle
import pylab as plt
from scipy.spatial.distance import pdist
from . import logging
import os
from .datapack import DataPack
import networkx as nx
from . import logging
from collections import namedtuple
from .coord_transforms import itrs_to_enu_with_references,tf_coord_transform
from .kernels import DTECIsotropicTimeGeneral

def dict2namedtuple(d, name="Result"):
    res = namedtuple(name, list(d.keys()))
    return res(**d)


def graph_store_set(key, value, graph = None, name="graph_store"):
    if isinstance(key,(list,tuple)):
        if len(key) != len(value):
            raise IndexError("keys and values must be equal {} {}".format(len(key), len(value)))
        for k,v in zip(key,value):
            graph_store_set(k,v,graph=graph, name=name)
    values_key = "{}_values".format(name)
    keys_key = "{}_keys".format(name)
    if graph is None:
        graph = tf.get_default_graph()
    with graph.as_default():
        graph.add_to_collection(keys_key, key)
        graph.add_to_collection(values_key, value)

def graph_store_get(key, graph = None, name="graph_store"):
    if isinstance(key, (list,tuple)):
        return [graph_store_get(k) for k in key]
    values_key = "{}_values".format(name)
    keys_key = "{}_keys".format(name)
    if graph is None:
        graph = tf.get_default_graph()
    with graph.as_default():
        keys = graph.get_collection(keys_key)
        values = graph.get_collection(values_key)
        if key not in keys:
            raise KeyError("{} not in the collection".format(key))
        index = keys_key.index(key)
        return values[index]

def plot_graph(tf_graph,ax=None, filter=False):
    '''Plot a DAG using NetworkX'''

    def children(op):
        return set(op for out in op.outputs for op in out.consumers())

    def get_graph(tf_graph):
        """Creates dictionary {node: {child1, child2, ..},..} for current
        TensorFlow graph. Result is compatible with networkx/toposort"""

        ops = tf_graph.get_operations()
        g = {}
        for op in ops:
            c = children(op)
            if len(c) == 0 and filter:
                continue
            g[op] = c
        return g

    def mapping(node):
        return node.name
    G = nx.DiGraph(get_graph(tf_graph))
    nx.relabel_nodes(G, mapping, copy=False)
    nx.draw(G, cmap = plt.get_cmap('jet'), with_labels = True, ax=ax)

def make_example_datapack(Nd, Nf, Nt, pols=None, gain_noise=0.05, name='test.hdf5', obs_type='DDTEC',clobber=False):
    """

    :param Nd:
    :param Nf:
    :param Nt:
    :param pols:
    :param time_corr:
    :param dir_corr:
    :param tec_scale:
    :param tec_noise:
    :param name:
    :param clobber:
    :return: DataPack
        New object referencing a file
    """
    from bayes_filter.feeds import TimeFeed, IndexFeed, CoordinateFeed, init_feed


    logging.info("=== Creating example datapack ===")
    name = os.path.abspath(name)
    if os.path.isfile(name) and clobber:
        os.unlink(name)

    datapack = DataPack(name, readonly=False)
    with datapack:
        datapack.add_solset('sol000')
        time0 = at.Time("2019-01-01T00:00:00.000", format='isot')
        altaz = ac.AltAz(location=datapack.array_center.earth_location, obstime=time0)
        up = ac.SkyCoord(alt=90.*au.deg,az=0.*au.deg,frame=altaz).transform_to('icrs')
        directions = np.stack([np.random.normal(up.ra.rad, np.pi / 180. * 2.5, size=[Nd]),
                               np.random.normal(up.dec.rad, np.pi / 180. * 2.5, size=[Nd])],axis=1)
        datapack.set_directions(None,directions)
        patch_names, directions = datapack.directions
        antenna_labels, antennas = datapack.antennas
        antennas /= 1000.
        # print(directions)
        Na = antennas.shape[0]


        ref_dist = np.linalg.norm(antennas - antennas[0:1, :], axis=1)[None, None, :, None]  # 1,1,Na,1

        times = at.Time(time0.gps+np.linspace(0, Nt * 8, Nt), format='gps').mjd[:, None] * 86400.  # mjs
        freqs = np.linspace(120, 160, Nf) * 1e6
        if pols is not None:
            use_pols = True
            assert isinstance(pols, (tuple, list))
        else:
            use_pols = False
            pols = ['XX']
        Npol = len(pols)
        tec_conversion = -8.440e6 / freqs  # Nf
        phase = []
        with tf.Session(graph=tf.Graph()) as sess:
            index_feed = IndexFeed(1)
            time_feed = TimeFeed(index_feed, times)
            coord_feed = CoordinateFeed(time_feed, directions, antennas, coord_map=tf_coord_transform(
                itrs_to_enu_with_references(antennas[0, :], [up.ra.rad, up.dec.rad], antennas[0, :])))
            init, next = init_feed(coord_feed)
            kern = DTECIsotropicTimeGeneral(0.17,obs_type=obs_type,b=100.,lengthscales=15.,kernel_params={'resolution':5})
            K = kern.K(next)
            Z = tf.random_normal(shape=tf.shape(K)[0:1],dtype=K.dtype)
            ddtec = tf.matmul(safe_cholesky(K),Z[:,None])[:,0]
            sess.run(init)
            for t in times:

                # plt.imshow(sess.run(K))
                # plt.show()
                phase.append(sess.run(ddtec)[:,None]*tec_conversion)
        phase = np.concatenate(phase,axis=0)
        phase = np.reshape(phase,(Nt, Nd, Na, Nf))
        phase = np.transpose(phase, (1,2,3,0))#Nd, Na, Nf, Nt
        phase = np.tile(phase[None,...], (Npol,1,1,1,1))#Npol, Nd, Na, Nf, Nt

        phase = np.angle(np.exp(1j*phase) + gain_noise * np.random.normal(size=phase.shape))




        # X = make_coord_array(directions / dir_corr, times / time_corr)  # Nd*Nt, 3
        # X2 = np.sum((X[:, :, None] - X.T[None, :, :]) ** 2, axis=1)  # N,N
        # K = tec_scale ** 2 * np.exp(-0.5 * X2)
        # L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))  # N,N
        # Z = np.random.normal(size=(K.shape[0], len(pols)))  # N,npols
        # tec = np.einsum("ab,bc->ac", L, Z)  # N,npols
        # tec = tec.reshape((Nd, Nt, len(pols))).transpose((2, 0, 1))  # Npols,Nd,Nt
        # tec = tec[:, :, None, :] * (0.2 + ref_dist / np.max(ref_dist))  # Npols,Nd,Na,Nt
        # #         print(tec)
        # tec += tec_noise * np.random.normal(size=tec.shape)
        # phase = tec[:, :, :, None, :] * tec_conversion[None, None, None, :, None]  ##Npols,Nd,Na,Nf,Nt
        # #         print(phase)
        # phase = np.angle(np.exp(1j * phase))
        #
        # if not use_pols:
        #     phase = phase[0, ...]
        #     pols = None
        datapack.add_soltab('phase000', values=phase, ant=antenna_labels, dir = patch_names, time=times[:, 0], freq=freqs, pol=pols)
        datapack.phase = phase
        return datapack


def get_screen_directions(srl_fits='/home/albert/git/bayes_tec/scripts/data/image.pybdsm.srl.fits', max_N = None):
    """Given a srl file containing the sources extracted from the apparent flux image of the field,
    decide the screen directions

    :param srl_fits: str
        The path to the srl file, typically created by pybdsf
    :return: float, array [N, 2]
        The `N` sources' coordinates as an ``astropy.coordinates.ICRS`` object
    """
    hdu = fits.open(srl_fits)
    data = hdu[1].data

    arg = np.argsort(data['Total_Flux'])[::-1]

    #75MHz NLcore
    exclude_radius = 7.82/2.
    flux_limit = 0.05

    ra = []
    dec = []
    idx = []
    for i in arg:
        if data['Total_Flux'][i] < flux_limit:
            continue
        ra_ = data['RA'][i]
        dec_ = data['DEC'][i]
        radius = np.sqrt((ra_ - 126)**2 + (dec_ - 65)**2)
        if radius > exclude_radius:
            continue
        elif radius > 4.75/2.:
            high_flux = 0.5
            threshold = 1.
            f_steps = 10**np.linspace(np.log10(high_flux), np.log10(np.max(data['Total_flux'])), 1000)[::-1]
            f_spacing = 10**(np.linspace(np.log10(1./60.),np.log10(10./60.),1000))
        elif radius > 3.56/2.:
            high_flux = 0.1
            threshold = 20./60.
            f_steps = 10**np.linspace(np.log10(high_flux), np.log10(np.max(data['Total_flux'])), 1000)[::-1]
            f_spacing = 10**(np.linspace(np.log10(1./60.),np.log10(10./60.),1000))
        else:
            high_flux = 0.05
            threshold = 15./60.
            f_steps = 10**np.linspace(np.log10(high_flux), np.log10(np.max(data['Total_flux'])), 1000)[::-1]
            f_spacing = 10**(np.linspace(np.log10(1./60.),np.log10(10./60.),1000))
        if data['Total_Flux'][i] > high_flux:
            a = np.searchsorted(f_steps, data['Total_Flux'][i])-1
            threshold = f_spacing[a]
        if len(ra) == 0:
            ra.append(ra_)
            dec.append(dec_)
            idx.append(i)
            continue
        dist = np.sqrt(np.square(np.subtract(ra_, ra)) + np.square(np.subtract(dec_,dec)))
        if np.all(dist > threshold):
            ra.append(ra_)
            dec.append(dec_)
            idx.append(i)
            continue

    f = data['Total_Flux'][idx]
    ra = data['RA'][idx]
    dec = data['DEC'][idx]
    c = data['S_code'][idx]

    if max_N is not None:
        arg = np.argsort(f)[::-1][:max_N]
        f = f[arg]
        ra = ra[arg]
        dec = dec[arg]
        c = c[arg]

    plt.scatter(ra,dec,c=np.linspace(0.,1.,len(ra)),cmap='jet',s=np.sqrt(10000.*f),alpha=1.)

    target = Circle((126., 65.),radius = 3.56/2.,fc=None, alpha=0.2)
    ax = plt.gca()
    ax.add_patch(target)
    target = Circle((126., 65.),radius = 4.75/2.,fc=None, alpha=0.2)
    ax = plt.gca()
    ax.add_patch(target)
    plt.title("Brightest {} sources".format(len(f)))
    plt.xlabel('ra (deg)')
    plt.xlabel('dec (deg)')
    plt.savefig("scren_directions.png")
    interdist = pdist(np.stack([ra,dec],axis=1)*60.)
    plt.hist(interdist,bins=len(f))
    plt.title("inter-facet distance distribution")
    plt.xlabel('inter-facet distance [arcmin]')
    plt.savefig("interfacet_distance_dist.png")
    return ac.ICRS(ra=ra*au.deg, dec=dec*au.deg)

def random_sample(t, n=None):
    """
    Randomly draw `n` slices from `t`.

    :param t: float_type, tf.Tensor, [M, ...]
        Tensor to draw from
    :param n: tf.int32
        number of slices to draw. None means shuffle.
    :return: float_type, tf.Tensor, [n, ...]
        The randomly sliced tensor.
    """
    if isinstance(t, (list, tuple)):
        with tf.control_dependencies([tf.assert_equal(tf.shape(t[i])[0], tf.shape(t[0])[0]) for i in range(len(t))]):
            if n is None:
                n = tf.shape(t[0])[0]
            n = tf.minimum(n, tf.shape(t[0])[0])
            idx = tf.random_shuffle(tf.range(n))
            return [tf.gather(t[i], idx, axis=0) for i in range(len(t))]
    else:
        if n is None:
            n = tf.shape(t)[0]
        n = tf.minimum(n, tf.shape(t)[0])
        idx = tf.random_shuffle(tf.range(n))
        return tf.gather(t, idx, axis=0)

def K_parts(kern, *X):
    L = len(X)
    K = [[None for _ in range(L)] for _ in range(L)]
    for i in range(L):
        for j in range(i,L):
            K_part = kern.matrix(X[i], X[j])
            K[i][j] = K_part
            if i == j:
                continue
            K[j][i] = tf.transpose(K_part,(1,0) if kern.squeeze else (0,2,1))
        K[i] = tf.concat(K[i],axis=-1)
    K = tf.concat(K, axis=-2)
    return K

def log_normal_solve_fwhm(a,b,D=0.5):
    """Solve the parameters for a log-normal distribution given the 1/D power at limits a and b.

    :param a: float
        The lower D power
    :param b: float
        The upper D power
    :return: tuple of (mu, stddev) parametrising the log-normal distribution
    """
    if b < a:
        raise ValueError("b should be greater than a")
    lower = np.log(a)
    upper = np.log(b)
    d = upper - lower #2 sqrt(2 sigma**2 ln(1/D))
    sigma2 = 0.5*(0.5*d)**2/np.log(1./D)
    s = upper + lower #2 (mu - sigma**2)
    mu = 0.5*s + sigma2
    return np.array(mu,dtype=np.float64), np.array(np.sqrt(sigma2),dtype=np.float64)



def diagonal_jitter(N):
    """
    Create diagonal matrix with jitter on the diagonal

    :param N: int, tf.int32
        The size of diagonal
    :return: float_type, Tensor, [N, N]
        The diagonal matrix with jitter on the diagonal.
    """
    return tf.diag(tf.fill([N],tf.convert_to_tensor(jitter,float_type)))

def safe_cholesky(K):
    n = tf.shape(K)[-1]
    s = tf.reduce_mean(tf.matrix_diag_part(K),axis=-1, keep_dims=True)[..., None]
    K = K/s
    L = tf.sqrt(s)*tf.cholesky(K + diagonal_jitter(n))
    return L

def timer():
    """
    Return system time as tensorflow op
    :return:
    """
    return tf.py_function(default_timer, [], float_type)

def flatten_batch_dims(t, num_batch_dims=None):
    """
    Flattens the first `num_batch_dims`
    :param t: Tensor [b0,...bB, n0,...nN]
        Flattening happens for first `B` dimensions
    :param num_batch_dims: int, or tf.int32
        Number of dims in batch to flatten. If None then all but last. If < 0 then count from end.
    :return: Tensor [b0*...*bB, n0,...,nN]
    """
    shape = tf.shape(t)
    if num_batch_dims is None:
        num_batch_dims = tf.size(shape) - 1
    out_shape = tf.concat([[-1], shape[num_batch_dims:]],axis=0)
    return tf.reshape(t,out_shape)

def make_coord_array(*X, flat=True, coord_map=None):
    """
    Create the design matrix from a list of coordinates
    :param X: list of length p of float, array [Ni, D]
        Ni can be different for each coordinate array, but D must be the same.
    :param flat: bool
        Whether to return a flattened representation
    :param coord_map: callable(coordinates), optional
            If not None then get mapped over the coordinates
    :return: float, array [N0,...,Np, D] if flat=False else [N0*...*Np, D]
        The coordinate design matrix
    """

    if coord_map is not None:
        X = [coord_map(x) for x in X]

    def add_dims(x, where, sizes):
        shape = []
        tiles = []
        for i in range(len(sizes)):
            if i not in where:
                shape.append(1)
                tiles.append(sizes[i])
            else:
                shape.append(-1)
                tiles.append(1)
        return np.tile(np.reshape(x, shape), tiles)

    N = [x.shape[0] for x in X]
    X_ = []

    for i, x in enumerate(X):
        for dim in range(x.shape[1]):
            X_.append(add_dims(x[:, dim], [i], N))
    X = np.stack(X_, axis=-1)
    if not flat:
        return X
    return np.reshape(X, (-1, X.shape[-1]))


def load_array_file(array_file):
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
        labels = np.array(["ant{:02d}".format(i) for i in range(Nantenna)])
        diameters = None
    return np.array(labels).astype(np.str_), locs.cartesian.xyz.to(dist_type).value.transpose()

# def save_array_file(array_file):
#     import time
#     ants = _solset.getAnt()
#     labels = []
#     locs = []
#     for label, pos in ants.items():
#         labels.append(label)
#         locs.append(pos)
#     Na = len(labels)
# with open(array_file, 'w') as f:
#     f.write('# Created on {0} by Joshua G. Albert\n'.format(time.strftime("%a %c", time.localtime())))
#     f.write('# ITRS(m)\n')
#     f.write('# X\tY\tZ\tlabels\n')
#     i = 0
#     while i < Na:
#         f.write(
#             '{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:d}\t{4}'.format(locs[i][0], locs[i][1], locs[i][2], labels[i]))
#         if i < Na - 1:
#             f.write('\n')
#         i += 1
