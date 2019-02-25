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


def make_example_datapack(Nd, Nf, Nt, pols=None, time_corr=50., dir_corr=0.5 * np.pi / 180., tec_scale=0.02,
                          tec_noise=1e-3, name='test.hdf5', clobber=False):
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
    :return:
    """
    logging.info("=== Creating example datapack ===")
    name = os.path.abspath(name)
    if os.path.isfile(name) and clobber:
        os.unlink(name)

    datapack = DataPack(name, readonly=False)
    with datapack:
        datapack.switch_solset('sol000')
        datapack.add_antennas()
        datapack.add_sources(np.random.normal(np.pi / 4., np.pi / 180. * 2.5, size=[Nd, 2]))
        _, directions = datapack.sources
        _, antennas = datapack.antennas
        ref_dist = np.linalg.norm(antennas - antennas[0:1, :], axis=1)[None, None, :, None]  # 1,1,Na,1

        times = at.Time(np.linspace(0, Nt * 8, Nt)[:, None], format='gps').mjd * 86400.  # mjs
        freqs = np.linspace(120, 160, Nf) * 1e6
        if pols is not None:
            use_pols = True
            assert isinstance(pols, (tuple, list))
        else:
            use_pols = False
            pols = ['XX']

        tec_conversion = -8.440e9 / freqs  # Nf

        X = make_coord_array(directions / dir_corr, times / time_corr)  # Nd*Nt, 3
        X2 = np.sum((X[:, :, None] - X.T[None, :, :]) ** 2, axis=1)  # N,N
        K = tec_scale ** 2 * np.exp(-0.5 * X2)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))  # N,N
        Z = np.random.normal(size=(K.shape[0], len(pols)))  # N,npols
        tec = np.einsum("ab,bc->ac", L, Z)  # N,npols
        tec = tec.reshape((Nd, Nt, len(pols))).transpose((2, 0, 1))  # Npols,Nd,Nt
        tec = tec[:, :, None, :] * (0.2 + ref_dist / np.max(ref_dist))  # Npols,Nd,Na,Nt
        #         print(tec)
        tec += tec_noise * np.random.normal(size=tec.shape)
        phase = tec[:, :, :, None, :] * tec_conversion[None, None, None, :, None]  ##Npols,Nd,Na,Nf,Nt
        #         print(phase)
        phase = np.angle(np.exp(1j * phase))

        if not use_pols:
            phase = phase[0, ...]
            pols = None
        datapack.add_freq_dep_tab('phase', times=times[:, 0], freqs=freqs, pols=pols, vals=phase)
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


def K_parts(kern, X_list, X_dims_list):
    L = len(X_list)
    if len(X_dims_list) != L:
        raise ValueError("X_list and X_dims_list size must be same, and are {} and {}".format(L, len(X_dims_list)))

    # def _rm_antenna(X,dims):
    #     #Nt, Nd, Na, ndims
    #     X = tf.reshape(X,tf.concat([dims, tf.shape(X)[-1:]],axis=0))
    #     return flatten_batch_dims(X[:,:,1:,:],3)
    #
    # def _rm_direction(X,dims):
    #     #Nt, Nd, Na, ndims
    #     X = tf.reshape(X,tf.concat([dims, tf.shape(X)[-1:]],axis=0))
    #     return flatten_batch_dims(X[:,1:,:,:],3)
    #
    #
    # if kern.obs_type in ['DTEC','DDTEC']:
    #     for i in range(1,L):
    #         X_list[i] = _rm_antenna(X_list[i])
    #         X_dims_list[i][2] -= 1
    # if kern.obs_type in ['DDTEC']:
    #     for i in range(1,L):
    #         X_list[i] = _rm_direction(X_list[i])
    #         X_dims_list[i][1] -= 1

    if L == 1:
        # raise ValueError("X_list should have more than 1 element.")
        return kern.K(X_list[0], X_dims_list[0], X_list[0], X_dims_list[0])
    if L == 2:
        X = tf.concat(X_list, axis=0)

        K00 = kern.K(X_list[0], X_dims_list[0], X_list[0], X_dims_list[0])
        K01 = kern.K(X_list[0], X_dims_list[0], X_list[1], X_dims_list[1])
        K11 = kern.K(X_list[1], X_dims_list[1], X_list[1], X_dims_list[1])
        return tf.concat([tf.concat([K00, K01],axis=-1),
                          tf.concat([tf.transpose(K01, (1,0) if kern.squeeze else (0,2,1)), K11], axis=-1)],
                         axis=-2)


    K = [[None for _ in range(L)] for _ in range(L)]
    for i in range(L):
        for j in range(i,L):
            K_part = kern.K(X_list[i], X_dims_list[i], X_list[j], X_dims_list[j])
            K[i][j] = K_part
            if i == j:
                continue
            K[j][i] = tf.transpose(K_part,(1,0) if kern.squeeze else (0,2,1))
        K[i] = tf.concat(K[i],axis=1 if kern.squeeze else 2)
    K = tf.concat(K, axis=0 if kern.squeeze else 1)
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
    s = tf.reduce_mean(tf.matrix_diag_part(K),axis=-1)
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
