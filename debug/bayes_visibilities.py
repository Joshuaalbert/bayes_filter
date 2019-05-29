import tensorflow as tf
import numpy as np
import pylab as plt
import os
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au
from bayes_filter.datapack import _load_array_file, DataPack
from bayes_filter.feeds import Feed
from collections import namedtuple
from bayes_filter.frames import ENU
from scipy.spatial.distance import pdist
import tensorflow_probability as tfp
from bayes_filter.misc import flatten_batch_dims

C = 299792458.

# class SampleTimeFeed(Feed):
#     def __init__(self, time0, step=1, end=None):
#         self._step = step
#         self._end = end
#         self.step = tf.convert_to_tensor(step, tf.int32)
#         self.index_feed = tf.data.Dataset.from_generator(self.index_generator,
#                                                          (tf.int32, tf.int32),
#                                                          (tf.TensorShape([]), tf.TensorShape([])))
#         self.feed = self.index_feed
#
#     def index_generator(self):
#         if self._end is None:
#             i = 0
#             while True:
#                 i += self._step
#                 yield i - self._step, i
#         else:
#             i = 0
#             while i < self._end:
#                 i += self._step
#                 yield i - self._step, min(i, self._end)

def make_obs_setup(Nd, time_total=10, fov=2.5, interval=1, Nf=2, seed = 0):
    np.random.seed(seed)
    antenna_labels, antennas = _load_array_file(DataPack.lofar_array)
    antennas = ac.SkyCoord(x=antennas[:,0] * au.m, y=antennas[:,1] * au.m, z=antennas[:,2] * au.m, frame='itrs')
    array_center = np.mean(antennas.cartesian.xyz, axis=1)
    array_center = ac.SkyCoord(x=array_center[0], y=array_center[1], z=array_center[2], frame='itrs')
    time0 = at.Time("2019-01-01T00:00:00.000", format='isot')
    altaz = ac.AltAz(location=array_center.earth_location, obstime=time0)
    up = ac.SkyCoord(alt=90. * au.deg, az=0. * au.deg, frame=altaz).transform_to('icrs')
    directions = np.stack([np.random.normal(up.ra.rad, np.pi / 180. * fov, size=[Nd]),
                           np.random.normal(up.dec.rad, np.pi / 180. * fov, size=[Nd])], axis=1)
    directions = ac.SkyCoord(directions[:,0]*au.rad, directions[:,1]*au.rad, frame='icrs')
    patch_names = np.array(['patch_{:04d}'.format(i) for i in range(len(directions))])

    Nt = int(time_total/interval)

    times = at.Time(time0.gps + np.linspace(0, time_total, Nt), format='gps')#.mjd[:, None] * 86400.  # mjs
    freqs = np.linspace(120, 160, Nf) * 1e6

    fluxes = np.random.uniform(low=0.1, high=1., size=Nd)

    return dict(times=times, freqs=freqs, fluxes=fluxes, antennas=antennas, directions=directions)

def simulate_vis(obs_setup, vis_noise=None, vis_frac_noise=0.5, seed=0):
    np.random.seed(seed)
    times = obs_setup['times']
    directions = obs_setup['directions']
    Nd = len(directions)
    antennas = obs_setup['antennas']
    Na = len(antennas)
    freqs = obs_setup['freqs']
    Nf = len(freqs)
    fluxes = obs_setup['fluxes']
    array_center = np.mean(antennas.cartesian.xyz, axis=1)
    array_center = ac.SkyCoord(x=array_center[0], y=array_center[1], z=array_center[2], frame='itrs')
    visibilities = []
    obs_times = []

    factor = -2.*np.pi*1j*freqs/C# Nf


    geometric = []
    for time in times:
        enu = ENU(location=array_center.earth_location, obstime=time)
        k = directions.transform_to(enu).cartesian.xyz.value #3, Nd
        x = antennas.transform_to(enu).cartesian.xyz.to(au.m).value #3, Na
        kx = np.einsum('si,sj->ij', k, x) #Nd, Na
        dkx = kx[:,:,None] - kx[:,None,:]#Nd, Na, Na
        geometric.append(np.exp(factor[None, None, None, :]*dkx[:, :, :, None]))

    dt = times[1].gps - times[0].gps

    geometric = np.stack(geometric,axis=-1)
    vis = np.sum(fluxes[:,None,None,None,None]*geometric,axis=0)*dt
    if vis_noise is not None:
        vis += vis_noise*np.random.normal(size=vis.shape) + vis_noise*1j*np.random.normal(size=vis.shape)
    else:
        vis += vis*vis_frac_noise * (np.random.normal(size=vis.shape) + 1j*np.random.normal(size=vis.shape))
    return dict(sim_visibilities=vis, geometric=geometric, times=times, uncert=vis*vis_frac_noise)

class Target:
    def __init__(self, obs_vis, uncert, times, geometric):
        self.obs_vis = tf.reshape(tf.convert_to_tensor(obs_vis, tf.complex128), [-1])#Na Na Nf Nt
        self.uncert = tf.reshape(tf.convert_to_tensor(uncert, tf.float64), [-1])#Na Na Nf Nt

        self.obs_vis_real = tf.real(self.obs_vis)
        self.obs_vis_imag = tf.imag(self.obs_vis)

        self.times = tf.convert_to_tensor(times) #Nt
        self.dt = tf.cast(self.times[1] - self.times[0], tf.float64)
        self.geometric = tf.convert_to_tensor(geometric, tf.complex128) #Nd, Na, Na, Nf, Nt
        self.geometric = tf.reshape(self.geometric, tf.concat([tf.shape(self.geometric)[0:1], [-1]],axis=0)) #Nd, Na Na Nf Nt
        self.geometric_real = tf.real(self.geometric)
        self.geometric_imag = tf.imag(self.geometric)

        self.fluxes_bijector = tfp.bijectors.Exp()
        self.sigma_bijector = tfp.bijectors.Exp()


    def forward(self,fluxes):

        geometric_real = self.geometric_real#tf.gather(self.geometric_real, selection, axis=1)
        geometric_imag = self.geometric_imag#tf.gather(self.geometric_imag, selection, axis=1)

        vis_real = tf.reduce_sum(fluxes[:, :, None] * geometric_real[None, :, :],
                                 axis=1) * self.dt
        vis_imag = tf.reduce_sum(fluxes[:, :, None] * geometric_imag[None, :, :],
                                 axis=1) * self.dt # Na Na Nf Nt*
        return vis_real, vis_imag

    def log_prob(self, log_fluxes):
        fluxes = self.fluxes_bijector.forward(log_fluxes)# num_chains, Nd
        # sigma = self.sigma_bijector.forward(log_sigma)# num_chains, 1

        # n = tf.cast(tf.shape(self.obs_vis_real)[0], dtype=sigma.dtype)
        # p_erdos = 2*tf.math.log(n) / n
        # selection = tf.where(
        #     tf.equal(tfp.distributions.Binomial(total_count=tf.constant(1., dtype=sigma.dtype),
        #                                         probs=1.-p_erdos).sample(tf.cast(n, tf.int32)),
        #        selection = tf.where(
        #     tf.equal(tfp.distributions.Binomial(total_count=tf.constant(1., dtype=sigma.dtype),
        #                                         probs=1.-p_erdos).sample(tf.cast(n, tf.int32)),
        #              tf.constant(1., dtype=sigma.dtype)))[:,0]



        model_vis = self.forward(fluxes)#self.forward(fluxes, selection)

        obs_vis_real = self.obs_vis_real#tf.gather(self.obs_vis_real, selection, axis=0)
        obs_vis_imag = self.obs_vis_imag#tf.gather(self.obs_vis_imag, selection, axis=0)

        vis_sigma = self.uncert#sigma# * tf.sqrt(tf.square(model_vis[0]) + tf.square(model_vis[1]))

        likelihood = tf.reduce_sum(tfp.distributions.Normal(loc=model_vis[0], scale=vis_sigma).log_prob(obs_vis_real[None,:]) \
                     + tfp.distributions.Normal(loc=model_vis[1], scale=vis_sigma).log_prob(obs_vis_imag[None,:]),axis=1)

        # prior = tfp.distributions.Normal(loc=tf.constant(0.1,sigma.dtype), scale=tf.constant(0.1, sigma.dtype)).log_prob(sigma[:,0])

        return likelihood# + prior



def run(output_folder='bayes_vis_output_1percent_10'):
    output_folder = os.path.abspath(output_folder)
    num_chains = 4
    Nd = 10
    obs_setup = make_obs_setup(Nd=Nd, time_total=2, fov=2.5, interval=1, Nf=2)
    sim = simulate_vis(obs_setup,vis_frac_noise=0.01)
    with tf.Session(graph=tf.Graph()) as sess:
        target = Target(sim['sim_visibilities'],sim['uncert'], sim['times'].gps, sim['geometric'])
        ###
        hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target.log_prob,
            num_leapfrog_steps=2,
            step_size=[1.]),
            num_adaptation_steps=800,
            target_accept_prob=tf.constant(0.6,tf.float64),
            adaptation_rate=0.1)

        # Run the chain (with burn-in maybe).
        # last state as initial point (mean of each chain)

        def trace_fn(_, pkr):
            # print(pkr)
            return (pkr.inner_results.log_accept_ratio,
                    pkr.inner_results.accepted_results.step_size)

        init_state = [#tf.cast(target.sigma_bijector.inverse(tf.random.uniform([num_chains, 1], 0.01,  0.1)), dtype=tf.float64),
                      tf.cast(target.fluxes_bijector.inverse(tf.random.uniform([num_chains, Nd], 0.1,  1.)), dtype=tf.float64)]

        # u = target.log_prob(*init_state)
        # print(tf.gradients(u,init_state))
        samples, (log_accept_ratio, stepsizes) = tfp.mcmc.sample_chain(
            num_results=10000,
            num_burnin_steps=1000,
            trace_fn=trace_fn,
            return_final_kernel_results=False,
            current_state=init_state,
            kernel=hmc,
            parallel_iterations=10)

        rhat = tfp.mcmc.potential_scale_reduction(samples)

        next_stepsizes = [#tf.sqrt(tf.reduce_mean(tfp.stats.variance(samples[0]))),
                          tf.sqrt(tf.reduce_mean(tfp.stats.variance(samples[0])))]

        final_state = dict(#sigma=flatten_batch_dims(target.sigma_bijector.forward(samples[0]),-1),
                       fluxes=flatten_batch_dims(target.fluxes_bijector.forward(samples[0]),-1),
                           stepsizes=stepsizes,
                           next_stepsizes=next_stepsizes,
                           rhat=rhat)

        res = sess.run(final_state)
        print(res['next_stepsizes'])

        os.makedirs(output_folder, exist_ok=True)
        idx = np.arange(Nd)
        plt.boxplot(res['fluxes'], positions=idx, sym="")
        plt.scatter(idx,obs_setup['fluxes'])
        plt.savefig(os.path.join(output_folder,'fluxes.png'))
        plt.close('all')

        plt.boxplot(res['fluxes']-obs_setup['fluxes'], positions=idx, sym="")
        plt.savefig(os.path.join(output_folder, 'res_fluxes.png'))
        plt.close('all')

        plt.hist(np.median(res['fluxes'] - obs_setup['fluxes'], axis=0), bins = max(10, int(np.sqrt(Nd))))
        plt.savefig(os.path.join(output_folder, 'hist_res_fluxes.png'))
        plt.close('all')

        # plt.hist(res['sigma'],bins=int(np.sqrt(res['sigma'].size)))
        # plt.savefig(os.path.join(output_folder,'sigma.png'))
        # plt.close('all')

        plt.plot(res['stepsizes'])
        plt.savefig(os.path.join(output_folder,'stepsizes.png'))
        plt.close('all')

        # plt.bar([0],res['rhat'][0])
        # plt.savefig(os.path.join(output_folder,'rhat_sigma.png'))
        # plt.close('all')

        plt.bar(idx, res['rhat'][0])
        plt.savefig(os.path.join(output_folder, 'rhat_fluxes.png'))
        plt.close('all')





if __name__=='__main__':
    run()