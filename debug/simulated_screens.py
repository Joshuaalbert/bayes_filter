from bayes_filter.tests.common_setup import *
import tensorflow_probability as tfp
import tensorflow as tf
from bayes_filter.misc import make_example_datapack, safe_cholesky, flatten_batch_dims, make_coord_array, random_sample, sample_laplacian, sqrt_with_finite_grads, load_array_file
from bayes_filter.coord_transforms import itrs_to_enu_with_references
from bayes_filter.feeds import init_feed
from bayes_filter.targets import DTECToGainsTarget
from bayes_filter.processes import DTECProcess
from bayes_filter.kernels import DTECIsotropicTimeGeneral
from bayes_filter.sample import sample_chain
from bayes_filter import float_type, TEC_CONV
from bayes_filter.sgd import stochastic_gradient_descent
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au




def run_target(tf_session, lofar_array):
    import os
    Nf = 10
    N = 5
    Nd = N*N
    Nt = 2
    shuffle = np.random.choice(Nd*62, Nd*62, replace=False)
    inv_sort = np.argsort(shuffle)
    data_slice, screen_slice = np.arange(Nd*62)[:Nd*31], np.arange(Nd*62)[Nd*31:]#shuffle[:Nd*31], shuffle[Nd*31:]

    Na = len(lofar_array[0])
    num_chains = 5
    gain_noise = 0.2

    output_folder_ = os.path.abspath(os.path.join('simulated_screens_output', 'run12'))
    os.makedirs(output_folder_, exist_ok=True)
    with tf_session.graph.as_default():
        tf.random.set_random_seed(0)
        np.random.seed(0)
        freqs = tf.constant(np.linspace(120e6, 160e6, Nf), dtype=float_type)

        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd * 86400. + tf.cast(tf.linspace(0., 8.*Nt-8., Nt)[:, None], float_type)
        time_feed = TimeFeed(index_feed, times)

        array_center = ac.SkyCoord(lofar_array[1][:,0].mean()*au.m,lofar_array[1][:,1].mean()*au.m,lofar_array[1][:,2].mean()*au.m, frame='itrs')
        altaz = ac.AltAz(location=array_center.earth_location, obstime=obstime_init)
        up = ac.SkyCoord(alt=90. * au.deg, az=0. * au.deg, frame=altaz).transform_to('icrs')

        ra = np.linspace(up.ra.rad - np.pi / 180. * 2.5,up.ra.rad + np.pi / 180. * 2.5,N)
        dec = np.linspace(up.dec.rad - np.pi / 180. * 2.5,up.dec.rad + np.pi / 180. * 2.5,N)

        Xd = tf.constant(make_coord_array(ra[:,None], dec[:,None], flat=True), dtype=float_type)

        Xa = tf.constant(lofar_array[1], dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(
                                        itrs_to_enu_with_references(lofar_array[1][0, :], [up.ra.rad, up.dec.rad],
                                                                    lofar_array[1][0, :])))
        init, X = init_feed(coord_feed)
        # X = tf.concat([tf.gather(X, data_slice, axis=0), tf.gather(X,screen_slice,axis=0)], axis=0)
        # data_slice = np.arange(Nd*62)[:Nd*31]
        # screen_slice = np.arange(Nd * 62)[Nd * 31:]
        tf_session.run(init)

        kern = DTECIsotropicTimeGeneral(variance=1.0, lengthscales=10.0,
                                        a=250., b=50., timescale=30., ref_location=[0., 0., 0.],
                                        fed_kernel='RBF', obs_type='DDTEC', kernel_params=dict(resolution=3),
                                        squeeze=True)

        K = kern.K(X)
        L = safe_cholesky(K)

        Ftrue = 4.5*tf.matmul(L, tf.random.normal(shape=[tf.shape(K)[0], 1], dtype=float_type))
        invfreqs = TEC_CONV * tf.math.reciprocal(freqs)

        phase_true = Ftrue * invfreqs
        Yimag_true, Yreal_true = tf.sin(phase_true), tf.cos(phase_true)

        Yimag = Yimag_true + gain_noise * sample_laplacian(shape=tf.shape(Yimag_true), dtype=float_type)#tf.random.random_normal(shape=tf.shape(Yimag_true), dtype=float_type)
        Yreal = Yreal_true + gain_noise * sample_laplacian(shape=tf.shape(Yreal_true), dtype=float_type)#tf.random_normal(shape=tf.shape(Yreal_true), dtype=float_type)

        phase_data = tf.atan2(Yimag, Yreal)

        # mag_data = tf.sqrt(tf.square(Yimag) + tf.square(Yreal))
        # Yimag /= mag_data
        # Yreal /= mag_data

        ###
        # sampling

        L_sample = tf.gather(L, data_slice,axis=0)#[data_slice,:]#tf.tile(L[None, :, :], (num_chains, 1, 1))
        Yimag_sample = tf.gather(Yimag, data_slice,axis=0)#Yimag[data_slice, :]
        Yreal_sample = tf.gather(Yreal, data_slice,axis=0)#Yreal[data_slice, :]

        N_data = tf.shape(Yreal_sample)[0]
        N_total = tf.shape(Yreal)[0]
        ratio = tf.cast(N_total,float_type)/tf.cast(N_data,float_type)

        def logp(log_y_sigma, log_amp, f):
            # with jit_scope():

            # log_y_sigma = log_y_sigma/log_y_sigma*tf.math.log(tf.constant(0.1, log_y_sigma.dtype))

            prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f),
                                                             scale_identity_multiplier=1.).log_prob(f)

            y_sigma = tf.exp(log_y_sigma)
            amp = tf.exp(log_amp)
            # L_sample_ = amp[:, :, None] * L_sample

            # L_ij f_sj -> f_sj L_ji
            dtec = amp * tf.matmul(f, L_sample, transpose_b=True)

            Yimag_model = tf.sin(dtec[:,:,None] * invfreqs)
            Yreal_model = tf.cos(dtec[:,:,None] * invfreqs)

            likelihood = -tf.math.reciprocal(y_sigma[:, :, None])*sqrt_with_finite_grads(
                tf.math.square(Yimag_sample[None, :, :] - Yimag_model) +
                tf.math.square(Yreal_sample[None, :, :] - Yreal_model)) \
                         - log_y_sigma[:,:,None]

            # likelihood = tfp.distributions.Laplace(loc=Yimag[None, :, :], scale=y_sigma[:, :, None]).log_prob(
            #     Yimag_model) \
            #              + tfp.distributions.Laplace(loc=Yreal[None, :, :], scale=y_sigma[:, :, None]).log_prob(
            #     Yreal_model)

            logp = tf.reduce_sum(likelihood, axis=[1, 2]) + prior \
                   + tfp.distributions.Normal(tf.constant(0.1, dtype=float_type), tf.constant(0.01, dtype=float_type)).log_prob(y_sigma[:, 0])
            return logp



        step_size = [tf.constant(0.06, dtype=float_type),tf.constant(0.21, dtype=float_type),tf.constant(0.9, dtype=float_type)]


        ###

        dtec_process = DTECProcess()
        dtec_process.setup_process(tf.gather(X, data_slice, axis=0),
                                   tf.gather(X, screen_slice,axis=0),
                                   fed_kernel='RBF',
                                   recalculate_prior=False,
                                   L = L)

        target = DTECToGainsTarget(dtec_process)
        target.setup_target(Yreal_sample, Yimag_sample, freqs=freqs)

        hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=lambda log_y_sigma, log_amp, white_dtec: target.log_prob(log_amp, log_y_sigma, white_dtec),#lambda log_amp, white_dtec: logp(tf.math.log(0.1), log_amp, white_dtec),
            num_leapfrog_steps=3,
            step_size=step_size,
            state_gradients_are_stopped=True),
            num_adaptation_steps=2000,
            target_accept_prob=tf.constant(0.65, dtype=float_type),
            adaptation_rate=0.10)

        # Run the chain (with burn-in maybe).
        # last state as initial point (mean of each chain)

        def trace_fn(_, pkr):
            return (pkr.inner_results.log_accept_ratio,
                    pkr.inner_results.accepted_results.step_size,
                    pkr.inner_results.accepted_results.target_log_prob)

        init_state = [tf.constant(np.log(np.random.uniform(0.01, 0.2, (num_chains, 1))), float_type),
                      tf.constant(np.log(np.random.uniform(0.5, 2., (num_chains, 1))), float_type),
                      # tf.zeros([num_chains, 567], dtype=float_type)
                      0.3*tf.random.normal(tf.concat([[num_chains], tf.shape(Ftrue)[0:1]], axis=0), dtype=float_type)
                      ]

        init_state, loss_array = stochastic_gradient_descent(logp, init_state, 0, learning_rate=1e-5)
        # samples = tfp.mcmc.sample_chain(  # ,
        #     num_results=10000,
        #     num_burnin_steps=5000,
        #     trace_fn=None,#trace_fn,  # trace_step,
        #     return_final_kernel_results=False,
        #     current_state=init_state,
        #     kernel=hmc,
        #     parallel_iterations=10)

        samples = sample_chain(  # ,
            num_results=5000,
            num_burnin_steps=2000,
            trace_fn=None,  # trace_fn,  # trace_step,
            return_final_kernel_results=False,
            current_state=init_state,
            kernel=hmc,
            parallel_iterations=10)


        rhats = [ ]
        for l in np.linspace(0.,1.,100):
            _l = int(l * 10000)
            # _s = [random_sample(s, tf.constant(_l, tf.int32)) for s in samples]
            _s = [s[:tf.constant(_l, tf.int32),...] for s in samples]
            rhats.append(tfp.mcmc.potential_scale_reduction(_s))

        # avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)), name='avg_acc_ratio')
        # posterior_log_prob = tf.reduce_mean((target_log_prob), name='marginal_log_likelihood')

        def reduce_median(X, axis=0, keepdims=False):
            return tfp.stats.percentile(X,50, axis=axis,keep_dims=keepdims)

        flat_samples = [flatten_batch_dims(s, -1) for s in samples]
        sample_stddevs = [tf.math.sqrt(tf.reduce_mean(tfp.stats.variance(s))) for s in flat_samples]

        transformed_y_sigma = tf.exp(flat_samples[0])
        transformed_amp = tf.exp(flat_samples[1])
        transformed_dtec = transformed_amp * tf.matmul(flat_samples[2], L, transpose_b=True)

        transformed_y_sigma_init = tf.exp(init_state[0])
        transformed_amp_init = tf.exp(init_state[1])
        transformed_dtec_init = transformed_amp_init * tf.matmul(init_state[2], L, transpose_b=True)

        diff_dtec = reduce_median(transformed_dtec, axis=0) - reduce_median(transformed_dtec_init, axis=0)
        diff_amp = reduce_median(transformed_amp, axis=0) - reduce_median(transformed_amp_init, axis=0)
        diff_y_sigma = reduce_median(transformed_y_sigma, axis=0) - reduce_median(transformed_y_sigma_init, axis=0)

        Lp = tfp.stats.covariance(flat_samples[2])
        mp = tf.reduce_mean(flat_samples[2],axis=0)

        phase_post = invfreqs * transformed_dtec[:, :, None]



        Yimag_post = reduce_median(tf.sin(phase_post), axis=0)
        Yreal_post = reduce_median(tf.cos(phase_post), axis=0)
        phase_post = tf.atan2(Yimag_post, Yreal_post)

        transformed_dtec = reduce_median(transformed_dtec, axis=0)



        # saem_opt = tf.train.AdamOptimizer(1e-3).minimize(-posterior_log_prob,var_list=[a,l])

        # tf_session.run(tf.global_variables_initializer())
        # for i in range(100):
        times = X[:, 0]

        # profiler = tf.profiler.Profiler(tf_session.graph)
        # run_meta = tf.compat.v1.RunMetadata()

        from timeit import default_timer

        t0 = default_timer()

        out = tf_session.run({
            'loss':loss_array,
            'init_dtec': tf.reshape(reduce_median(transformed_dtec_init, axis=0), (N, N, Na)),
            'init_amp': reduce_median(transformed_amp_init, axis=0),
            'init_y_sigma': reduce_median(transformed_y_sigma_init, axis=0),
            'diff_dtec':tf.reshape(diff_dtec, (N, N, Na)),
            'diff_amp':diff_amp,
            'diff_y_sigma':diff_y_sigma,
            'mp': mp,
            'Lp': Lp,
            'dtec': tf.reshape(transformed_dtec, (N, N, Na)),
            'y_sigma': transformed_y_sigma,
            'amp': transformed_amp,
            # 'avg_acceptance_ratio': avg_acceptance_ratio,
            # 'posterior_log_prob': posterior_log_prob,
            'Ftrue': tf.reshape(Ftrue, (N, N, Na)),
            'Yimag_true': tf.reshape(Yimag_true, (N, N, Na, Nf)),
            'Yimag_data': tf.reshape(Yimag, (N, N, Na, Nf)),
            'Yimag_post': tf.reshape(Yimag_post, (N, N, Na, Nf)),
            'phase_true': tf.reshape(phase_true, (N, N, Na, Nf)),
            'phase_data': tf.reshape(phase_data, (N, N, Na, Nf)),
            'phase_post': tf.reshape(phase_post, (N, N, Na, Nf)),
            'rhats':rhats,
            'sample_sizes':sample_stddevs
        },
            # options=tf.compat.v1.RunOptions(
            #     trace_level=tf.RunOptions.FULL_TRACE),
            # run_metadata=run_meta
        )

        print("Time: {}".format(default_timer() - t0))

        # profiler.add_step(0, run_meta)
        # # Profile the parameters of your model.
        # profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder
        #                                      .with_file_output('/home/albert/git/bayes_filter/profiler/profiler_output')
        #                                      .with_timeline_output('/home/albert/git/bayes_filter/profiler/timeline')
        #                                      .float_operation()
        #                                      .build()
        #                             )
        # profiler.profile_graph(options=tf.profiler.ProfileOptionBuilder
        #                                      .with_file_output('/home/albert/git/bayes_filter/profiler/profiler_output_g')
        #                                      .with_timeline_output('/home/albert/git/bayes_filter/profiler/timeline_g')
        #                                      .float_operation()
        #                                      .build())
        # return
        # print(out['avg_acceptance_ratio'])
        # print(out['posterior_log_prob'])
        print("Standard devs.", out['sample_sizes'])

        def _flag_data(data):
            shape = data.shape
            data = data.ravel()
            data[screen_slice] = np.nan
            # data = data[inv_sort]
            data = data.reshape(shape)
            return data

        import pylab as plt
        import os
        for i in range(Na):

            output_folder = os.path.abspath(os.path.join(output_folder_, lofar_array[0][i]))

            os.makedirs(output_folder, exist_ok=True)

            np.savez(os.path.join(output_folder, 'rhats.npz'), rhats=out['rhats'])
            # break

            plt.plot(out['loss'])
            plt.savefig(os.path.join(output_folder, 'sgd_loss.png'))
            plt.close('all')

            plt.imshow(out['Lp'])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'Lp.png'))
            plt.close('all')

            plt.plot(out['mp'])
            plt.savefig(os.path.join(output_folder, 'mp.png'))
            plt.close('all')

            plt.imshow(out['diff_dtec'][:, :, i])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'diff_dtec.png'))
            plt.close('all')

            plt.imshow(out['init_dtec'][:, :, i])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'init_dtec.png'))
            plt.close('all')

            plt.imshow(out['dtec'][:,:,i])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'dtec_post.png'))
            plt.close('all')
            plt.imshow(out['Ftrue'][:,:,i])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'dtec_true.png'))
            plt.close('all')
            plt.imshow(out['dtec'][:,:,i] - out['Ftrue'][:,:,i])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'dtec_post_true_res.png'))
            plt.close('all')
            plt.imshow(out['Yimag_post'][:, :,i, 0])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'Yimag_post.png'))
            plt.close('all')
            plt.imshow(out['Yimag_true'][:, :,i, 0])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'Yimag_true.png'))
            plt.close('all')

            data = out['Yimag_data'][:, :,:, 0]
            data = _flag_data(data)
            data = data[:,:,i]
            plt.imshow(data)
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'Yimag_data.png'))
            plt.close('all')

            plt.imshow(out['Yimag_post'][:, :,i, 0] - out['Yimag_true'][:, :,i, 0])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'Yimag_post_true_res.png'))
            plt.close('all')

            data = out['Yimag_post'][:, :,:, 0] - out['Yimag_data'][:, :,:, 0]
            data = _flag_data(data)
            data = data[:, :, i]
            plt.imshow(data)
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'Yimag_post_data_res.png'))
            plt.close('all')

            plt.imshow(out['phase_post'][:, :,i, 0])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'phase_post.png'))
            plt.close('all')
            plt.imshow(out['phase_true'][:, :,i, 0])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'phase_true.png'))
            plt.close('all')

            data = out['phase_data'][:, :,:, 0]
            data = _flag_data(data)
            data = data[:, :, i]
            plt.imshow(data)
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'phase_data.png'))
            plt.close('all')

            plt.imshow(out['phase_post'][:, :,i, 0] - out['phase_true'][:, :,i, 0])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'phase_post_true_res.png'))
            plt.close('all')

            data = out['phase_post'][:, :, :, 0] - out['phase_data'][:, :,:, 0]
            data = _flag_data(data)
            data = data[:, :, i]
            plt.imshow(data)
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'phase_post_data_res.png'))
            plt.close('all')

            plt.hist(out['y_sigma'].flatten(), bins=100, label='y_sigma')
            plt.legend()
            plt.savefig(os.path.join(output_folder, 'y_sigma.png'))
            plt.close('all')
            plt.hist(out['amp'].flatten(), bins=100, label='amp')
            plt.legend()
            plt.savefig(os.path.join(output_folder, 'amp.png'))
            plt.close('all')

            plt.hist(out['diff_y_sigma'].flatten(), bins=100, label='diff_y_sigma')
            plt.legend()
            plt.savefig(os.path.join(output_folder, 'diff_y_sigma.png'))
            plt.close('all')
            plt.hist(out['diff_amp'].flatten(), bins=100, label='diff_amp')
            plt.legend()
            plt.savefig(os.path.join(output_folder, 'diff_amp.png'))
            plt.close('all')


if __name__ == '__main__':
    tf_graph = tf.Graph()
    tf_session = tf.Session(graph=tf_graph)

    arrays = os.path.dirname(sys.modules["bayes_filter"].__file__)
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    lofar_array = load_array_file(lofar_array)

    run_target(tf_session, lofar_array)

