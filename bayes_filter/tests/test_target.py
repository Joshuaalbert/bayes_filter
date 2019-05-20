from .common_setup import *
import tensorflow_probability as tfp
import tensorflow as tf
from ..misc import make_example_datapack, safe_cholesky, flatten_batch_dims, make_coord_array, random_sample, sample_laplacian, sqrt_with_finite_grads
from ..coord_transforms import itrs_to_enu_with_references
from ..feeds import init_feed
from ..kernels import DTECIsotropicTimeGeneral
from ..processes import DTECProcess
from ..targets import DTECToGainsSAEM
from .. import float_type, TEC_CONV
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au

def test_custom_gradient(tf_session, lofar_array):
    import os
    Nf = 4
    N = 3
    Nd = N * N
    Na = len(lofar_array[0])
    print(lofar_array[0].shape, lofar_array[1].shape)
    num_chains = 5
    gain_noise = 0.1

    output_folder = os.path.abspath(os.path.join(TEST_FOLDER, 'test_target'))
    os.makedirs(output_folder, exist_ok=True)
    with tf_session.graph.as_default():
        tf.random.set_random_seed(0)
        np.random.seed(0)
        freqs = tf.constant(np.linspace(120e6, 160e6, Nf), dtype=float_type)

        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd * 86400. + tf.cast(tf.linspace(0., 50., 9)[:, None], float_type)
        time_feed = TimeFeed(index_feed, times)

        # ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
        # dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
        # Xd = tf.concat([ra, dec], axis=1)

        array_center = ac.SkyCoord(lofar_array[1][:, 0].mean() * au.m, lofar_array[1][:, 1].mean() * au.m,
                                   lofar_array[1][:, 2].mean() * au.m, frame='itrs')
        altaz = ac.AltAz(location=array_center.earth_location, obstime=obstime_init)
        up = ac.SkyCoord(alt=90. * au.deg, az=0. * au.deg, frame=altaz).transform_to('icrs')

        # directions = np.stack([np.random.normal(up.ra.rad, np.pi / 180. * 2.5, size=[Nd]),
        #                        np.random.normal(up.dec.rad, np.pi / 180. * 2.5, size=[Nd])], axis=1)

        ra = np.linspace(up.ra.rad - np.pi / 180. * 2.5, up.ra.rad + np.pi / 180. * 2.5, N)
        dec = np.linspace(up.dec.rad - np.pi / 180. * 2.5, up.dec.rad + np.pi / 180. * 2.5, N)

        Xd = tf.constant(make_coord_array(ra[:, None], dec[:, None], flat=True), dtype=float_type)

        Xa = tf.constant(lofar_array[1], dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(
                                        itrs_to_enu_with_references(lofar_array[1][0, :], [up.ra.rad, up.dec.rad],
                                                                    lofar_array[1][0, :])))
        init, X = init_feed(coord_feed)
        tf_session.run(init)

        kern = DTECIsotropicTimeGeneral(variance=1.0, lengthscales=10.0,
                                        a=250., b=50., timescale=30., ref_location=[0., 0., 0.],
                                        fed_kernel='RBF', obs_type='DDTEC', kernel_params=dict(resolution=3),
                                        squeeze=True)

        K = kern.K(X)
        L = safe_cholesky(K)

        Ftrue = 3.5 * tf.matmul(L, tf.random.normal(shape=[tf.shape(K)[0], 1], dtype=float_type))
        invfreqs = TEC_CONV * tf.math.reciprocal(freqs)

        phase_true = Ftrue * invfreqs
        Yimag_true, Yreal_true = tf.sin(phase_true), tf.cos(phase_true)

        Yimag = Yimag_true + gain_noise * sample_laplacian(shape=tf.shape(Yimag_true),
                                                           dtype=float_type)  # tf.random.random_normal(shape=tf.shape(Yimag_true), dtype=float_type)
        Yreal = Yreal_true + gain_noise * sample_laplacian(shape=tf.shape(Yreal_true),
                                                           dtype=float_type)  # tf.random_normal(shape=tf.shape(Yreal_true), dtype=float_type)
        phase_data = tf.atan2(Yimag, Yreal)

        # mag_data = tf.sqrt(tf.square(Yimag) + tf.square(Yreal))
        # Yimag /= mag_data
        # Yreal /= mag_data

        # outliers = np.zeros((N * N, Nf))
        # outliers[np.random.choice(N * N, size=num_outliers, replace=False), :] = 1.
        #
        # Yimag += tf.constant(outliers, float_type)
        # Yreal += tf.constant(outliers, float_type)

        ###
        # sampling

        L_sample = L  # tf.tile(L[None, :, :], (num_chains, 1, 1))

        def logp(log_y_sigma, log_amp, f):
            # with jit_scope():

            prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f),
                                                             scale_identity_multiplier=1.).log_prob(f)

            y_sigma = tf.exp(log_y_sigma)
            amp = tf.exp(log_amp)
            # L_sample_ = amp[:, :, None] * L_sample

            # L_ij f_sj -> f_sj L_ji
            dtec = amp * tf.matmul(f, L_sample, transpose_b=True)

            Yimag_model = tf.sin(dtec[:, :, None] * invfreqs)
            Yreal_model = tf.cos(dtec[:, :, None] * invfreqs)

            likelihood = tfp.distributions.Laplace(loc=Yimag[None, :, :], scale=y_sigma[:, :, None]).log_prob(
                Yimag_model) \
                         + tfp.distributions.Laplace(loc=Yreal[None, :, :], scale=y_sigma[:, :, None]).log_prob(
                Yreal_model)

            logp = tf.reduce_sum(likelihood, axis=[1, 2]) + prior + tfp.distributions.Normal(
                tf.constant(0.07, dtype=float_type), tf.constant(0.1, dtype=float_type)).log_prob(y_sigma[:, 0])
            return logp

        @tf.custom_gradient
        def logp_custom(log_y_sigma, log_amp, f):

            y_sigma = tf.exp(log_y_sigma)
            amp = tf.exp(log_amp)
            # L_ij f_sj -> f_sj L_ji
            dtec_A = tf.matmul(f, L_sample, transpose_b=True)
            dtec = amp * dtec_A

            N = tf.cast(tf.shape(f)[-1], dtype=tf.float64)
            Nf = tf.cast(tf.shape(invfreqs)[0], dtype=tf.float64)

            two = tf.constant(2., dtype=float_type)
            half = tf.math.reciprocal(two)

            phase_model = dtec[:,:,None] * invfreqs

            dphase = phase_model - phase_data

            #letting g = 1

            sin_term_half = two*tf.math.sin(half* dphase)
            sin_term_half_abs = tf.math.abs(sin_term_half)
            #S,N,Nf
            sin_over_sin_half = tf.math.sin(dphase)/sin_term_half_abs
            sin_over_sin_half_fix = tf.where(tf.less(tf.abs(dphase), tf.constant(1e-14, dtype=tf.float64)), tf.ones_like(dphase), sin_over_sin_half)
            #S, N
            sin_over_sin_half_fix_sum_freq = tf.reduce_sum(sin_over_sin_half_fix*invfreqs/y_sigma[:,:,None], axis=2)
            #S
            sin_term_half_abs_sum = tf.reduce_sum(sin_term_half_abs, axis=[1,2])

            #S
            log_prob_data = N*Nf*(-tf.math.log(two) - log_y_sigma[:,0]) - tf.math.reciprocal(y_sigma[:,0]) * sin_term_half_abs_sum

            #S
            chi2 = tf.math.reciprocal(tf.square(amp[:, 0])) * tf.reduce_sum(tf.square(f), axis=-1)

            #S
            log_prob_prior = -N*log_amp[:,0] - half*chi2

            logp = log_prob_data + log_prob_prior

            def grad(d_logp):

                #S
                d_y_sigma = (-N*Nf + tf.math.reciprocal(y_sigma[:,0]) * sin_term_half_abs_sum)*d_logp
                #S, 1
                d_y_sigma = d_y_sigma[:,None]

                # S
                d_amp_A = tf.reduce_sum(-dtec * sin_over_sin_half_fix_sum_freq, axis=1) - N + chi2
                d_amp = d_amp_A * d_logp
                #S,1
                d_amp = d_amp[:, None]

                #S, N
                d_f_A = -sin_over_sin_half_fix_sum_freq
                #sn, nm ->sm
                d_f_C = tf.matmul(d_f_A, L_sample)
                d_f = (d_f_C - tf.math.reciprocal(tf.square(amp))*f)*d_logp[:,None]


                return d_y_sigma, d_amp, d_f

            return logp, grad

        init_state = [tf.constant(np.log(np.random.uniform(0.01, 0.2, (num_chains, 1))), float_type),
                      tf.constant(np.log(np.random.uniform(0.5, 2., (num_chains, 1))), float_type),
                      # tf.zeros([num_chains, 567], dtype=float_type)
                      0.1 * tf.random.normal(tf.concat([[num_chains], tf.shape(Ftrue)[0:1]], axis=0), dtype=float_type)
                      ]

        y1 = logp(*init_state)
        g1 = tf.gradients(y1,init_state)

        y2 = logp_custom(*init_state)
        g2 = tf.gradients(y2, init_state)

        g1, g2 = tf_session.run([g1,g2])

        print(g1,g2)


def test_target(tf_session, lofar_array):
    import os
    Nf = 4
    N = 5
    Nd = N*N
    shuffle = np.random.choice(Nd*62, Nd*62, replace=False)
    data_slice, screen_slice = shuffle[:Nd*31], shuffle[Nd*31:]

    Na = len(lofar_array[0])
    print(lofar_array[0].shape, lofar_array[1].shape)
    num_chains = 10
    gain_noise = 0.1

    output_folder = os.path.abspath(os.path.join(TEST_FOLDER, 'test_target'))
    os.makedirs(output_folder, exist_ok=True)
    with tf_session.graph.as_default():
        tf.random.set_random_seed(0)
        np.random.seed(0)
        freqs = tf.constant(np.linspace(120e6, 160e6, Nf), dtype=float_type)

        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd * 86400. + tf.cast(tf.linspace(0., 50., 9)[:, None], float_type)
        time_feed = TimeFeed(index_feed, times)

        # ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
        # dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(Nd, 1))
        # Xd = tf.concat([ra, dec], axis=1)

        array_center = ac.SkyCoord(lofar_array[1][:,0].mean()*au.m,lofar_array[1][:,1].mean()*au.m,lofar_array[1][:,2].mean()*au.m, frame='itrs')
        altaz = ac.AltAz(location=array_center.earth_location, obstime=obstime_init)
        up = ac.SkyCoord(alt=90. * au.deg, az=0. * au.deg, frame=altaz).transform_to('icrs')

        # directions = np.stack([np.random.normal(up.ra.rad, np.pi / 180. * 2.5, size=[Nd]),
        #                        np.random.normal(up.dec.rad, np.pi / 180. * 2.5, size=[Nd])], axis=1)

        ra = np.linspace(up.ra.rad- np.pi / 180. * 2.5,up.ra.rad + np.pi / 180. * 2.5,N)
        dec = np.linspace(up.dec.rad- np.pi / 180. * 2.5,up.dec.rad + np.pi / 180. * 2.5,N)

        Xd = tf.constant(make_coord_array(ra[:,None], dec[:,None], flat=True), dtype=float_type)

        Xa = tf.constant(lofar_array[1], dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(
                                        itrs_to_enu_with_references(lofar_array[1][0, :], [up.ra.rad, up.dec.rad],
                                                                    lofar_array[1][0, :])))
        init, X = init_feed(coord_feed)
        tf_session.run(init)

        kern = DTECIsotropicTimeGeneral(variance=1.0, lengthscales=10.0,
                                        a=250., b=50., timescale=30., ref_location=[0., 0., 0.],
                                        fed_kernel='RBF', obs_type='DDTEC', kernel_params=dict(resolution=3),
                                        squeeze=True)

        K = kern.K(X)
        L = safe_cholesky(K)

        Ftrue = 3.5*tf.matmul(L, tf.random.normal(shape=[tf.shape(K)[0], 1], dtype=float_type))
        invfreqs = TEC_CONV * tf.math.reciprocal(freqs)

        phase_true = Ftrue * invfreqs
        Yimag_true, Yreal_true = tf.sin(phase_true), tf.cos(phase_true)

        Yimag = Yimag_true + gain_noise * sample_laplacian(shape=tf.shape(Yimag_true), dtype=float_type)#tf.random.random_normal(shape=tf.shape(Yimag_true), dtype=float_type)
        Yreal = Yreal_true + gain_noise * sample_laplacian(shape=tf.shape(Yreal_true), dtype=float_type)#tf.random_normal(shape=tf.shape(Yreal_true), dtype=float_type)

        phase_data = tf.atan2(Yimag, Yreal)

        # mag_data = tf.sqrt(tf.square(Yimag) + tf.square(Yreal))
        # Yimag /= mag_data
        # Yreal /= mag_data

        # outliers = np.zeros((N * N, Nf))
        # outliers[np.random.choice(N * N, size=num_outliers, replace=False), :] = 1.
        #
        # Yimag += tf.constant(outliers, float_type)
        # Yreal += tf.constant(outliers, float_type)

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

            f_data = f
            prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f_data),
                                                             scale_identity_multiplier=1.).log_prob(f_data)

            y_sigma = tf.exp(log_y_sigma)
            amp = tf.exp(log_amp)
            # L_sample_ = amp[:, :, None] * L_sample

            # L_ij f_sj -> f_sj L_ji
            dtec = amp * tf.matmul(f, L_sample, transpose_b=True)

            Yimag_model = tf.sin(dtec[:,:,None] * invfreqs)
            Yreal_model = tf.cos(dtec[:,:,None] * invfreqs)

            likelihood = -tf.math.reciprocal(y_sigma[:, :, None])*sqrt_with_finite_grads( tf.math.square(Yimag_sample[None, :, :] - Yimag_model) + tf.math.square(Yreal_sample[None, :, :] - Yreal_model)) - log_y_sigma[:,:,None]

            # likelihood = tfp.distributions.Laplace(loc=Yimag[None, :, :], scale=y_sigma[:, :, None]).log_prob(
            #     Yimag_model) \
            #              + tfp.distributions.Laplace(loc=Yreal[None, :, :], scale=y_sigma[:, :, None]).log_prob(
            #     Yreal_model)

            logp = tf.reduce_sum(likelihood, axis=[1, 2]) + prior
                   # + tfp.distributions.Normal(tf.constant(0.07, dtype=float_type), tf.constant(0.1, dtype=float_type)).log_prob(y_sigma[:, 0])
            return logp

        # @tf.custom_gradient
        # def logp(log_y_sigma, log_amp, f):
        #
        #     y_sigma = tf.exp(log_y_sigma)
        #     amp = tf.exp(log_amp)
        #     # L_ij f_sj -> f_sj L_ji
        #     dtec_A = tf.matmul(f, L_sample, transpose_b=True)
        #     dtec = amp * dtec_A
        #
        #     N = tf.cast(tf.shape(f)[-1], dtype=tf.float64)
        #     Nf = tf.cast(tf.shape(invfreqs)[0], dtype=tf.float64)
        #
        #     two = tf.constant(2., dtype=float_type)
        #     half = tf.math.reciprocal(two)
        #
        #     phase_model = dtec[:,:,None] * invfreqs
        #
        #     dphase = phase_model - phase_data
        #
        #     #letting g = 1
        #
        #     sin_term_half = two*tf.math.sin(half* dphase)
        #     sin_term_half_abs = tf.math.abs(sin_term_half)
        #     #S,N,Nf
        #     sin_over_sin_half = tf.math.sin(dphase)/sin_term_half_abs
        #     sin_over_sin_half_fix = tf.where(tf.less(tf.abs(dphase), tf.constant(1e-14, dtype=tf.float64)), tf.ones_like(dphase), sin_over_sin_half)
        #     #S, N
        #     sin_over_sin_half_fix_sum_freq = tf.reduce_sum(sin_over_sin_half_fix*invfreqs/y_sigma[:,:,None], axis=2)
        #     #S
        #     sin_term_half_abs_sum = tf.reduce_sum(sin_term_half_abs, axis=[1,2])
        #
        #     #S
        #     log_prob_data = N*Nf*(-tf.math.log(two) - log_y_sigma[:,0]) - tf.math.reciprocal(y_sigma[:,0]) * sin_term_half_abs_sum
        #
        #     #S
        #     chi2 = tf.math.reciprocal(tf.square(amp[:, 0])) * tf.reduce_sum(tf.square(f), axis=-1)
        #
        #     #S
        #     log_prob_prior = -N*log_amp[:,0] - half*chi2
        #
        #     logp = log_prob_data + log_prob_prior
        #
        #     def grad(d_logp):
        #
        #         #S
        #         d_y_sigma = (-N*Nf + tf.math.reciprocal(y_sigma[:,0]) * sin_term_half_abs_sum)*d_logp
        #         #S, 1
        #         d_y_sigma = d_y_sigma[:,None]
        #
        #         # S
        #         d_amp_A = tf.reduce_sum(-dtec * sin_over_sin_half_fix_sum_freq, axis=1) - N + chi2
        #         d_amp = d_amp_A * d_logp
        #         #S,1
        #         d_amp = d_amp[:, None]
        #
        #         #S, N
        #         d_f_A = -sin_over_sin_half_fix_sum_freq
        #         #sn, nm ->sm
        #         d_f_C = tf.matmul(d_f_A, L_sample)
        #         d_f = (d_f_C - tf.math.reciprocal(tf.square(amp))*f)*d_logp[:,None]
        #
        #
        #         return d_y_sigma, d_amp, d_f
        #
        #     return logp, grad

        step_size = [tf.constant(0.02, dtype=float_type),tf.constant(0.13, dtype=float_type),tf.constant(1., dtype=float_type)]

        # tf.get_variable(
        #                 name='step_size',
        #                 initializer=lambda: tf.constant(0.001, dtype=float_type),
        #                 use_resource=True,
        #                 dtype=float_type,
        #                 trainable=False)]

        ###
        hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=logp,
            num_leapfrog_steps=3,
            step_size=step_size,
            state_gradients_are_stopped=True),
            num_adaptation_steps=4000,
            target_accept_prob=tf.constant(0.6, dtype=float_type),
            adaptation_rate=0.10)

        # Run the chain (with burn-in maybe).
        # last state as initial point (mean of each chain)

        def trace_fn(_, pkr):
            return (pkr.inner_results.log_accept_ratio,
                    pkr.inner_results.accepted_results.step_size,
                    pkr.inner_results.accepted_results.target_log_prob)

        init_state = [tf.constant(np.log(np.random.uniform(0.01, 0.2, (num_chains, 1))), float_type),
                      tf.constant(np.log(np.random.uniform(0.09, 2., (num_chains, 1))), float_type),
                      # tf.zeros([num_chains, 567], dtype=float_type)
                      0.3*tf.random.normal(tf.concat([[num_chains], tf.shape(Ftrue)[0:1]], axis=0), dtype=float_type)
                      ]

        samples = tfp.mcmc.sample_chain(  # ,
            num_results=10000,
            num_burnin_steps=5000,
            trace_fn=None,#trace_fn,  # trace_step,
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

        flat_samples = [flatten_batch_dims(s, -1) for s in samples]
        sample_stddevs = [tf.math.sqrt(tf.reduce_mean(tfp.stats.variance(s))) for s in flat_samples]

        transformed_y_sigma = tf.exp(flat_samples[0])
        transformed_amp = tf.exp(flat_samples[1])
        transformed_dtec = transformed_amp * tf.matmul(flat_samples[2], L, transpose_b=True)

        Lp = tfp.stats.covariance(flat_samples[2])
        mp = tf.reduce_mean(flat_samples[2],axis=0)

        phase_post = invfreqs * transformed_dtec[:, :, None]

        def reduce_median(X, axis=0, keepdims=False):
            return tfp.stats.percentile(X,50, axis=axis,keep_dims=keepdims)

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
            data = data.reshape(shape)
            return data

        import pylab as plt
        import os
        for i in range(Na):

            output_folder = os.path.abspath(os.path.join(TEST_FOLDER,'test_target','run35', lofar_array[0][i]))

            os.makedirs(output_folder, exist_ok=True)

            np.savez(os.path.join(output_folder, 'rhats.npz'), rhats=out['rhats'])
            # break

            plt.imshow(out['Lp'])
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, 'Lp.png'))
            plt.close('all')

            plt.plot(out['mp'])
            plt.savefig(os.path.join(output_folder, 'mp.png'))
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
