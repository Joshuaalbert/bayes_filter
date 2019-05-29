import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayes_filter import float_type, TEC_CONV
from bayes_filter.misc import safe_cholesky, flatten_batch_dims, make_coord_array


def hmc_matrix_stepsizes_1D():
    num_chains = 2

    # config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # tf_session = tf.Session(graph=tf.Graph(), config=config)

    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

    tf_session = tf.Session(graph=tf.Graph())
    with tf_session.graph.as_default():
        X = tf.cast(tf.linspace(0., 10., 100), float_type)[:, None]

        freqs = tf.cast(tf.linspace(100e6, 160e6, 2), float_type)

        # with jit_scope():

        kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=tf.convert_to_tensor(3., float_type),
                                                                        length_scale=tf.convert_to_tensor(0.5,
                                                                                                          float_type),
                                                                        feature_ndims=1)
        K = kern.matrix(X, X)
        L = safe_cholesky(K)
        Ftrue = tf.matmul(L, tf.random.normal(shape=[tf.shape(K)[0], 1], dtype=float_type))
        invfreqs = TEC_CONV * tf.math.reciprocal(freqs)
        Yimag_true, Yreal_true = tf.sin(Ftrue * invfreqs), tf.cos(Ftrue * invfreqs)
        Yimag = Yimag_true + 0.3 * tf.random_normal(shape=tf.shape(Yimag_true), dtype=float_type)
        Yreal = Yreal_true + 0.3 * tf.random_normal(shape=tf.shape(Yreal_true), dtype=float_type)

        outliers = np.zeros((100, 2))
        outliers[np.random.choice(100, size=10, replace=False), :] = 3.

        Yimag += tf.constant(outliers, float_type)
        # Yreal += tf.constant(outliers, float_type)

        ###
        # sampling

        a = tf.Variable(0., dtype=float_type)
        l = tf.Variable(0., dtype=float_type)

        # with jit_scope():
        kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=3. * tf.exp(a),
                                                                        length_scale=0.5 * tf.exp(l),
                                                                        feature_ndims=1)
        K = kern.matrix(X, X)
        L = safe_cholesky(K)

        def logp(log_y_sigma, f):
            # with jit_scope():

            prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f),
                                                             scale_identity_multiplier=1.).log_prob(f)

            y_sigma = tf.exp(log_y_sigma)
            Yimag_model = tf.sin(tf.matmul(tf.tile(L[None, :, :], (num_chains, 1, 1)), f[:, :, None]) * invfreqs)
            Yreal_model = tf.cos(tf.matmul(tf.tile(L[None, :, :], (num_chains, 1, 1)), f[:, :, None]) * invfreqs)

            likelihood = tfp.distributions.Laplace(loc=Yimag[None, :, :], scale=y_sigma[:, :, None]).log_prob(
                Yimag_model) \
                         + tfp.distributions.Laplace(loc=Yreal[None, :, :], scale=y_sigma[:, :, None]).log_prob(
                Yreal_model)

            logp = tf.reduce_sum(likelihood, axis=[1, 2]) + prior
            return logp

        step_size = [tf.get_variable(
            name='step_size',
            initializer=lambda: tf.constant(0.001, dtype=float_type),
            use_resource=True,
            dtype=float_type,
            trainable=False)]

        ###
        hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=logp,
            num_leapfrog_steps=2,
            step_size=step_size,
            state_gradients_are_stopped=True),
            num_adaptation_steps=1000,
            target_accept_prob=tf.constant(0.6, dtype=float_type),
            adaptation_rate=0.05)

        # Run the chain (with burn-in maybe).
        # last state as initial point (mean of each chain)
        # TODO: add trace once working without it
        # TODO: let noise be a hmc param

        def trace_fn(_, pkr):
            return (pkr.inner_results.log_accept_ratio,
                    pkr.inner_results.accepted_results.step_size,
                    pkr.inner_results.accepted_results.target_log_prob)

        init_state = [tf.constant(np.log(0.1 * np.ones((num_chains, 1))), float_type),
                      tf.zeros(tf.concat([[num_chains], tf.shape(Ftrue)[0:1]], axis=0), dtype=float_type)]

        samples, (log_accept_ratio, stepsizes, target_log_prob) = tfp.mcmc.sample_chain(  # ,
            num_results=2000,
            num_burnin_steps=1000,
            trace_fn=trace_fn,  # trace_step,
            return_final_kernel_results=False,
            current_state=init_state,
            kernel=hmc,
            parallel_iterations=10)

        avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)), name='avg_acc_ratio')
        posterior_log_prob = tf.reduce_mean(tf.exp(target_log_prob), name='marginal_log_likelihood')

        flat_samples = [flatten_batch_dims(s, -1) for s in samples]
        transformed_y_sigma = tf.exp(flat_samples[0])
        transformed_dtec = tf.matmul(L, flat_samples[1], transpose_b=True)

        Yimag_post = tf.reduce_mean(tf.sin(invfreqs * transformed_dtec[:, :, None]), axis=1)
        Yreal_post = tf.reduce_mean(tf.cos(invfreqs * transformed_dtec[:, :, None]), axis=1)
        transformed_dtec = tf.reduce_mean(transformed_dtec, axis=1)

        # saem_opt = tf.train.AdamOptimizer(1e-3).minimize(-posterior_log_prob,var_list=[a,l])

        tf_session.run(tf.global_variables_initializer())
        # for i in range(100):
        times = tf_session.run(X[:, 0])
        out = tf_session.run({
            'dtec': transformed_dtec, 'y_sigma': transformed_y_sigma,
            'avg_acceptance_ratio': avg_acceptance_ratio, 'posterior_log_prob': posterior_log_prob,
            'Ftrue': Ftrue[:, 0],
            'Yimag_true': Yimag_true,
            'Yimag': Yimag,
            'Yimag_post': Yimag_post})
        print(out['y_sigma'])
        import pylab as plt
        import os
        output_folder = os.path.abspath('hmc_debug_output_1D')
        os.makedirs(output_folder, exist_ok=True)
        plt.plot(times, out['dtec'], label='mean')
        plt.plot(times, out['Ftrue'], label='true')
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'dtec_vs_true.png'))
        plt.close('all')
        plt.plot(times, out['dtec'] - out['Ftrue'])
        plt.savefig(os.path.join(output_folder, 'residuals.png'))
        plt.close('all')
        plt.plot(times, out['Yimag_true'], label='true')
        plt.plot(times, out['Yimag_post'], label='post')
        plt.plot(times, out['Yimag'], label='data')
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'Yimag.png'))
        plt.close('all')
        plt.hist(out['y_sigma'].flatten(), bins=100, label='y_sigma')
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'y_sigma.png'))


def hmc_matrix_stepsizes_2D():
    num_chains = 2

    # config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # tf_session = tf.Session(graph=tf.Graph(), config=config)

    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

    tf_session = tf.Session(graph=tf.Graph())
    with tf_session.graph.as_default():
        N = 15
        Nf = 5

        x = tf.cast(tf.linspace(0.,10.,N), float_type)[:,None]
        x = tf_session.run(x)
        X = tf.constant(make_coord_array(x,x,flat=True))


        freqs = tf.cast(tf.linspace(100e6, 160e6, Nf),float_type)

        # with jit_scope():

        kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=tf.convert_to_tensor(25.,float_type),
                                                               length_scale=tf.convert_to_tensor(1.,float_type),
                                                               feature_ndims=1)
        K = kern.matrix(X, X)
        L = safe_cholesky(K)
        Ftrue = tf.matmul(L, tf.random.normal(shape=[tf.shape(K)[0], 1],dtype=float_type))
        invfreqs = TEC_CONV*tf.math.reciprocal(freqs)

        phase_true = Ftrue*invfreqs
        Yimag_true, Yreal_true = tf.sin(phase_true), tf.cos(phase_true)
        Yimag = Yimag_true + 0.07*tf.random_normal(shape=tf.shape(Yimag_true),dtype=float_type)
        Yreal = Yreal_true + 0.07*tf.random_normal(shape=tf.shape(Yreal_true),dtype=float_type)
        phase_data = tf.atan2(Yimag,Yreal)

        mag_data = tf.sqrt(tf.square(Yimag) + tf.square(Yreal))
        Yimag /= mag_data
        Yreal /= mag_data


        outliers = np.zeros((N*N, Nf))
        outliers[np.random.choice(N*N, size=6, replace=False),:] = 1.

        Yimag += tf.constant(outliers, float_type)
        # Yreal += tf.constant(outliers, float_type)

        ###
        # sampling

        a = tf.Variable(0., dtype=float_type)
        l = tf.Variable(0., dtype=float_type)

        # with jit_scope():
        kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=1. * tf.exp(a),
                                                                        length_scale=1. * tf.exp(l),
                                                                        feature_ndims=1)
        K = kern.matrix(X, X)
        L = safe_cholesky(K)

        L_sample = tf.tile(L[None, :, :], (num_chains, 1, 1))

        def logp(log_y_sigma, log_amp, f):
            # with jit_scope():

            prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f), scale_identity_multiplier=1.).log_prob(f)

            y_sigma = tf.exp(log_y_sigma)
            amp = tf.exp(log_amp)
            L_sample_ = amp[:,:,None] * L_sample

            Yimag_model = tf.sin(tf.matmul(L_sample_, f[:,:,None])*invfreqs)
            Yreal_model = tf.cos(tf.matmul(L_sample_, f[:,:,None]) * invfreqs)

            likelihood = tfp.distributions.Laplace(loc = Yimag[None, :, :], scale=y_sigma[:,:,None]).log_prob(Yimag_model) \
                         + tfp.distributions.Laplace(loc = Yreal[None, :, :], scale=y_sigma[:,:,None]).log_prob(Yreal_model)


            logp = tf.reduce_sum(likelihood, axis=[1,2]) + prior + tfp.distributions.Normal(tf.constant(0.07,dtype=float_type), tf.constant(0.1,dtype=float_type)).log_prob(y_sigma[:,0])
            return logp

        step_size = [tf.constant(1e-1, dtype=float_type)]

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
            num_adaptation_steps=2000,
            target_accept_prob=tf.constant(0.6, dtype=float_type),
            adaptation_rate=0.05)

        # Run the chain (with burn-in maybe).
        # last state as initial point (mean of each chain)
        # TODO: add trace once working without it
        # TODO: let noise be a hmc param

        def trace_fn(_, pkr):
            return (pkr.inner_results.log_accept_ratio,
                    pkr.inner_results.accepted_results.step_size,
                    pkr.inner_results.accepted_results.target_log_prob)


        init_state = [tf.constant(np.log(0.1*np.ones((num_chains, 1))), float_type),
                      tf.constant(np.log(5. * np.ones((num_chains, 1))), float_type),
                      tf.zeros(tf.concat([[num_chains], tf.shape(Ftrue)[0:1]], axis=0), dtype=float_type)]

        samples,(log_accept_ratio, stepsizes, target_log_prob) = tfp.mcmc.sample_chain(#,
            num_results=10000,
            num_burnin_steps=3000,
            trace_fn=trace_fn,  # trace_step,
            return_final_kernel_results=False,
            current_state=init_state,
            kernel=hmc,
            parallel_iterations=10)


        avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)), name='avg_acc_ratio')
        posterior_log_prob = tf.reduce_mean((target_log_prob), name='marginal_log_likelihood')

        flat_samples = [flatten_batch_dims(s,-1) for s in samples]
        transformed_y_sigma = tf.exp(flat_samples[0])
        transformed_amp = tf.exp(flat_samples[1])
        transformed_dtec = transformed_amp[:,0]*tf.matmul(L, flat_samples[2],transpose_b=True)

        phase_post = invfreqs*transformed_dtec[:,:,None]

        Yimag_post = tf.reduce_mean(tf.sin(phase_post),axis=1)
        Yreal_post = tf.reduce_mean(tf.cos(phase_post),axis=1)
        phase_post = tf.atan2(Yimag_post, Yreal_post)

        transformed_dtec = tf.reduce_mean(transformed_dtec,axis=1)


        # saem_opt = tf.train.AdamOptimizer(1e-3).minimize(-posterior_log_prob,var_list=[a,l])

        tf_session.run(tf.global_variables_initializer())
        # for i in range(100):
        times = x[:,0]
        out = tf_session.run({
            'dtec':tf.reshape(transformed_dtec, (N, N)),
            'y_sigma':transformed_y_sigma,
            'amp':transformed_amp,
            'avg_acceptance_ratio':avg_acceptance_ratio,
            'posterior_log_prob':posterior_log_prob,
            'Ftrue':tf.reshape(Ftrue[:,0],(N,N)),
            'Yimag_true':tf.reshape(Yimag_true,(N, N, Nf)),
            'Yimag_data':tf.reshape(Yimag,(N,N,Nf)),
            'Yimag_post':tf.reshape(Yimag_post,(N,N,Nf)),
            'phase_true': tf.reshape(phase_true, (N, N, Nf)),
            'phase_data': tf.reshape(phase_data, (N, N, Nf)),
            'phase_post': tf.reshape(phase_post, (N, N, Nf))
        })
        print(out['avg_acceptance_ratio'])
        print(out['posterior_log_prob'])
        import pylab as plt
        import os
        output_folder = os.path.abspath('hmc_debug_output_2D')
        os.makedirs(output_folder,exist_ok=True)
        plt.imshow(out['dtec'])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder,'dtec_post.png'))
        plt.close('all')
        plt.imshow(out['Ftrue'])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'dtec_true.png'))
        plt.close('all')
        plt.imshow(out['dtec']-out['Ftrue'])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'residuals.png'))
        plt.close('all')
        plt.imshow(out['Yimag_post'][:,:,0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'Yimag_post.png'))
        plt.close('all')
        plt.imshow(out['Yimag_true'][:, :, 0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'Yimag_true.png'))
        plt.close('all')
        plt.imshow(out['Yimag_data'][:, :, 0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'Yimag_data.png'))
        plt.close('all')
        plt.imshow(out['Yimag_post'][:, :, 0] - out['Yimag_true'][:, :, 0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'Yimag_post_true_res.png'))
        plt.close('all')
        plt.imshow(out['Yimag_post'][:, :, 0] - out['Yimag_data'][:, :, 0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'Yimag_post_data_res.png'))
        plt.close('all')

        plt.imshow(out['phase_post'][:, :, 0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'phase_post.png'))
        plt.close('all')
        plt.imshow(out['phase_true'][:, :, 0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'phase_true.png'))
        plt.close('all')
        plt.imshow(out['phase_data'][:, :, 0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'phase_data.png'))
        plt.close('all')
        plt.imshow(out['phase_post'][:, :, 0] - out['phase_true'][:, :, 0])
        plt.colorbar()
        plt.savefig(os.path.join(output_folder, 'phase_post_true_res.png'))
        plt.close('all')
        plt.imshow(out['phase_post'][:, :, 0] - out['phase_data'][:, :, 0])
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


if __name__ == '__main__':
    hmc_matrix_stepsizes_2D()