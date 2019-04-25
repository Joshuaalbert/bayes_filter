import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayes_filter import float_type
from bayes_filter.misc import diagonal_jitter


def hmc_matrix_stepsizes():
    tf_session = tf.Session(graph=tf.Graph())
    with tf_session.graph.as_default():

        X = tf.cast(tf.linspace(0.,10.,100), float_type)[:,None]

        kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=tf.convert_to_tensor(1.,float_type),
                                                               length_scale=tf.convert_to_tensor(1.,float_type),
                                                               feature_ndims=1)
        K = kern.matrix(X, X)
        L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))
        Ftrue = 0.01*tf.matmul(L, tf.random_normal(shape=[tf.shape(K)[0], 1],dtype=float_type))
        invfreqs = -8.448e9*tf.reciprocal(tf.cast(tf.linspace(100e6, 160e6, 24),float_type))
        Ytrue = tf.sin(Ftrue*invfreqs)
        Y = Ytrue + 0.3*tf.random_normal(shape=tf.shape(Ytrue),dtype=float_type)

        outliers = np.zeros((100, 24))
        outliers[np.random.choice(100, size=10, replace=False),1:-1] = 3.
        Y += tf.constant(outliers, float_type)
        Ytrue_cos = tf.cos(Ftrue * invfreqs)
        Y_cos = Ytrue_cos + 0.3 * tf.random_normal(shape=tf.shape(Ytrue_cos), dtype=float_type)

        a = tf.Variable(0., dtype=float_type)
        l = tf.Variable(0., dtype=float_type)

        def logp(y_sigma, f):
            # l = tf.get_variable('l', initializer=lambda: tf.constant(0., dtype=float_type))
            # a = tf.get_variable('a', initializer=lambda: tf.constant(0., dtype=float_type))
            kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=2. * tf.exp(a),
                                                                   length_scale=2. * tf.exp(l),
                                                                   feature_ndims=1)
            K = kern.matrix(X, X)
            L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))
            # kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=tf.convert_to_tensor(1.,float_type),
            #                                                             length_scale=tf.convert_to_tensor(1.,float_type),feature_ndims=1)
            # K = kern.matrix(X,X)
            # L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))
            prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros_like(f), scale_identity_multiplier=1.)
            # prior = tfp.distributions.MultivariateNormalTriL(scale_tril=L[None, :, :])
            Y_model = tf.sin(tf.matmul(L[None, :, :],f[:,:,None])*invfreqs)
            Y_model_cos = tf.cos(tf.matmul(L[None, :, :], f[:, :, None]) * invfreqs)

            likelihood = tfp.distributions.Laplace(loc = Y_model, scale=y_sigma)
            likelihood_cos = tfp.distributions.Laplace(loc=Y_model_cos, scale=y_sigma)
            logp = tf.reduce_sum(likelihood.log_prob(Y[None, :, :]), axis=[1,2]) + tf.reduce_sum(likelihood_cos.log_prob(Y_cos[None, :, :]), axis=[1,2]) +  prior.log_prob(f)
            # logp.set_shape(tf.TensorShape([]))
            return logp

        step_size = tf.get_variable(
                        name='step_size',
                        initializer=lambda: tf.constant(0.001, dtype=float_type),
                        use_resource=True,
                        dtype=float_type,
                        trainable=False),


        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=logp,
            num_leapfrog_steps=3,  # tf.random_shuffle(tf.range(3,60,dtype=tf.int64))[0],
            step_size=step_size,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps=None,
                                                                             target_rate=0.75),
            state_gradients_are_stopped=True)
        #                         step_size_update_fn=lambda v, _: v)

        q0 = [tf.constant([0.01],float_type), 0.*Ftrue[None, :, 0]]

        # Run the chain (with burn-in).
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=1000,
            num_burnin_steps=0,
            current_state=q0,
            kernel=hmc,
            parallel_iterations=5)

        avg_acceptance_ratio = tf.reduce_mean(tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)),
                                                         name='avg_acc_ratio')
        posterior_log_prob = tf.reduce_mean(kernel_results.accepted_results.target_log_prob,
                                                       name='marginal_log_likelihood')

        saem_opt = tf.train.AdamOptimizer(1e-3).minimize(-posterior_log_prob,var_list=[a,l])

        tf_session.run(tf.global_variables_initializer())
        for i in range(100):
            _, loss = tf_session.run([saem_opt, posterior_log_prob])
            print(i, loss)





        with tf.control_dependencies([saem_opt]):
            kern = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(amplitude=2. * tf.exp(a),
                                                                            length_scale=2. * tf.exp(l),
                                                                            feature_ndims=1)
            K = kern.matrix(X, X)
            L = tf.cholesky(K + diagonal_jitter(tf.shape(K)[0]))

        samples = [samples[0], tf.einsum("ab,snb->sna", L, samples[1])[...,None], 2.*tf.exp(a), 2.*tf.exp(l)]


        invfreqs,times, logp0, Y, Ytrue, Ftrue, samples,step_sizes, ess, chain_logp = tf_session.run(
            [invfreqs, X[:,0], logp(*q0),Y,Ytrue,Ftrue,samples,
              kernel_results.extra.step_size_assign,
              tfp.mcmc.effective_sample_size(samples[0:2]),
              tf.reduce_mean(kernel_results.accepted_results.target_log_prob,name='marginal_log_likelihood')])
        print(logp0)
        print(chain_logp)
        print(ess)
        import pylab as plt
        import seaborn as sns
        plt.style.use('ggplot')
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

        ax1.plot(times, Ftrue[:, 0])
        ax1.plot(times, samples[1][:,0,:,0].mean(0))
        ax1.fill_between(times, samples[1][:,0,:,0].mean(0) - samples[1][:,0,:,0].std(0), samples[1][:,0,:,0].mean(0) + samples[1][:,0,:,0].std(0), alpha=0.5)
        ax1.legend()
        ax1.set_title("Model space solution vs true")

        Ypost = [np.sin(samples[1][:,0,:,:] *invfreqs).mean(0), np.sin(samples[1][:,0,:,:] *invfreqs).std(0)]
        ax2.plot(times, Ytrue, label='Ytrue')
        ax2.plot(times, Ypost[0], label='posterior')
        [ax2.fill_between(times, Ypost[0][...,i] - Ypost[1][...,i] , Ypost[0][...,i]  + Ypost[1][...,i] , alpha=0.5) for i in range(24)]
        ax2.set_title("Data space solution vs true")
        ax2.legend()

        ax3.plot(times, Y, label='Ytrue')
        ax3.plot(times, Ypost[0], label='posterior')
        [ax3.fill_between(times, Ypost[0][...,i] - Ypost[1][...,i] , Ypost[0][...,i]  + Ypost[1][...,i] , alpha=0.5) for i in range(24)]
        ax3.legend()
        ax3.set_title("Data space solution vs data")

        print(samples[2:4])
        sns.kdeplot(samples[0].flatten(), ax=ax4,shade=True, alpha=0.5)
        # ax4.plot(step_sizes[0], label='y_sigma stepsize')
        # ax4.plot(step_sizes[1], label='dtec stepsize')
        # ax4.set_yscale("log")
        # ax4.set_title("stepsizes")
        ax4.legend()
        plt.show()

if __name__ == '__main__':
    hmc_matrix_stepsizes()