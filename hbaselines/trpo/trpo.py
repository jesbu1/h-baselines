"""Script containing the TRPO algorithm object."""
import gym
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common import zipsame, dataset
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.cg import conjugate_gradient
from hbaselines.trpo.algorithm import RLAlgorithm

try:
    from flow.utils.registry import make_create_env
except (ImportError, ModuleNotFoundError):
    pass


class TRPO(RLAlgorithm):
    """Trust Region Policy Optimization.

    See: https://arxiv.org/abs/1502.05477

    Attributes
    ----------
    gamma : float
        the discount value
    timesteps_per_batch : int
        the number of timesteps to run per batch (epoch)
    max_kl : float
        the Kullback-Leibler loss threshold
    cg_iters : int
        the number of iterations for the conjugate gradient calculation
    lam : float
        GAE factor
    entcoeff : float
        the weight for the entropy loss
    cg_damping : float
        the compute gradient dampening factor
    vf_stepsize : float
        the value function stepsize
    vf_iters : int
        the value function's number iterations for learning
    loss_names : TODO
        TODO
    assign_old_eq_new : TODO
        TODO
    compute_losses : TODO
        TODO
    compute_lossandgrad : TODO
        TODO
    compute_fvp : TODO
        TODO
    compute_vflossandgrad : TODO
        TODO
    vfadam : TODO
        TODO
    get_flat : TODO
        TODO
    set_from_flat : TODO
        TODO
    reward_giver : TODO
        TODO
    params : TODO
        TODO
    summary : TODO
        TODO
    """

    def __init__(self,
                 policy,
                 env,
                 gamma=0.99,
                 timesteps_per_batch=1024,
                 max_kl=0.01,
                 cg_iters=10,
                 lam=0.98,
                 entcoeff=0.0,
                 cg_damping=1e-2,
                 vf_stepsize=3e-4,
                 vf_iters=3,
                 verbose=0,
                 policy_kwargs=None):
        """Instantiate the algorithm.

        Parameters
        ----------
        policy : TODO
            The policy model to use
        env : gym.Env or str
            The environment to learn from (if registered in Gym, can be str)
        gamma : float
            the discount value
        timesteps_per_batch : int
            the number of timesteps to run per batch (epoch)
        max_kl : float
            the Kullback-Leibler loss threshold
        cg_iters : int
            the number of iterations for the conjugate gradient calculation
        lam : float
            GAE factor
        entcoeff : float
            the weight for the entropy loss
        cg_damping : float
            the compute gradient dampening factor
        vf_stepsize : float
            the value function stepsize
        vf_iters : int
            the value function's number iterations for learning
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        policy_kwargs : dict
            additional arguments to be passed to the policy on creation
        """
        super(TRPO, self).__init__(policy, env, timesteps_per_batch, gamma,
                                   lam, verbose, policy_kwargs)

        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff

        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.vfadam = None
        self.get_flat = None
        self.set_from_flat = None
        self.reward_giver = None
        self.params = None
        self.summary = None

        # Perform the algorithm-specific model setup procedure.
        self.setup_model()

        with self.graph.as_default():
            # Create the tensorboard summary.
            self.summary = tf.summary.merge_all()

    def setup_model(self):
        with self.graph.as_default():
            # Network for old policy
            with tf.variable_scope("oldpi", reuse=False):
                old_policy = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    reuse=False,
                    **self.policy_kwargs)

            with tf.variable_scope("loss", reuse=False):
                # Target advantage function (if applicable)
                atarg = tf.placeholder(dtype=tf.float32, shape=[None])
                # Empirical return
                ret = tf.placeholder(dtype=tf.float32, shape=[None])

                observation = self.policy_pi.obs_ph
                action = tf.placeholder(
                    tf.float32,
                    shape=(None, self.action_space.shape[0]))

                kloldnew = old_policy.proba_distribution.kl(
                    self.policy_pi.proba_distribution)
                ent = self.policy_pi.proba_distribution.entropy()
                meankl = tf.reduce_mean(kloldnew)
                meanent = tf.reduce_mean(ent)
                entbonus = self.entcoeff * meanent

                vferr = tf.reduce_mean(
                    tf.square(self.policy_pi.value_flat - ret))

                # advantage * pnew / pold
                ratio = tf.exp(
                    self.policy_pi.proba_distribution.logp(action) -
                    old_policy.proba_distribution.logp(action))
                surrgain = tf.reduce_mean(ratio * atarg)

                optimgain = surrgain + entbonus
                losses = [optimgain, meankl, entbonus, surrgain, meanent]
                self.loss_names = ["optimgain", "meankl", "entloss",
                                   "surrgain", "entropy"]

                dist = meankl

                all_var_list = tf_util.get_trainable_vars("model")
                var_list = [v for v in all_var_list
                            if "/vf" not in v.name
                            and "/q/" not in v.name]
                vf_var_list = [v for v in all_var_list
                               if "/pi" not in v.name
                               and "/logstd" not in v.name]

                self.get_flat = tf_util.GetFlat(var_list, sess=self.sess)
                self.set_from_flat = tf_util.SetFromFlat(var_list,
                                                         sess=self.sess)

                klgrads = tf.gradients(dist, var_list)
                flat_tangent = tf.placeholder(
                    dtype=tf.float32, shape=[None], name="flat_tan")
                shapes = [var.get_shape().as_list() for var in var_list]
                start = 0
                tangents = []
                for shape in shapes:
                    var_size = tf_util.intprod(shape)
                    tangents.append(tf.reshape(
                        flat_tangent[start: start + var_size], shape))
                    start += var_size
                gvp = tf.add_n([tf.reduce_sum(grad * tangent)
                                for (grad, tangent)
                                in zipsame(klgrads, tangents)])
                # Fisher vector products
                fvp = tf_util.flatgrad(gvp, var_list)

                tf.summary.scalar('entropy_loss', meanent)
                tf.summary.scalar('policy_gradient_loss', optimgain)
                tf.summary.scalar('value_function_loss', surrgain)
                tf.summary.scalar('approximate_kullback-leibler', meankl)
                tf.summary.scalar('loss', optimgain + meankl + entbonus
                                  + surrgain + meanent)

                self.assign_old_eq_new = tf_util.function(
                    [], [],
                    updates=[tf.assign(oldv, newv) for (oldv, newv) in
                             zipsame(tf_util.get_globals_vars("oldpi"),
                                     tf_util.get_globals_vars("model"))])
                self.compute_losses = tf_util.function(
                    [observation, old_policy.obs_ph, action, atarg],
                    losses)
                self.compute_fvp = tf_util.function(
                    [flat_tangent, observation, old_policy.obs_ph, action,
                     atarg], fvp)
                self.compute_vflossandgrad = tf_util.function(
                    [observation, old_policy.obs_ph, ret],
                    tf_util.flatgrad(vferr, vf_var_list))

                tf_util.initialize(sess=self.sess)

                th_init = self.get_flat()
                self.set_from_flat(th_init)

            with tf.variable_scope("Adam_mpi", reuse=False):
                self.vfadam = MpiAdam(vf_var_list, sess=self.sess)
                self.vfadam.sync()

            with tf.variable_scope("input_info", reuse=False):
                tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
                tf.summary.scalar('learning_rate',
                                  tf.reduce_mean(self.vf_stepsize))
                tf.summary.scalar('advantage', tf.reduce_mean(atarg))
                tf.summary.scalar('kl_clip_range', tf.reduce_mean(self.max_kl))

            self.params = tf_util.get_trainable_vars("model") \
                + tf_util.get_trainable_vars("oldpi")

            self.compute_lossandgrad = \
                tf_util.function(
                    [observation, old_policy.obs_ph, action, atarg, ret],
                    [self.summary, tf_util.flatgrad(optimgain, var_list)]
                    + losses)

    def _train(self, seg, writer, total_timesteps):
        """See parent class.

        TODO: describe
        """
        def fisher_vector_product(vec):  # TODO: move somewhere else
            return self.compute_fvp(vec, *fvpargs, sess=self.sess) \
                + self.cg_damping * vec

        # ------------------ Update G ------------------
        print("Optimizing Policy...")
        mean_losses = None
        atarg, tdlamret = seg["adv"], seg["tdlamret"]

        # predicted value function before update
        vpredbefore = seg["vpred"]
        # standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / atarg.std()

        args = (seg["observations"], seg["observations"],
                seg["actions"], atarg)

        # Subsampling: see p40-42 of John Schulman thesis
        # http://joschu.net/docs/thesis.pdf
        fvpargs = [arr[::5] for arr in args]

        self.assign_old_eq_new(sess=self.sess)

        with self.timed("computegrad"):
            steps = self.timesteps_so_far + seg["total_timestep"]
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run loss backprop with summary, and save the metadata
            # (memory, compute time, ...)
            if writer is not None:
                summary, grad, *lossbefore = self.compute_lossandgrad(
                    *args,
                    tdlamret,
                    sess=self.sess,
                    options=run_options)
                writer.add_summary(summary, steps)
            else:
                _, grad, *lossbefore = self.compute_lossandgrad(
                    *args,
                    tdlamret,
                    sess=self.sess,
                    options=run_options)

        lossbefore = np.array(lossbefore)
        if np.allclose(grad, 0):
            print("Got zero gradient. not updating")
        else:
            with self.timed("conjugate_gradient"):
                stepdir = conjugate_gradient(
                    fisher_vector_product,
                    grad,
                    cg_iters=self.cg_iters,
                    verbose=self.verbose >= 1)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            # abs(shs) to avoid taking square root of negative
            # values
            lagrange_multiplier = np.sqrt(
                abs(shs) / self.max_kl)
            fullstep = stepdir / lagrange_multiplier
            expectedimprove = grad.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                mean_losses = surr, kl_loss, *_ = np.array(
                    self.compute_losses(*args, sess=self.sess))
                improve = surr - surrbefore
                print("Expected: %.3f Actual: %.3f" % (
                    expectedimprove, improve))
                if not np.isfinite(mean_losses).all():
                    print("Got non-finite value of losses -- bad!")
                elif kl_loss > self.max_kl * 1.5:
                    print("violated KL constraint. shrinking step.")
                elif improve < 0:
                    print("surrogate didn't improve. shrinking step.")
                else:
                    print("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                print("couldn't compute a good step")
                self.set_from_flat(thbefore)

        with self.timed("vf"):
            for _ in range(self.vf_iters):
                # NOTE: for recurrent policies, use shuffle=False?
                for (mbob, mbret) in dataset.iterbatches(
                        (seg["observations"], seg["tdlamret"]),
                        include_final_partial_batch=False,
                        batch_size=128,  # TODO: make tunable parameter
                        shuffle=True):
                    grad = self.compute_vflossandgrad(
                        mbob, mbob, mbret, sess=self.sess)
                    self.vfadam.update(grad, self.vf_stepsize)

        return mean_losses, vpredbefore, tdlamret
