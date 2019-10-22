"""Script containing the TRPO algorithm object."""
import time
from contextlib import contextmanager
from collections import deque
import os
import csv

import gym
from gym.spaces import Discrete, Box
import tensorflow as tf
import numpy as np
import random

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common import explained_variance, zipsame, dataset, \
    colorize
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.a2c.utils import total_episode_reward_logger
from hbaselines.trpo.utils import add_vtarg_and_adv
from hbaselines.common.utils import ensure_dir

try:
    from flow.utils.registry import make_create_env
except (ImportError, ModuleNotFoundError):
    pass


class TRPO(object):
    """Trust Region Policy Optimization.

    See: https://arxiv.org/abs/1502.05477

    Attributes
    ----------
    policy : ActorCriticPolicy or str
        The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    env : gym.Env or str
        The environment to learn from (if registered in Gym, can be str)
    observation_space : TODO
        TODO
    action_space : TODO
        TODO
    gamma : float
        the discount value
    timesteps_per_batch : int
        the number of timesteps to run per batch (horizon)
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
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    _init_setup_model : bool
        Whether or not to build the network at the creation of the instance
    policy_kwargs : dict
        additional arguments to be passed to the policy on creation
    full_tensorboard_log : bool  TODO: remove?
        enable additional logging when using tensorboard. WARNING: this logging
        can take a lot of space quickly
    graph : TODO
        TODO
    sess = None
        TODO
    policy_pi : TODO
        TODO
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
    d_adam : TODO
        TODO
    vfadam : TODO
        TODO
    get_flat : TODO
        TODO
    set_from_flat : TODO
        TODO
    timed : TODO
        TODO
    reward_giver : TODO
        TODO
    params : TODO
        TODO
    summary : TODO
        TODO
    episode_reward : TODO
        TODO
    len_buffer : TODO
        TODO
    reward_buffer : TODO
        TODO
    episodes_so_far : TODO
        TODO
    timesteps_so_far : TODO
        TODO
    iters_so_far : TODO
        TODO
    num_timesteps : TODO
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
                 _init_setup_model=True,
                 policy_kwargs=None,
                 full_tensorboard_log=False):
        """Instantiate the algorithm.

        Parameters
        ----------
        policy : (ActorCriticPolicy or str)
            The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
        env : gym.Env or str
            The environment to learn from (if registered in Gym, can be str)
        gamma : float
            the discount value
        timesteps_per_batch : int
            the number of timesteps to run per batch (horizon)
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
        _init_setup_model : bool
            Whether or not to build the network at the creation of the instance
        policy_kwargs : dict
            additional arguments to be passed to the policy on creation
        full_tensorboard_log : bool
            enable additional logging when using tensorboard. WARNING: this
            logging can take a lot of space quickly
        """
        self.policy = policy
        self.env = self._create_env(env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.timesteps_per_batch = timesteps_per_batch
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff
        self.full_tensorboard_log = full_tensorboard_log
        self.verbose = verbose
        self.policy_kwargs = policy_kwargs or {}

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.d_adam = None
        self.vfadam = None
        self.get_flat = None
        self.set_from_flat = None
        self.timed = None
        self.reward_giver = None
        self.params = None
        self.summary = None

        # total results from the most recent training step.
        self.episode_reward = np.zeros((1,))
        # rolling buffer for episode lengths
        self.len_buffer = deque(maxlen=40)
        # rolling buffer for episode rewards
        self.reward_buffer = deque(maxlen=40)
        # TODO
        self.episodes_so_far = 0
        # TODO
        self.timesteps_so_far = 0
        # TODO
        self.iters_so_far = 0
        # TODO
        self.num_timesteps = 0

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_pi
        action_ph = policy.pdtype.sample_placeholder([None])
        if isinstance(self.action_space, Discrete):
            return policy.obs_ph, action_ph, policy.policy
        return policy.obs_ph, action_ph, policy.deterministic_action

    def setup_model(self):
        np.set_printoptions(precision=3)

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create the tensorflow session.
            self.sess = tf_util.single_threaded_session(graph=self.graph)

            # Construct network for new policy
            self.policy_pi = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                reuse=False,
                **self.policy_kwargs)

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

                self.assign_old_eq_new = \
                    tf_util.function([], [], updates=[
                        tf.assign(oldv, newv) for (oldv, newv) in
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

                @contextmanager
                def timed(msg):
                    if self.verbose >= 1:
                        print(colorize(msg, color='magenta'))
                        start_time = time.time()
                        yield
                        print(colorize("done in {:.3f} seconds".format(
                            (time.time() - start_time)), color='magenta'))
                    else:
                        yield

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

                if self.full_tensorboard_log:
                    tf.summary.histogram('discounted_rewards', ret)
                    tf.summary.histogram('learning_rate', self.vf_stepsize)
                    tf.summary.histogram('advantage', atarg)
                    tf.summary.histogram('kl_clip_range', self.max_kl)
                    if tf_util.is_image(self.observation_space):
                        tf.summary.image('observation', observation)
                    else:
                        tf.summary.histogram('observation', observation)

            self.timed = timed

            self.params = tf_util.get_trainable_vars("model") \
                + tf_util.get_trainable_vars("oldpi")

            self.summary = tf.summary.merge_all()

            self.compute_lossandgrad = \
                tf_util.function(
                    [observation, old_policy.obs_ph, action, atarg, ret],
                    [self.summary, tf_util.flatgrad(optimgain, var_list)]
                    + losses)

    def learn(self, total_timesteps, log_dir, seed=None):
        """FIXME

        :param total_timesteps:
        :param log_dir:
        :param seed:
        :return:
        """
        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)
        ensure_dir(os.path.join(log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        # save_path = os.path.join(log_dir, "tb_log")
        # writer = tf.compat.v1.summary.FileWriter(save_path)
        writer = None

        # file path for training statistics
        train_filepath = os.path.join(log_dir, "train.csv")

        # Set all relevant seeds.
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # prng was removed in latest gym version
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(seed)

        # Compute the start time, for logging purposes
        t_start = time.time()

        with self.sess.as_default():
            seg_gen = self._collect_samples()

            while self.timesteps_so_far < total_timesteps:
                print("********** Iteration %i ************" %
                      self.iters_so_far)

                # Collect samples.
                with self.timed("Sampling"):
                    seg = seg_gen.__next__()

                # Perform the training procedure.
                mean_losses, vpredbefore, tdlamret = self._train(seg, writer)

                # Log the training statistics.
                self._log_training(train_filepath, mean_losses, vpredbefore,
                                   tdlamret, seg, t_start)

        return self

    @staticmethod
    def _create_env(env):
        """Return, and potentially create, the environment.

        Parameters
        ----------
        env : str or gym.Env
            the environment, or the name of a registered environment.

        Returns
        -------
        gym.Env or list of gym.Env
            gym-compatible environment(s)
        """
        if env in ["figureeight0", "figureeight1", "figureeight2",
                   "merge0", "merge1", "merge2",
                   "bottleneck0", "bottleneck1", "bottleneck2",
                   "grid0", "grid1"]:
            # Import the benchmark and fetch its flow_params
            benchmark = __import__("flow.benchmarks.{}".format(env),
                                   fromlist=["flow_params"])
            flow_params = benchmark.flow_params

            # Get the env name and a creator for the environment.
            create_env, _ = make_create_env(flow_params, version=0)

            # Create the environment.
            env = create_env()

        elif isinstance(env, str):
            # This is assuming the environment is registered with OpenAI gym.
            env = gym.make(env)

        # Reset the environment.
        if env is not None:
            env.reset()

        return env

    def _collect_samples(self):
        """Compute target value using TD estimator, and advantage with GAE.

        Returns
        -------
        dict
            generator that returns a dict with the following keys:

            - observations: (np.ndarray) observations
            - rewards: (numpy float) rewards
            TODO: remove
            - true_rewards: (numpy float) if gail is used it is the original
              reward
            - vpred: (numpy float) action logits
            - dones: (numpy bool) dones (is end of episode, used for logging)
            - episode_starts: (numpy bool) True if first timestep of an
              episode, used for GAE
            - actions: (np.ndarray) actions
            - nextvpred: (numpy float) next action logits
            - ep_rets: (float) cumulated current episode reward
            - ep_lens: (int) the length of the current episode
            - ep_true_rets: (float) the real environment reward
        """
        # Initialize state variables
        step = 0
        # not used, just so we have the datatype
        action = self.env.action_space.sample()
        observation = self.env.reset()

        cur_ep_ret = 0  # return in current episode
        current_it_len = 0  # len of current iteration
        current_ep_len = 0  # len of current episode
        cur_ep_true_ret = 0
        ep_true_rets = []
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # Episode lengths

        # Initialize history arrays
        observations = np.array([observation
                                 for _ in range(self.timesteps_per_batch)])
        true_rewards = np.zeros(self.timesteps_per_batch, 'float32')
        rewards = np.zeros(self.timesteps_per_batch, 'float32')
        vpreds = np.zeros(self.timesteps_per_batch, 'float32')
        episode_starts = np.zeros(self.timesteps_per_batch, 'bool')
        dones = np.zeros(self.timesteps_per_batch, 'bool')
        actions = np.array([action for _ in range(self.timesteps_per_batch)])
        episode_start = True  # marks if we're on first timestep of an episode

        while True:
            action, vpred, _ = self.policy_pi.step(np.array([observation]))
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if step > 0 and step % self.timesteps_per_batch == 0:
                yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len
                }
                _, vpred, _ = self.policy_pi.step(np.array([observation]))
                # Be careful!!! if you change the downstream algorithm to
                # aggregate several of these batches, then be sure to do a
                # deepcopy
                ep_rets = []
                ep_true_rets = []
                ep_lens = []
                # Reset current iteration length
                current_it_len = 0
            i = step % self.timesteps_per_batch
            observations[i] = observation
            vpreds[i] = vpred[0]
            actions[i] = action[0]
            episode_starts[i] = episode_start

            clipped_action = action
            # Clip the actions to avoid out of bound error.
            if isinstance(self.env.action_space, Box):
                clipped_action = np.clip(action,
                                         a_min=self.env.action_space.low,
                                         a_max=self.env.action_space.high)

            observation, reward, done, info = self.env.step(clipped_action[0])
            true_reward = reward
            rewards[i] = reward
            true_rewards[i] = true_reward
            dones[i] = done
            episode_start = done

            cur_ep_ret += reward
            cur_ep_true_ret += true_reward
            current_it_len += 1
            current_ep_len += 1
            if done:
                # Retrieve unnormalized reward if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    cur_ep_ret = maybe_ep_info['r']
                    cur_ep_true_ret = maybe_ep_info['r']

                ep_rets.append(cur_ep_ret)
                ep_true_rets.append(cur_ep_true_ret)
                ep_lens.append(current_ep_len)
                cur_ep_ret = 0
                cur_ep_true_ret = 0
                current_ep_len = 0
                observation = self.env.reset()
            step += 1

    def _train(self, seg, writer):
        """

        :param seg:
        :param writer:
        :return:
        """
        def fisher_vector_product(vec):  # TODO: move somewhere else
            return self.compute_fvp(vec, *fvpargs, sess=self.sess) \
                   + self.cg_damping * vec

        # ------------------ Update G ------------------
        print("Optimizing Policy...")
        mean_losses = None
        add_vtarg_and_adv(seg, self.gamma, self.lam)
        atarg, tdlamret = seg["adv"], seg["tdlamret"]

        # predicted value function before update
        vpredbefore = seg["vpred"]
        # standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / atarg.std()

        # true_rew is the reward without discount  TODO: remove
        if writer is not None:
            self.episode_reward = total_episode_reward_logger(
                self.episode_reward,
                seg["true_rewards"].reshape((1, -1)),
                seg["dones"].reshape((1, -1)),
                writer,
                self.num_timesteps)

        args = (seg["observations"], seg["observations"],
                seg["actions"], atarg)

        # Subsampling: see p40-42 of John Schulman thesis
        # http://joschu.net/docs/thesis.pdf
        fvpargs = [arr[::5] for arr in args]

        self.assign_old_eq_new(sess=self.sess)

        with self.timed("computegrad"):
            steps = self.num_timesteps + seg["total_timestep"]
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata() \
                if self.full_tensorboard_log else None
            # run loss backprop with summary, and save the metadata
            # (memory, compute time, ...)
            if writer is not None:
                summary, grad, *lossbefore = \
                    self.compute_lossandgrad(
                        *args,
                        tdlamret,
                        sess=self.sess,
                        options=run_options,
                        run_metadata=run_metadata)
                if self.full_tensorboard_log:
                    writer.add_run_metadata(
                        run_metadata, 'step%d' % steps)
                writer.add_summary(summary, steps)
            else:
                _, grad, *lossbefore = \
                    self.compute_lossandgrad(
                        *args,
                        tdlamret,
                        sess=self.sess,
                        options=run_options,
                        run_metadata=run_metadata)

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
                        batch_size=128,
                        shuffle=True):
                    grad = self.compute_vflossandgrad(
                        mbob, mbob, mbret, sess=self.sess)
                    self.vfadam.update(grad, self.vf_stepsize)

        return mean_losses, vpredbefore, tdlamret

    def _log_training(self,
                      file_path,
                      mean_losses,
                      vpredbefore,
                      tdlamret,
                      seg,
                      t_start):
        """TODO

        :param file_path:
        :param mean_losses:
        :param vpredbefore:
        :param tdlamret:
        :param seg:
        :param t_start:
        :return:
        """
        lens, rews = seg["ep_lens"], seg["ep_rets"]
        current_it_timesteps = seg["total_timestep"]

        self.len_buffer.extend(lens)
        self.reward_buffer.extend(rews)
        self.episodes_so_far += len(lens)
        self.timesteps_so_far += current_it_timesteps
        self.num_timesteps += current_it_timesteps
        self.iters_so_far += 1

        stats = {
            "episode_steps": np.mean(self.len_buffer),
            "mean_return": np.mean(self.reward_buffer),
            "max_return": np.max(self.reward_buffer),
            "min_return": np.min(self.reward_buffer),
            "std_return": np.std(self.reward_buffer),
            "episodes_this_itr": len(lens),
            "episodes_total": self.episodes_so_far,
            "steps": self.num_timesteps,
            "epoch": self.iters_so_far,
            "duration": time.time() - t_start,
            "steps_per_second":
                self.timesteps_so_far / (time.time() - t_start),
            # TODO: what is this?
            "explained_variance": explained_variance(vpredbefore, tdlamret),
        }
        for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
            stats[loss_name] = loss_val

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(stats)

        # Print statistics.
        print("-" * 47)
        for key in sorted(stats.keys()):
            val = stats[key]
            print("| {:<20} | {:<20g} |".format(key, val))
        print("-" * 47)
        print('')

    def save(self, save_path):
        """FIXME

        :param save_path:
        :return:
        """
        pass  # TODO

    def load(self, file_path):
        """FIXME

        :param file_path:
        :return:
        """
        pass  # TODO
