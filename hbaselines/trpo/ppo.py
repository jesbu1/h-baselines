import time
from collections import deque
import os
import csv

import gym
from gym.spaces import Discrete
import tensorflow as tf
import numpy as np
import random

from stable_baselines.common import Dataset, explained_variance, fmt_row, \
    zipsame
from stable_baselines import logger
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_moments import mpi_moments
from stable_baselines.trpo_mpi.utils import add_vtarg_and_adv
from hbaselines.trpo.algorithm import RLAlgorithm
from hbaselines.common.utils import ensure_dir


class PPO(RLAlgorithm):
    """Proximal Policy Optimization.

    See: https://arxiv.org/abs/1707.06347

    Parameters
    ----------
    env : gym.Env or str
        The environment to learn from (if registered in Gym, can be str)
    policy : TODO
        The policy model to use
    timesteps_per_actorbatch : int
        timesteps per actor per update
    clip_param : float
        clipping parameter epsilon
    entcoeff : float
        the entropy loss weight
    optim_epochs : float
        the optimizer's number of epochs
    optim_stepsize : float
        the optimizer's stepsize
    optim_batchsize : int
        the optimizer's the batch size
    gamma : float
        discount factor
    lam : float
        advantage estimation
    adam_epsilon : float
        the epsilon value for the adam optimizer
    schedule : str
        The type of scheduler for the learning rate update ('linear',
        'constant', 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    _init_setup_model : bool
        Whether or not to build the network at the creation of the instance
    policy_kwargs : dict
        additional arguments to be passed to the policy on creation
    """

    def __init__(self,
                 policy,
                 env,
                 gamma=0.99,
                 timesteps_per_actorbatch=256,
                 clip_param=0.2,
                 entcoeff=0.01,
                 optim_epochs=4,
                 optim_stepsize=1e-3,
                 optim_batchsize=64,
                 lam=0.95,
                 adam_epsilon=1e-5,
                 schedule='linear',
                 verbose=0,
                 _init_setup_model=True,
                 policy_kwargs=None):
        """Initialize the algorithm.

        Parameters
        ----------
        env : gym.Env or str
            The environment to learn from (if registered in Gym, can be str)
        policy : TODO
            The policy model to use
        timesteps_per_actorbatch : int
            timesteps per actor per update
        clip_param : float
            clipping parameter epsilon
        entcoeff : float
            the entropy loss weight
        optim_epochs : float
            the optimizer's number of epochs
        optim_stepsize : float
            the optimizer's stepsize
        optim_batchsize : int
            the optimizer's the batch size
        gamma : float
            discount factor
        lam : float
            advantage estimation
        adam_epsilon : float
            the epsilon value for the adam optimizer
        schedule : str
            The type of scheduler for the learning rate update ('linear',
            'constant', 'double_linear_con', 'middle_drop' or
            'double_middle_drop')
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        _init_setup_model : bool
            Whether or not to build the network at the creation of the instance
        policy_kwargs : dict
            additional arguments to be passed to the policy on creation
        """
        super(PPO, self).__init__(policy, env, verbose, policy_kwargs)

        self.gamma = gamma
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.lam = lam
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.lossandgrad = None
        self.adam = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.params = None
        self.step = None
        self.proba_step = None
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

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_util.single_threaded_session(graph=self.graph)

            # Construct network for new policy
            self.policy_pi = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                reuse=False,
                **self.policy_kwargs
            )

            # Network for old policy
            with tf.variable_scope("oldpi", reuse=False):
                old_pi = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    reuse=False,
                    **self.policy_kwargs
                )

            with tf.variable_scope("loss", reuse=False):
                # Target advantage function (if applicable)
                atarg = tf.placeholder(dtype=tf.float32, shape=[None])

                # Empirical return
                ret = tf.placeholder(dtype=tf.float32, shape=[None])

                # learning rate multiplier, updated with schedule
                lrmult = tf.placeholder(
                    name='lrmult', dtype=tf.float32, shape=[])

                # Annealed clipping parameter epsilon
                clip_param = self.clip_param * lrmult

                obs_ph = self.policy_pi.obs_ph
                action_ph = tf.placeholder(
                    tf.float32,
                    shape=(None, self.action_space.shape[0]))

                kloldnew = old_pi.proba_distribution.kl(
                    self.policy_pi.proba_distribution)
                ent = self.policy_pi.proba_distribution.entropy()
                meankl = tf.reduce_mean(kloldnew)
                meanent = tf.reduce_mean(ent)
                pol_entpen = (-self.entcoeff) * meanent

                # pnew / pold
                ratio = tf.exp(
                    self.policy_pi.proba_distribution.logp(action_ph) -
                    old_pi.proba_distribution.logp(action_ph))

                # surrogate from conservative policy iteration
                surr1 = ratio * atarg
                surr2 = tf.clip_by_value(
                    ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

                # PPO's pessimistic surrogate (L^CLIP)
                pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                vf_loss = tf.reduce_mean(
                    tf.square(self.policy_pi.value_flat - ret))
                total_loss = pol_surr + pol_entpen + vf_loss
                losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
                self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl",
                                   "ent"]

                tf.summary.scalar('entropy_loss', pol_entpen)
                tf.summary.scalar('policy_gradient_loss', pol_surr)
                tf.summary.scalar('value_function_loss', vf_loss)
                tf.summary.scalar('approximate_kullback-leibler', meankl)
                tf.summary.scalar('clip_factor', clip_param)
                tf.summary.scalar('loss', total_loss)

                self.params = tf_util.get_trainable_vars("model")

                self.assign_old_eq_new = tf_util.function(
                    [], [],
                    updates=[tf.assign(oldv, newv) for (oldv, newv) in
                             zipsame(tf_util.get_globals_vars("oldpi"),
                                     tf_util.get_globals_vars("model"))])

            with tf.variable_scope("Adam_mpi", reuse=False):
                self.adam = MpiAdam(self.params, epsilon=self.adam_epsilon,
                                    sess=self.sess)

            with tf.variable_scope("input_info", reuse=False):
                tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
                tf.summary.scalar('learning_rate',
                                  tf.reduce_mean(self.optim_stepsize))
                tf.summary.scalar('advantage', tf.reduce_mean(atarg))
                tf.summary.scalar('clip_range',
                                  tf.reduce_mean(self.clip_param))

            self.step = self.policy_pi.step
            self.proba_step = self.policy_pi.proba_step

            tf_util.initialize(sess=self.sess)

            self.summary = tf.summary.merge_all()

            self.lossandgrad = tf_util.function(
                [obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                [self.summary, tf_util.flatgrad(total_loss, self.params)]
                + losses)
            self.compute_losses = tf_util.function(
                [obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                losses)

    def learn(self, total_timesteps, log_dir, seed=None):
        """See parent class."""
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
            self.adam.sync()

            # Prepare for rollouts
            seg_gen = self._collect_samples(self.policy_pi,
                                            self.timesteps_per_actorbatch)

            while self.timesteps_so_far < total_timesteps:
                logger.log("\n********** Iteration %i ************"
                           % self.iters_so_far)

                # Collect samples.
                with self.timed("Sampling"):
                    seg = seg_gen.__next__()

                # Perform the training procedure.
                mean_losses, vpredbefore, tdlamret = self._train(
                    seg, writer, total_timesteps)

                # Log the training statistics.
                self._log_training(t_start, train_filepath, seg, mean_losses,
                                   vpredbefore, tdlamret)

        return self

    def _train(self, seg, writer, total_timesteps):
        """TODO

        :param seg:
        :param writer:
        :param total_timesteps:  FIXME: remove?
        :return:
        """
        if self.schedule == 'constant':
            cur_lrmult = 1.0
        elif self.schedule == 'linear':
            cur_lrmult = max(
                1.0 - float(self.timesteps_so_far) / total_timesteps,
                0)
        else:
            raise NotImplementedError

        add_vtarg_and_adv(seg, self.gamma, self.lam)

        observations, actions = seg["observations"], seg["actions"]
        atarg, tdlamret = seg["adv"], seg["tdlamret"]

        # predicted value function before update
        vpredbefore = seg["vpred"]

        # standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / atarg.std()
        dataset = Dataset(dict(ob=observations, ac=actions,
                               atarg=atarg, vtarg=tdlamret),
                          shuffle=True)
        optim_batchsize = self.optim_batchsize or observations.shape[0]

        # set old parameter values to new parameter values
        self.assign_old_eq_new(sess=self.sess)
        logger.log("Optimizing...")
        logger.log(fmt_row(13, self.loss_names))

        # Here we do a bunch of optimization epochs over the data.
        for k in range(int(self.optim_epochs)):
            # list of tuples, each of which gives the loss for a minibatch
            losses = []
            for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                steps = (self.num_timesteps +
                         k * optim_batchsize +
                         int(i * (optim_batchsize / len(dataset.data_map))))
                if writer is not None:
                    # run loss backprop with summary, but once every 10
                    # runs save the metadata (memory, compute time, ...)
                    summary, grad, *newlosses = self.lossandgrad(
                        batch["ob"], batch["ob"], batch["ac"],
                        batch["atarg"], batch["vtarg"],
                        cur_lrmult, sess=self.sess)
                    writer.add_summary(summary, steps)
                else:
                    _, grad, *newlosses = self.lossandgrad(
                        batch["ob"], batch["ob"], batch["ac"],
                        batch["atarg"], batch["vtarg"], cur_lrmult,
                        sess=self.sess)

                self.adam.update(grad, self.optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in dataset.iterate_once(optim_batchsize):
            newlosses = self.compute_losses(
                batch["ob"], batch["ob"], batch["ac"], batch["atarg"],
                batch["vtarg"], cur_lrmult, sess=self.sess)
            losses.append(newlosses)
        mean_losses, _, _ = mpi_moments(losses, axis=0)

        return mean_losses, vpredbefore, tdlamret

    def _log_training(self,
                      t_start,
                      file_path,
                      seg,
                      mean_losses,
                      vpredbefore,
                      tdlamret):
        """TODO

        :param t_start:
        :param file_path:
        :param seg:
        :param mean_losses:
        :param vpredbefore:
        :param tdlamret:
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

        logger.log(fmt_row(13, mean_losses))
        for (loss_val, name) in zipsame(mean_losses, self.loss_names):
            logger.record_tabular("loss_" + name, loss_val)
        logger.record_tabular("ev_tdlam_before",
                              explained_variance(vpredbefore, tdlamret))

        if len(self.len_buffer) > 0:
            logger.record_tabular("EpLenMean", np.mean(self.len_buffer))
            logger.record_tabular("EpRewMean", np.mean(self.reward_buffer))
        logger.record_tabular("EpThisIter", len(lens))
        logger.record_tabular("EpisodesSoFar", self.episodes_so_far)
        logger.record_tabular("TimestepsSoFar", self.num_timesteps)
        logger.record_tabular("TimeElapsed", time.time() - t_start)

        if self.verbose >= 1:
            logger.dump_tabular()
