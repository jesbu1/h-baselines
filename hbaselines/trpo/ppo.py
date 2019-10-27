import time
from collections import deque
import os
import csv

import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common import Dataset, explained_variance, fmt_row, \
    zipsame
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_moments import mpi_moments
from stable_baselines.trpo_mpi.utils import add_vtarg_and_adv
from hbaselines.trpo.algorithm import RLAlgorithm


class PPO(RLAlgorithm):
    """Proximal Policy Optimization.

    See: https://arxiv.org/abs/1707.06347

    Parameters
    ----------
    env : gym.Env or str
        The environment to learn from (if registered in Gym, can be str)
    policy : TODO
        The policy model to use
    timesteps_per_batch : int
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
    policy_kwargs : dict
        additional arguments to be passed to the policy on creation
    """

    def __init__(self,
                 policy,
                 env,
                 gamma=0.99,
                 timesteps_per_batch=256,
                 clip_param=0.2,
                 entcoeff=0.01,
                 optim_epochs=4,
                 optim_stepsize=1e-3,
                 optim_batchsize=64,
                 lam=0.95,
                 adam_epsilon=1e-5,
                 schedule='linear',
                 verbose=0,
                 policy_kwargs=None):
        """Initialize the algorithm.

        Parameters
        ----------
        env : gym.Env or str
            The environment to learn from (if registered in Gym, can be str)
        policy : TODO
            The policy model to use
        timesteps_per_batch : int
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
        policy_kwargs : dict
            additional arguments to be passed to the policy on creation
        """
        super(PPO, self).__init__(policy, env, timesteps_per_batch, verbose,
                                  policy_kwargs)

        self.gamma = gamma
        self.timesteps_per_batch = timesteps_per_batch
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.lam = lam
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule

        self.loss_names = None
        self.adam = None
        self.assign_old_eq_new = None
        self.params = None
        self.summary = None

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

        # Perform the algorithm-specific model setup procedure.
        self.setup_model()

        with self.graph.as_default():
            # Create the tensorboard summary.
            self.summary = tf.summary.merge_all()

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.adam.sync()

    def setup_model(self):
        """

        :return:
        """
        with self.graph.as_default():
            # Network for old policy
            with tf.variable_scope("oldpi", reuse=False):
                old_pi = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    reuse=False,
                    **self.policy_kwargs
                )
                self.old_pi = old_pi

            with tf.variable_scope("loss", reuse=False):
                # Target advantage function (if applicable)
                atarg = tf.placeholder(dtype=tf.float32, shape=[None])
                self.atarg = atarg

                # Empirical return
                ret = tf.placeholder(dtype=tf.float32, shape=[None])
                self.ret = ret

                # learning rate multiplier, updated with schedule
                lrmult = tf.placeholder(
                    name='lrmult', dtype=tf.float32, shape=[])
                self.lrmult = lrmult

                # Annealed clipping parameter epsilon
                clip_param = self.clip_param * lrmult

                self.action_ph = tf.placeholder(
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
                    self.policy_pi.proba_distribution.logp(self.action_ph) -
                    old_pi.proba_distribution.logp(self.action_ph))

                # surrogate from conservative policy iteration
                surr1 = ratio * atarg
                surr2 = tf.clip_by_value(
                    ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

                # PPO's pessimistic surrogate (L^CLIP)
                pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                vf_loss = tf.reduce_mean(
                    tf.square(self.policy_pi.value_flat - ret))
                total_loss = pol_surr + pol_entpen + vf_loss
                self.losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
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

            self.flat_grad = tf_util.flatgrad(total_loss, self.params)

    def _train(self, seg, writer, total_timesteps):
        """See parent class.

        TODO: describe
        """
        if self.schedule == 'constant':
            cur_lrmult = 1.0
        elif self.schedule == 'linear':
            cur_lrmult = max(
                1.0 - float(self.timesteps_so_far) / total_timesteps, 0)
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
        print("Optimizing...")
        print(fmt_row(13, self.loss_names))

        # Here we do a bunch of optimization epochs over the data.
        for k in range(int(self.optim_epochs)):
            # list of tuples, each of which gives the loss for a minibatch
            losses = []
            for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                # TODO: remove?
                steps = (self.num_timesteps +
                         k * optim_batchsize +
                         int(i * (optim_batchsize / len(dataset.data_map))))
                # Run loss backprop with summary.
                summary, grad, *newlosses = self.sess.run(
                    [self.summary, self.flat_grad] + self.losses,
                    feed_dict={
                        self.policy_pi.obs_ph: batch["ob"],
                        self.old_pi.obs_ph: batch["ob"],
                        self.action_ph: batch["ac"],
                        self.atarg: batch["atarg"],
                        self.ret: batch["vtarg"],
                        self.lrmult: cur_lrmult,
                    }
                )
                # TODO: out of loop?
                writer.add_summary(summary, steps)

                self.adam.update(grad, self.optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            print(fmt_row(13, np.mean(losses, axis=0)))

        # TODO: remove?
        print("Evaluating losses...")
        losses = []
        for batch in dataset.iterate_once(optim_batchsize):
            newlosses = self.sess.run(
                self.losses,
                feed_dict={
                    self.policy_pi.obs_ph: batch["ob"],
                    self.old_pi.obs_ph: batch["ob"],
                    self.action_ph: batch["ac"],
                    self.atarg: batch["atarg"],
                    self.ret: batch["vtarg"],
                    self.lrmult: cur_lrmult,
                }
            )
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
        """See parent class."""
        lens, rews = seg["ep_lens"], seg["ep_rets"]
        current_it_timesteps = seg["total_timestep"]

        self.len_buffer.extend(lens)
        self.reward_buffer.extend(rews)
        self.episodes_so_far += len(lens)
        self.timesteps_so_far += current_it_timesteps
        self.num_timesteps += current_it_timesteps  # TODO: remove
        self.iters_so_far += 1

        stats = {
            "episode_steps": np.mean(lens),
            "mean_return_history": np.mean(self.reward_buffer),
            "mean_return": np.mean(rews),
            "max_return": max(rews, default=0),
            "min_return": min(rews, default=0),
            "std_return": np.std(rews),
            "episodes_this_itr": len(lens),
            "episodes_total": self.episodes_so_far,
            "steps": self.num_timesteps,
            "epoch": self.iters_so_far,
            "duration": time.time() - t_start,
            "steps_per_second": self.timesteps_so_far / (time.time()-t_start),
            # TODO: what is this?
            "ev_tdlam_before": explained_variance(vpredbefore, tdlamret)
        }
        for (loss_val, name) in zipsame(mean_losses, self.loss_names):
            stats["loss_" + name] = loss_val

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(stats)

        # Print statistics.
        if self.verbose >= 1:
            print("-" * 47)
            for key in sorted(stats.keys()):
                val = stats[key]
                print("| {:<20} | {:<20g} |".format(key, val))
            print("-" * 47)
            print('')
