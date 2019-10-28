"""Script containing the TRPO algorithm object."""
import numpy as np
# import json
import csv
import os
import time
from contextlib import contextmanager
from collections import deque
import random
import gym
from gym.spaces import Box
import tensorflow as tf
from stable_baselines.common import colorize, explained_variance
from hbaselines.common.utils import ensure_dir

try:
    from flow.utils.registry import make_create_env
except (ImportError, ModuleNotFoundError):
    pass


DEFAULT_POLICY_KWARGS = dict(
    # The size of the Neural network for the policy
    layers=[64, 64],
    # the activation function to use in the neural network.
    act_fun=tf.nn.tanh,
    # specifies whether to shared the hidden layers between the actor and
    # critic
    shared=True,
    # specifies whether to create two value functions. If not, only one is
    # created.
    duel_vf=False,
)


class RLAlgorithm(object):
    """Base RL algorithm object.

    Attributes
    ----------
    policy : TODO
        The policy model to use
    env : gym.Env or str
        The environment to learn from (if registered in Gym, can be str)
    observation_space : gym.spaces.*
        the observation space of the training environment
    action_space : gym.spaces.*
        the action space of the training environment
    timesteps_per_batch : int
        the number of timesteps to run per batch (epoch)
    gamma : float
        discount factor
    lam : float
        advantage estimation
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    policy_kwargs : dict
        additional arguments to be passed to the policy on creation
    len_buffer : collections.deque
        rolling buffer for episode lengths
    reward_buffer : collections.deque
        rolling buffer for episode rewards
    episodes_so_far : int
        the total number of rollouts performed since training began
    timesteps_so_far : int
        the total number of steps that have been executed since training began
    iters_so_far : int
        the total number of training iterations since training began
    loss_names : list of str
        The names of the losses as they are added to the training statistics.
        This should be filled by the child classes.
    graph : tf.Graph
        the current tensorflow graph
    sess : tf.compat.v1.Session
        the current tensorflow session
    timed : function
        a utility method that is used to compute the time a specific process
        takes to finish.
    """

    def __init__(self,
                 policy,
                 env,
                 timesteps_per_batch,
                 gamma,
                 lam,
                 ignore_dones=False,
                 verbose=0,
                 policy_kwargs=None):
        """Instantiate the algorithm.

        Parameters
        ----------
        policy : (ActorCriticPolicy or str)
            The policy model to use
        env : gym.Env or str
            The environment to learn from (if registered in Gym, can be str)
        timesteps_per_batch : int
            the number of timesteps to run per batch (epoch)
        gamma : float
            discount factor
        lam : float
            advantage estimation
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        policy_kwargs : dict
            additional arguments to be passed to the policy on creation
        """
        self.policy = policy
        self.env = self._create_env(env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.timesteps_per_batch = timesteps_per_batch
        self.gamma = gamma
        self.lam = lam
        self.ignore_dones = ignore_dones
        self.verbose = verbose
        self.policy_kwargs = DEFAULT_POLICY_KWARGS.copy()
        self.policy_kwargs.update(policy_kwargs or {})
        self.duel_vf = self.policy_kwargs["duel_vf"]

        # some variables used during the logging procedure
        self.len_buffer = deque(maxlen=40)
        self.reward_buffer = deque(maxlen=40)
        self.episodes_so_far = 0
        self.timesteps_so_far = 0
        self.iters_so_far = 0
        self.loss_names = []  # this should be filled by the child classes

        # Create the tensorflow graph and session objects.
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session(graph=self.graph)

            # Construct network for new policy.
            self.policy_pi = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                reuse=False,
                **self.policy_kwargs
            )

        # The following method is used to compute the time a specific process
        # takes to finish.
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
        self.timed = timed

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

    def learn(self, total_timesteps, log_dir, seed=None):
        """Perform the training operation.

        TODO: describe

        Parameters
        ----------
        total_timesteps : int
            the total number of samples to train on
        log_dir : str
            the directory where the training and evaluation statistics, as well
            as the tensorboard log, should be stored
        seed : int or None
            the initial seed for training, if None: keep current seed
        """
        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)
        ensure_dir(os.path.join(log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        save_path = os.path.join(log_dir, "tb_log")
        writer = tf.compat.v1.summary.FileWriter(save_path)

        # file path for training statistics
        train_filepath = os.path.join(log_dir, "train.csv")

        # # Save the hypeparameters to a json file.
        # with open(os.path.join(log_dir, "hyperparameters.json"), "w") as f:
        #     json.dump(self.__dict__, f, sort_keys=True, indent=4)

        # Set all relevant seeds.
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Compute the start time, for logging purposes.
        t_start = time.time()

        with self.sess.as_default():
            iters_so_far = 0
            timesteps_so_far = 0
            while timesteps_so_far < total_timesteps:
                print("\n********** Iteration %i ************" % iters_so_far)

                # Collect samples.
                with self.timed("Sampling"):
                    seg = self._collect_samples(self.timesteps_per_batch)
                    self._add_vtarg_and_adv(seg, self.gamma, self.lam,
                                            self.ignore_dones)

                # Perform the training procedure.
                mean_losses, vpredbefore, tdlamret = self._train(
                    seg, writer, total_timesteps)

                # Log the training statistics.
                self._log_training(t_start, train_filepath, seg, mean_losses,
                                   vpredbefore, tdlamret)

                # Increment relevant variables.
                iters_so_far += 1
                timesteps_so_far += seg["total_timestep"]

        return self

    def _collect_samples(self, n_samples):
        """Compute target value using TD estimator, and advantage with GAE.

        Parameters
        ----------
        n_samples : int
            number of samples to collect before returning from the method

        Returns
        -------
        dict
            generator that returns a dict with the following keys:

            * observations: (np.ndarray) observations
            * next_observations: (np.ndarray) next observations
            * rewards: (numpy float) rewards
            * dones: (numpy bool) dones (is end of episode, used for logging)
            * actions: (np.ndarray) actions
            * ep_rets: (float) cumulated current episode reward
            * ep_lens: (int) the length of the current episode
            * total_timestep: (int) total number of stored steps
        """
        # Initialize history arrays
        observations = []
        next_observations = []
        rewards = []
        dones = []
        actions = []
        cur_ep_ret = 0  # return in current episode
        current_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # Episode lengths

        # Initialize state variables
        obs = self.env.reset()

        while len(observations) < n_samples:
            # Compute the next action and the estimated value of the action.
            action = self.policy_pi.compute_action(np.array([obs]))

            # Clip the actions to avoid out of bound error.
            clipped_action = action
            if isinstance(self.env.action_space, Box):
                clipped_action = np.clip(action,
                                         a_min=self.env.action_space.low,
                                         a_max=self.env.action_space.high)

            # Advance the simulation by one step.
            next_obs, reward, done, info = self.env.step(clipped_action[0])

            # Add sample information.
            observations.append(obs)
            next_observations.append(next_obs)
            actions.append(action[0])
            rewards.append(reward)
            dones.append(done)

            # Update relevant variables.
            obs = next_obs
            cur_ep_ret += reward
            current_ep_len += 1

            if done:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(current_ep_len)
                cur_ep_ret = 0
                current_ep_len = 0
                obs = self.env.reset()

        return {
            "observations": np.asarray(observations),
            "next_observations": np.asarray(next_observations),
            "rewards": np.asarray(rewards),
            "dones": np.asarray(dones),
            "actions": np.asarray(actions),
            "ep_rets": ep_rets,
            "ep_lens": ep_lens,
            "total_timestep": len(observations),
        }

    def _add_vtarg_and_adv(self, seg, gamma, lam, ignore_dones=False):
        """Compute target value using TD estimator, and advantage with GAE.

        Parameters
        ----------
        seg : dict
            the current segment of the trajectory (see _collect_samples return
            for more information)
        gamma : float
            Discount factor
        lam : float
            GAE factor
        ignore_dones : bool
            specifies whether to ignore the done mask when computing the delta
            terms
        """
        # Grab relevant variables.
        observations = seg['observations']
        next_observations = seg['next_observations']
        dones = seg['dones']
        rewards = seg["rewards"]

        # Initialize empty advantages.
        seg_len = len(seg["rewards"])
        seg["adv"] = np.empty(seg_len, 'float32')
        seg["vpred"] = np.empty(seg_len, 'float32')

        lastgaelam = 0
        for step in reversed(range(seg_len)):
            # Compute the predictions of the value function.
            vpred = self.policy_pi.value([observations[step]])
            next_vpred = self.policy_pi.value([next_observations[step]])

            # In case of duel value functions, we use the output from the first
            # value function.  # TODO: maybe iteratively swap?
            if self.duel_vf:
                vpred = vpred[0]
                next_vpred = next_vpred[0]

            seg["vpred"][step] = vpred

            # Compute the next step advantage
            notdone = 1 if ignore_dones else 1 - dones[step]
            delta = rewards[step] + gamma * notdone * next_vpred - vpred
            seg["adv"][step] = lastgaelam = \
                delta + gamma * lam * (1 - dones[step]) * lastgaelam

        seg["tdlamret"] = seg["adv"] + seg["vpred"]

    def _train(self, seg, writer, total_timesteps):
        """Perform a training step.

        To be implemented by the inheriting algorithms.

        Parameters
        ----------
        seg : dict
            the current segment of the trajectory (see _collect_samples return
            for more information)
        writer : TODO
            TODO
        total_timesteps : int
            the total number of samples to train on. Used when, for example,
            performing learning rate annealing.

        Returns
        -------
        TODO
            TODO
        TODO
            TODO
        TODO
            TODO
        """
        raise NotImplementedError

    def _log_training(self,
                      t_start,
                      file_path,
                      seg,
                      mean_losses,
                      vpredbefore,
                      tdlamret):
        """Log training statistics.

        Parameters
        ----------
        t_start : float
            the time when training began. This is used to print the total
            training time.
        file_path : str
            the list of cumulative rewards from every episode in the evaluation
            phase
        seg : dict
            the current segment of the trajectory (see _collect_samples return
            for more information)
        mean_losses : TODO
            TODO
        vpredbefore : TODO
            TODO
        tdlamret : TODO
            TODO
        """
        lens, rews = seg["ep_lens"], seg["ep_rets"]
        current_it_timesteps = seg["total_timestep"]

        self.len_buffer.extend(lens)
        self.reward_buffer.extend(rews)
        self.episodes_so_far += len(lens)
        self.timesteps_so_far += current_it_timesteps
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
            "total_steps": self.timesteps_so_far,
            "epoch": self.iters_so_far,
            "duration": time.time() - t_start,
            "steps_per_second": self.timesteps_so_far / (time.time()-t_start),
            # TODO: what is this?
            "explained_variance": explained_variance(vpredbefore, tdlamret),
        }
        for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
            stats[loss_name] = loss_val

        # Save combined stats in a csv file.
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
