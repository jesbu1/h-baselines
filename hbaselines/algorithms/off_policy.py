"""Script algorithm contain the base off-policy RL algorithm class.

Supported algorithms through this class:

* Twin Delayed Deep Deterministic Policy Gradient (TD3): see
  https://arxiv.org/pdf/1802.09477.pdf
* Soft Actor Critic (SAC): see https://arxiv.org/pdf/1801.01290.pdf

This algorithm class also contains modifications to support contextual
environments and hierarchical policies.
"""
import ray
import os
import time
import csv
import random
import numpy as np
import tensorflow as tf
import math
from collections import deque
from copy import deepcopy
from gym.spaces import Box

from hbaselines.algorithms.utils import is_td3_policy, is_sac_policy
from hbaselines.algorithms.utils import is_feedforward_policy
from hbaselines.algorithms.utils import is_goal_conditioned_policy
from hbaselines.algorithms.utils import is_multiagent_policy
from hbaselines.algorithms.utils import add_fingerprint
from hbaselines.algorithms.utils import get_obs
from hbaselines.utils.sampler import Sampler
from hbaselines.utils.tf_util import make_session
from hbaselines.utils.misc import ensure_dir
from hbaselines.utils.env_util import create_env


# =========================================================================== #
#                          Policy parameters for TD3                          #
# =========================================================================== #

TD3_PARAMS = dict(
    # scaling term to the range of the action space, that is subsequently used
    # as the standard deviation of Gaussian noise added to the action if
    # `apply_noise` is set to True in `get_action`
    noise=0.1,
    # standard deviation term to the noise from the output of the target actor
    # policy. See TD3 paper for more.
    target_policy_noise=0.2,
    # clipping term for the noise injected in the target actor policy
    target_noise_clip=0.5,
)


# =========================================================================== #
#                          Policy parameters for SAC                          #
# =========================================================================== #

SAC_PARAMS = dict(
    # target entropy used when learning the entropy coefficient. If set to
    # None, a heuristic value is used.
    target_entropy=None,
)


# =========================================================================== #
#       Policy parameters for FeedForwardPolicy (shared by TD3 and SAC)       #
# =========================================================================== #

FEEDFORWARD_PARAMS = dict(
    # the max number of transitions to store
    buffer_size=200000,
    # the size of the batch for learning the policy
    batch_size=128,
    # the actor learning rate
    actor_lr=3e-4,
    # the critic learning rate
    critic_lr=3e-4,
    # the soft update coefficient (keep old values, between 0 and 1)
    tau=0.005,
    # the discount rate
    gamma=0.99,
    # enable layer normalisation
    layer_norm=False,
    # the size of the neural network for the policy
    layers=[256, 256],
    # the activation function to use in the neural network
    act_fun=tf.nn.relu,
    # specifies whether to use the huber distance function as the loss for the
    # critic. If set to False, the mean-squared error metric is used instead
    use_huber=False,
)


# =========================================================================== #
#     Policy parameters for GoalConditionedPolicy (shared by TD3 and SAC)     #
# =========================================================================== #

GOAL_CONDITIONED_PARAMS = FEEDFORWARD_PARAMS.copy()
GOAL_CONDITIONED_PARAMS.update(dict(
    # number of levels within the hierarchy. Must be greater than 1. Two levels
    # correspond to a Manager/Worker paradigm.
    num_levels=2,
    # meta-policy action period
    meta_period=10,
    # the reward function to be used by lower-level policies. See the base
    # goal-conditioned policy for a description.
    intrinsic_reward_type="negative_distance",
    # the value that the intrinsic reward should be scaled by
    intrinsic_reward_scale=1,
    # specifies whether the goal issued by the higher-level policies is meant
    # to be a relative or absolute goal, i.e. specific state or change in state
    relative_goals=False,
    # whether to use off-policy corrections during the update procedure. See:
    # https://arxiv.org/abs/1805.08296
    off_policy_corrections=False,
    # whether to include hindsight action and goal transitions in the replay
    # buffer. See: https://arxiv.org/abs/1712.00948
    hindsight=False,
    # rate at which the original (non-hindsight) sample is stored in the
    # replay buffer as well. Used only if `hindsight` is set to True.
    subgoal_testing_rate=0.3,
    # whether to use the connected gradient update actor update procedure to
    # the higher-level policies. See: https://arxiv.org/abs/1912.02368v1
    connected_gradients=False,
    # weights for the gradients of the loss of the lower-level policies with
    # respect to the parameters of the higher-level policies. Only used if
    # `connected_gradients` is set to True.
    cg_weights=0.0005,
    # specifies whether to add a time-dependent fingerprint to the observations
    use_fingerprints=False,
    # the low and high values for each fingerprint element, if they are being
    # used
    fingerprint_range=([0, 0], [5, 5]),
    # specifies whether to use centralized value functions
    centralized_value_functions=False,
    # specifies whether you are pre-training the lower-level policies. Actions
    # by the high-level policy are randomly sampled from its action space.
    pretrain_worker=False,
    # path to the pre-trained worker policy checkpoints
    pretrain_path=None,
    # the checkpoint number to use within the worker policy path. If set to
    # None, the most recent checkpoint is used.
    pretrain_ckpt=None,
))


# =========================================================================== #
#    Policy parameters for MultiFeedForwardPolicy (shared by TD3 and SAC)     #
# =========================================================================== #

MULTI_FEEDFORWARD_PARAMS = FEEDFORWARD_PARAMS.copy()
MULTI_FEEDFORWARD_PARAMS.update(dict(
    # whether to use a shared policy for all agents
    shared=False,
    # whether to use an algorithm-specific variant of the MADDPG algorithm
    maddpg=False,
))


class OffPolicyRLAlgorithm(object):
    """Off-policy RL algorithm class.

    Supports the training of TD3 and SAC policies.

    Attributes
    ----------
    policy : type [ hbaselines.base_policies.ActorCriticPolicy ]
        the policy model to use
    env_name : str
        name of the environment. Affects the action bounds of the higher-level
        policies
    sampler : list of hbaselines.utils.sampler.Sampler
        the training environment sampler object. One environment is provided
        for each CPU
    eval_env : gym.Env or list of gym.Env
        the environment(s) to evaluate from
    nb_train_steps : int
        the number of training steps
    nb_rollout_steps : int
        the number of rollout steps
    nb_eval_episodes : int
        the number of evaluation episodes
    actor_update_freq : int
        number of training steps per actor policy update step. The critic
        policy is updated every training step.
    meta_update_freq : int
        number of training steps per meta policy update step. The actor policy
        of the meta-policy is further updated at the frequency provided by the
        actor_update_freq variable. Note that this value is only relevant when
        using the GoalConditionedPolicy policy.
    reward_scale : float
        the value the reward should be scaled by
    render : bool
        enable rendering of the training environment
    render_eval : bool
        enable rendering of the evaluation environment
    eval_deterministic : bool
        if set to True, the policy provides deterministic actions to the
        evaluation environment. Otherwise, stochastic or noisy actions are
        returned.
    num_envs : int
        number of environments used to run simulations in parallel. Each
        environment is run on a separate CPUS and uses the same policy as the
        rest. Must be less than or equal to nb_rollout_steps.
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    ac_space : gym.spaces.*
        the action space of the training environment
    ob_space : gym.spaces.*
        the observation space of the training environment
    co_space : gym.spaces.*
        the context space of the training environment (i.e. the same of the
        desired environmental goal)
    policy_kwargs : dict
        policy-specific hyperparameters
    horizon : int
        time horizon, which is used to check if an environment terminated early
        and used to compute the done mask as per TD3 implementation (see
        appendix A of their paper). If the horizon cannot be found, it is
        assumed to be 500 (default value for most gym environments).
    graph : tf.Graph
        the current tensorflow graph
    policy_tf : hbaselines.base_policies.ActorCriticPolicy
        the policy object
    sess : tf.compat.v1.Session
        the current tensorflow session
    summary : tf.Summary
        tensorboard summary object
    obs : list of array_like or list of dict < str, array_like >
        the most recent training observation. If you are using a multi-agent
        environment, this will be a dictionary of observations for each agent,
        indexed by the agent ID. One element for each environment.
    all_obs : list of array_like or list of None
        additional information, used by MADDPG variants of the multi-agent
        policy to pass full-state information. One element for each environment
    episode_step : list of int
        the number of steps since the most recent rollout began. One for each
        environment.
    episodes : int
        the total number of rollouts performed since training began
    total_steps : int
        the total number of steps that have been executed since training began
    epoch_episode_rewards : list of float
        a list of cumulative rollout rewards from the most recent training
        iterations
    epoch_episode_steps : list of int
        a list of rollout lengths from the most recent training iterations
    epoch_episodes : int
        the total number of rollouts performed since the most recent training
        iteration began
    epoch : int
        the total number of training iterations
    episode_rew_history : list of float
        the cumulative return from the last 100 training episodes
    episode_reward : list of float
        the cumulative reward since the most reward began. One for each
        environment.
    saver : tf.compat.v1.train.Saver
        tensorflow saver object
    trainable_vars : list of str
        the trainable variables
    rew_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last epoch. Used
        for logging purposes.
    rew_history_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last 100
        episodes. Used for logging purposes.
    eval_rew_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation return from the last time
        evaluations occurred. Used for logging purposes.
    eval_success_ph : tf.compat.v1.placeholder
        placeholder for the average evaluation success rate from the last time
        evaluations occurred. Used for logging purposes.
    """

    def __init__(self,
                 policy,
                 env,
                 eval_env=None,
                 nb_train_steps=1,
                 nb_rollout_steps=1,
                 nb_eval_episodes=50,
                 actor_update_freq=2,
                 meta_update_freq=10,
                 reward_scale=1.,
                 render=False,
                 render_eval=False,
                 eval_deterministic=True,
                 num_envs=1,
                 verbose=0,
                 policy_kwargs=None,
                 _init_setup_model=True):
        """Instantiate the algorithm object.

        Parameters
        ----------
        policy : type [ hbaselines.base_policies.ActorCriticPolicy ]
            the policy model to use
        env : gym.Env or str
            the environment to learn from (if registered in Gym, can be str)
        eval_env : gym.Env or str
            the environment to evaluate from (if registered in Gym, can be str)
        nb_train_steps : int
            the number of training steps
        nb_rollout_steps : int
            the number of rollout steps
        nb_eval_episodes : int
            the number of evaluation episodes
        actor_update_freq : int
            number of training steps per actor policy update step. The critic
            policy is updated every training step.
        meta_update_freq : int
            number of training steps per meta policy update step. The actor
            policy of the meta-policy is further updated at the frequency
            provided by the actor_update_freq variable. Note that this value is
            only relevant when using the GoalConditionedPolicy policy.
        reward_scale : float
            the value the reward should be scaled by
        render : bool
            enable rendering of the training environment
        render_eval : bool
            enable rendering of the evaluation environment
        eval_deterministic : bool
            if set to True, the policy provides deterministic actions to the
            evaluation environment. Otherwise, stochastic or noisy actions are
            returned.
        num_envs : int
            number of environments used to run simulations in parallel. Each
            environment is run on a separate CPUS and uses the same policy as
            the rest. Must be less than or equal to nb_rollout_steps.
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        policy_kwargs : dict
            policy-specific hyperparameters
        _init_setup_model : bool
            Whether or not to build the network at the creation of the instance

        Raises
        ------
        AssertionError
            if num_envs > nb_rollout_steps
        """
        shared = False if policy_kwargs is None else \
            policy_kwargs.get("shared", False)
        maddpg = False if policy_kwargs is None else \
            policy_kwargs.get("maddpg", False)

        # Run assertions.
        assert num_envs <= nb_rollout_steps, \
            "num_envs must be less than or equal to nb_rollout_steps"

        # Instantiate the ray instance.
        ray.init(num_cpus=num_envs+1, ignore_reinit_error=True)

        self.policy = policy
        self.env_name = deepcopy(env) if isinstance(env, str) \
            else env.__str__()
        self.eval_env, _ = create_env(
            eval_env, render_eval, shared, maddpg, evaluate=True)
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_eval_episodes = nb_eval_episodes
        self.actor_update_freq = actor_update_freq
        self.meta_update_freq = meta_update_freq
        self.reward_scale = reward_scale
        self.render = render
        self.render_eval = render_eval
        self.eval_deterministic = eval_deterministic
        self.num_envs = num_envs
        self.verbose = verbose
        self.policy_kwargs = {'verbose': verbose}

        # Create the environment and collect the initial observations.
        self.sampler = [
            Sampler.remote(
                env_name=env,
                render=render,
                shared=shared,
                maddpg=maddpg,
                env_num=env_num,
                evaluate=False,
            )
            for env_num in range(num_envs)
        ]
        obs = ray.get([s.get_init_obs.remote() for s in self.sampler])
        self.obs = [get_obs(o)[0] for o in obs]
        self.all_obs = [get_obs(o)[1] for o in obs]

        # Collect the spaces of the environments.
        self.ac_space = ray.get(self.sampler[0].action_space.remote())
        self.ob_space = ray.get(self.sampler[0].observation_space.remote())
        self.co_space = ray.get(self.sampler[0].context_space.remote())

        # Add the default policy kwargs to the policy_kwargs term.
        if is_feedforward_policy(policy):
            self.policy_kwargs.update(FEEDFORWARD_PARAMS.copy())
        elif is_goal_conditioned_policy(policy):
            self.policy_kwargs.update(GOAL_CONDITIONED_PARAMS.copy())
            self.policy_kwargs['env_name'] = self.env_name.__str__()
            self.policy_kwargs['num_envs'] = num_envs
        elif is_multiagent_policy(policy):
            self.policy_kwargs.update(MULTI_FEEDFORWARD_PARAMS.copy())
            self.policy_kwargs["all_ob_space"] = ray.get(
                self.sampler[0].all_observation_space.remote())

        if is_td3_policy(policy):
            self.policy_kwargs.update(TD3_PARAMS.copy())
        elif is_sac_policy(policy):
            self.policy_kwargs.update(SAC_PARAMS.copy())

        self.policy_kwargs.update(policy_kwargs or {})

        # Compute the time horizon, which is used to check if an environment
        # terminated early and used to compute the done mask for TD3.
        self.horizon = ray.get(self.sampler[0].horizon.remote())

        # init
        self.graph = None
        self.policy_tf = None
        self.sess = None
        self.summary = None
        self.episode_step = [0 for _ in range(num_envs)]
        self.episodes = 0
        self.total_steps = 0
        self.epoch_episode_steps = []
        self.epoch_episode_rewards = []
        self.epoch_episodes = 0
        self.epoch = 0
        self.episode_rew_history = deque(maxlen=100)
        self.episode_reward = [0 for _ in range(num_envs)]
        self.rew_ph = None
        self.rew_history_ph = None
        self.eval_rew_ph = None
        self.eval_success_ph = None
        self.saver = None

        if self.policy_kwargs.get("use_fingerprints", False):
            # Append the fingerprint dimension to the observation dimension.
            fingerprint_range = self.policy_kwargs["fingerprint_range"]
            low = np.concatenate(
                (self.ob_space.low, fingerprint_range[0]))
            high = np.concatenate(
                (self.ob_space.high, fingerprint_range[1]))
            self.ob_space = Box(low, high, dtype=np.float32)

            # Add the fingerprint term to the first observation.
            self.obs = [add_fingerprint(obs, 0, 1, True) for obs in self.obs]

        # Create the model variables and operations.
        if _init_setup_model:
            self.trainable_vars = self.setup_model()

    def setup_model(self):
        """Create the graph, session, policy, and summary objects."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create the tensorflow session.
            self.sess = make_session(num_cpu=3, graph=self.graph)

            # Create the policy.
            self.policy_tf = self.policy(
                self.sess,
                self.ob_space,
                self.ac_space,
                self.co_space,
                **self.policy_kwargs
            )

            # for tensorboard logging
            with tf.compat.v1.variable_scope("Train"):
                self.rew_ph = tf.compat.v1.placeholder(tf.float32)
                self.rew_history_ph = tf.compat.v1.placeholder(tf.float32)

            # Add tensorboard scalars for the return, return history, and
            # success rate.
            tf.compat.v1.summary.scalar("Train/return", self.rew_ph)
            tf.compat.v1.summary.scalar("Train/return_history",
                                        self.rew_history_ph)

            # Create the tensorboard summary.
            self.summary = tf.compat.v1.summary.merge_all()

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.compat.v1.global_variables_initializer())
                self.policy_tf.initialize()

            return tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    def _policy(self,
                obs,
                context,
                apply_noise=True,
                random_actions=False,
                env_num=0):
        """Get the actions and critic output, from a given observation.

        Parameters
        ----------
        obs : array_like
            the observation
        context : array_like or None
            the contextual term. Set to None if no context is provided by the
            environment.
        apply_noise : bool
            enable the noise
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.

        Returns
        -------
        list of float
            the action value
        """
        # Reshape the observation to match the input structure of the policy.
        if isinstance(obs, dict):
            # In multi-agent environments, observations come in dict form
            for key in obs.keys():
                # Shared policies with have one observation space, while
                # independent policies have a different observation space based
                # on their agent ID.
                if isinstance(self.ob_space, dict):
                    ob_shape = self.ob_space[key].shape
                else:
                    ob_shape = self.ob_space.shape
                obs[key] = np.array(obs[key]).reshape((-1,) + ob_shape)
        else:
            obs = np.array(obs).reshape((-1,) + self.ob_space.shape)

        action = self.policy_tf.get_action(
            obs, context,
            apply_noise=apply_noise,
            random_actions=random_actions,
            env_num=env_num,
        )

        # Flatten the actions. Dictionaries correspond to multi-agent policies.
        if isinstance(action, dict):
            action = {key: action[key].flatten() for key in action.keys()}
        else:
            action = action.flatten()

        return action

    def _store_transition(self,
                          obs0,
                          context0,
                          action,
                          reward,
                          obs1,
                          context1,
                          terminal1,
                          is_final_step,
                          env_num=0,
                          evaluate=False,
                          **kwargs):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : array_like
            the last observation
        action : array_like
            the action
        reward : float
            the reward
        obs1 : array_like
            the current observation
        terminal1 : bool
            is the episode done
        is_final_step : bool
            whether the time horizon was met in the step corresponding to the
            current sample. This is used by the TD3 algorithm to augment the
            done mask.
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.
        evaluate : bool
            whether the sample is being provided by the evaluation environment.
            If so, the data is not stored in the replay buffer.
        kwargs : dict
            additional parameters, containing the current and next-step full
            observations for policies using MADDPG
        """
        # Scale the rewards by the provided term. Rewards are dictionaries when
        # training independent multi-agent policies.
        if isinstance(reward, dict):
            reward = {k: self.reward_scale * reward[k] for k in reward.keys()}
        else:
            reward *= self.reward_scale

        self.policy_tf.store_transition(
            obs0=obs0,
            context0=context0,
            action=action,
            reward=reward,
            obs1=obs1,
            context1=context1,
            done=terminal1,
            is_final_step=is_final_step,
            env_num=env_num,
            evaluate=evaluate,
            **(kwargs if self.policy_kwargs.get("maddpg", False) else {}),
        )

    def learn(self,
              total_steps,
              log_dir=None,
              seed=None,
              log_interval=2000,
              eval_interval=50000,
              save_interval=10000,
              initial_exploration_steps=10000):
        """Perform the complete training operation.

        Parameters
        ----------
        total_steps : int
            the total number of samples to train on
        log_dir : str
            the directory where the training and evaluation statistics, as well
            as the tensorboard log, should be stored
        seed : int or None
            the initial seed for training, if None: keep current seed
        log_interval : int
            the number of training steps before logging training results
        eval_interval : int
            number of simulation steps in the training environment before an
            evaluation is performed
        save_interval : int
            number of simulation steps in the training environment before the
            model is saved
        initial_exploration_steps : int
            number of timesteps that the policy is run before training to
            initialize the replay buffer with samples
        """
        # Create a saver object.
        self.saver = tf.compat.v1.train.Saver(
            self.trainable_vars,
            max_to_keep=total_steps // save_interval)

        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)
        ensure_dir(os.path.join(log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        save_path = os.path.join(log_dir, "tb_log")
        writer = tf.compat.v1.summary.FileWriter(save_path)

        # file path for training and evaluation results
        train_filepath = os.path.join(log_dir, "train.csv")
        eval_filepath = os.path.join(log_dir, "eval.csv")

        # Setup the seed value.
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        if self.verbose >= 2:
            print('Using agent with the following configuration:')
            print(str(self.__dict__.items()))

        eval_steps_incr = 0
        save_steps_incr = 0
        start_time = time.time()

        with self.sess.as_default(), self.graph.as_default():
            # Collect preliminary random samples.
            print("Collecting initial exploration samples...")
            self._collect_samples(total_steps,
                                  run_steps=initial_exploration_steps,
                                  random_actions=True)
            print("Done!")

            # Reset total statistics variables.
            self.episodes = 0
            self.total_steps = 0
            self.episode_rew_history = deque(maxlen=100)

            while True:
                # Reset epoch-specific variables.
                self.epoch_episodes = 0
                self.epoch_episode_steps = []
                self.epoch_episode_rewards = []

                for _ in range(round(log_interval / self.nb_rollout_steps)):
                    # If the requirement number of time steps has been met,
                    # terminate training.
                    if self.total_steps >= total_steps:
                        return

                    # Perform rollouts.
                    self._collect_samples(total_steps)

                    # Train.
                    self._train()

                # Log statistics.
                self._log_training(train_filepath, start_time)

                # Evaluate.
                if self.eval_env is not None and \
                        (self.total_steps - eval_steps_incr) >= eval_interval:
                    eval_steps_incr += eval_interval

                    # Run the evaluation operations over the evaluation env(s).
                    # Note that multiple evaluation envs can be provided.
                    if isinstance(self.eval_env, list):
                        eval_rewards = []
                        eval_successes = []
                        eval_info = []
                        for env in self.eval_env:
                            rew, suc, inf = self._evaluate(total_steps, env)
                            eval_rewards.append(rew)
                            eval_successes.append(suc)
                            eval_info.append(inf)
                    else:
                        eval_rewards, eval_successes, eval_info = \
                            self._evaluate(total_steps, self.eval_env)

                    # Log the evaluation statistics.
                    self._log_eval(eval_filepath, start_time, eval_rewards,
                                   eval_successes, eval_info)

                # Run and store summary.
                if writer is not None:
                    td_map = self.policy_tf.get_td_map()

                    # Check if td_map is empty.
                    if not td_map:
                        break

                    td_map.update({
                        self.rew_ph: np.mean(self.epoch_episode_rewards),
                        self.rew_history_ph: np.mean(self.episode_rew_history),
                    })
                    summary = self.sess.run(self.summary, td_map)
                    writer.add_summary(summary, self.total_steps)

                # Save a checkpoint of the model.
                if (self.total_steps - save_steps_incr) >= save_interval:
                    save_steps_incr += save_interval
                    self.save(os.path.join(log_dir, "checkpoints/itr"))

                # Update the epoch count.
                self.epoch += 1

    def save(self, save_path):
        """Save the parameters of a tensorflow model.

        Parameters
        ----------
        save_path : str
            Prefix of filenames created for the checkpoint
        """
        self.saver.save(self.sess, save_path, global_step=self.total_steps)

    def load(self, load_path):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        self.saver.restore(self.sess, load_path)

    def _collect_samples(self,
                         total_steps,
                         run_steps=None,
                         random_actions=False):
        """Perform the sample collection operation over multiple steps.

        This method calls collect_sample for a multiple steps, and attempts to
        run the operation in parallel if multiple environments are available.

        Parameters
        ----------
        total_steps : int
            the total number of samples to train on. Used by the fingerprint
            element
        run_steps : int, optional
            number of steps to collect samples from. If not provided, the value
            defaults to `self.nb_rollout_steps`.
        random_actions : bool
            if set to True, actions are sampled randomly from the action space
            instead of being computed by the policy. This is used for
            exploration purposes.
        """
        # Loop through the sampling procedure the number of times it would
        # require to run through each environment in parallel until the number
        # of required steps have been collected.
        run_steps = run_steps or self.nb_rollout_steps
        n_itr = math.ceil(run_steps / self.num_envs)
        for itr in range(n_itr):
            n_steps = self.num_envs if itr < n_itr - 1 \
                else run_steps - (n_itr - 1) * self.num_envs

            # Predict next action. Use random actions when initializing the
            # replay buffer.
            action = [self._policy(
                obs=self.obs[env_num],
                context=ray.get(self.sampler[env_num].get_context.remote()),
                apply_noise=True,
                random_actions=random_actions,
                env_num=env_num,
            ) for env_num in range(n_steps)]

            # Update the environment.
            ret = ray.get([
                self.sampler[env_num].collect_sample.remote(
                    action=action[env_num],
                    multiagent=is_multiagent_policy(self.policy),
                    steps=self.total_steps,
                    total_steps=total_steps,
                    use_fingerprints=self.policy_kwargs.get(
                        "use_fingerprints", False)
                )
                for env_num in range(n_steps)
            ])

            for ret_i in ret:
                num = ret_i["env_num"]
                context = ret_i["context"]
                action = ret_i["action"]
                reward = ret_i["reward"]
                obs = ret_i["obs"]
                done = ret_i["done"]
                all_obs = ret_i["all_obs"]

                # Store a transition in the replay buffer.
                self._store_transition(
                    obs0=self.obs[num],
                    context0=context,
                    action=action,
                    reward=reward,
                    obs1=obs[0] if done else obs,
                    context1=context,
                    terminal1=done,
                    is_final_step=(self.episode_step[num] >= self.horizon - 1),
                    all_obs0=self.all_obs[num],
                    all_obs1=all_obs[0] if done else all_obs,
                    env_num=num,
                )

                # Book-keeping.
                self.total_steps += 1
                self.episode_step[num] += 1
                if isinstance(reward, dict):
                    self.episode_reward[num] += sum(
                        reward[k] for k in reward.keys())
                else:
                    self.episode_reward[num] += reward

                # Update the current observation.
                self.obs[num] = (obs[1] if done else obs).copy()
                self.all_obs[num] = all_obs[1] if done else all_obs

                # Handle episode done.
                if done:
                    self.epoch_episode_rewards.append(self.episode_reward[num])
                    self.episode_rew_history.append(self.episode_reward[num])
                    self.epoch_episode_steps.append(self.episode_step[num])
                    self.episode_reward[num] = 0
                    self.episode_step[num] = 0
                    self.epoch_episodes += 1
                    self.episodes += 1

    def _train(self):
        """Perform the training operation.

        Through this method, the actor and critic networks are updated within
        the policy, and the summary information is logged to tensorboard.
        """
        # Added to adjust the actor update frequency based on the rate at which
        # training occurs.
        total_steps = int(self.total_steps / self.nb_rollout_steps)

        if is_goal_conditioned_policy(self.policy):
            # specifies whether to update the meta actor and critic
            # policies based on the meta and actor update frequencies
            kwargs = {
                "update_meta":
                    total_steps % self.meta_update_freq == 0,
                "update_meta_actor":
                    total_steps %
                    (self.meta_update_freq * self.actor_update_freq) == 0
            }
        else:
            kwargs = {}

        # Specifies whether to update the actor policy, base on the actor
        # update frequency.
        update = total_steps % self.actor_update_freq == 0

        # Run a step of training from batch.
        for _ in range(self.nb_train_steps):
            _ = self.policy_tf.update(update_actor=update, **kwargs)

    def _evaluate(self, total_steps, env):
        """Perform the evaluation operation.

        This method runs the evaluation environment for a number of episodes
        and returns the cumulative rewards and successes from each environment.

        Parameters
        ----------
        total_steps : int
            the total number of samples to train on
        env : gym.Env
            the evaluation environment that the policy is meant to be tested on

        Returns
        -------
        list of float
            the list of cumulative rewards from every episode in the evaluation
            phase
        list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        dict
            additional information that is meant to be logged
        """
        num_steps = deepcopy(self.total_steps)
        eval_episode_rewards = []
        eval_episode_successes = []
        ret_info = {'initial': [], 'final': [], 'average': []}

        if self.verbose >= 1:
            for _ in range(3):
                print("-------------------")
            print("Running evaluation for {} episodes:".format(
                self.nb_eval_episodes))

        # Clear replay buffer-related memory in the policy to allow for the
        # meta-actions to properly updated.
        if is_goal_conditioned_policy(self.policy):
            for env_num in range(self.num_envs):
                self.policy_tf.clear_memory(env_num)

        for i in range(self.nb_eval_episodes):
            # Reset the environment.
            eval_obs = env.reset()
            eval_obs, eval_all_obs = get_obs(eval_obs)

            # Add the fingerprint term, if needed.
            eval_obs = add_fingerprint(
                obs=eval_obs,
                steps=self.total_steps,
                total_steps=total_steps,
                use_fingerprints=self.policy_kwargs.get(
                    "use_fingerprints", False),
            )

            # Reset rollout-specific variables.
            eval_episode_reward = 0.
            eval_episode_step = 0

            rets = np.array([])
            while True:
                # Collect the contextual term. None if it is not passed.
                context = [env.current_context] \
                    if hasattr(env, "current_context") else None

                eval_action = self._policy(
                    obs=eval_obs,
                    context=context,
                    apply_noise=not self.eval_deterministic,
                    random_actions=False,
                    env_num=0,
                )

                # Update the environment.
                obs, eval_r, done, info = env.step(eval_action)
                obs, all_obs = get_obs(obs)

                # Visualize the current step.
                if self.render_eval:
                    self.eval_env.render()  # pragma: no cover

                # Add the distance to this list for logging purposes (applies
                # only to the Ant* environments).
                if hasattr(env, "current_context"):
                    context = getattr(env, "current_context")
                    reward_fn = getattr(env, "contextual_reward")
                    rets = np.append(rets, reward_fn(eval_obs, context, obs))

                # Get the contextual term.
                context0 = context1 = getattr(env, "current_context", None)

                # Store a transition in the replay buffer. This is just for the
                # purposes of calling features in the store_transition method
                # of the policy.
                self._store_transition(
                    obs0=eval_obs,
                    context0=context0,
                    action=eval_action,
                    reward=eval_r,
                    obs1=obs,
                    context1=context1,
                    terminal1=False,
                    is_final_step=False,
                    all_obs0=eval_all_obs,
                    all_obs1=all_obs,
                    evaluate=True,
                )

                # Update the previous step observation.
                eval_obs = obs.copy()
                eval_all_obs = all_obs

                # Add the fingerprint term, if needed.
                eval_obs = add_fingerprint(
                    obs=eval_obs,
                    steps=self.total_steps,
                    total_steps=total_steps,
                    use_fingerprints=self.policy_kwargs.get(
                        "use_fingerprints", False),
                )

                # Increment the reward and step count.
                num_steps += 1
                eval_episode_reward += eval_r
                eval_episode_step += 1

                if done:
                    eval_episode_rewards.append(eval_episode_reward)
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        eval_episode_successes.append(float(maybe_is_success))

                    if self.verbose >= 1:
                        if rets.shape[0] > 0:
                            print("%d/%d: initial: %.3f, final: %.3f, average:"
                                  " %.3f, success: %d"
                                  % (i + 1, self.nb_eval_episodes, rets[0],
                                     rets[-1], float(rets.mean()),
                                     int(info.get('is_success'))))
                        else:
                            print("%d/%d" % (i + 1, self.nb_eval_episodes))

                    if hasattr(env, "current_context"):
                        ret_info['initial'].append(rets[0])
                        ret_info['final'].append(rets[-1])
                        ret_info['average'].append(float(rets.mean()))

                    # Exit the loop.
                    break

        if self.verbose >= 1:
            print("Done.")
            print("Average return: {}".format(np.mean(eval_episode_rewards)))
            if len(eval_episode_successes) > 0:
                print("Success rate: {}".format(
                    np.mean(eval_episode_successes)))
            for _ in range(3):
                print("-------------------")
            print("")

        # get the average of the reward information
        ret_info['initial'] = np.mean(ret_info['initial'])
        ret_info['final'] = np.mean(ret_info['final'])
        ret_info['average'] = np.mean(ret_info['average'])

        # Clear replay buffer-related memory in the policy once again so that
        # it does not affect the training procedure.
        if is_goal_conditioned_policy(self.policy):
            self.policy_tf.clear_memory()

        return eval_episode_rewards, eval_episode_successes, ret_info

    def _log_training(self, file_path, start_time):
        """Log training statistics.

        Parameters
        ----------
        file_path : str
            the list of cumulative rewards from every episode in the evaluation
            phase
        start_time : float
            the time when training began. This is used to print the total
            training time.
        """
        # Log statistics.
        duration = time.time() - start_time

        combined_stats = {
            # Rollout statistics.
            'rollout/episodes': self.epoch_episodes,
            'rollout/episode_steps': np.mean(self.epoch_episode_steps),
            'rollout/return': np.mean(self.epoch_episode_rewards),
            'rollout/return_history': np.mean(self.episode_rew_history),

            # Total statistics.
            'total/epochs': self.epoch + 1,
            'total/steps': self.total_steps,
            'total/duration': duration,
            'total/steps_per_second': self.total_steps / duration,
            'total/episodes': self.episodes,
        }

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=combined_stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(combined_stats)

        # Print statistics.
        print("-" * 67)
        for key in sorted(combined_stats.keys()):
            val = combined_stats[key]
            print("| {:<30} | {:<30} |".format(key, val))
        print("-" * 67)
        print('')

    def _log_eval(self, file_path, start_time, rewards, successes, info):
        """Log evaluation statistics.

        Parameters
        ----------
        file_path : str
            path to the evaluation csv file
        start_time : float
            the time when training began. This is used to print the total
            training time.
        rewards : array_like
            the list of cumulative rewards from every episode in the evaluation
            phase
        successes : list of bool
            a list of boolean terms representing if each episode ended in
            success or not. If the list is empty, then the environment did not
            output successes or failures, and the success rate will be set to
            zero.
        info : dict
            additional information that is meant to be logged
        """
        duration = time.time() - start_time

        if isinstance(info, dict):
            rewards = [rewards]
            successes = [successes]
            info = [info]

        for i, (rew, suc, info_i) in enumerate(zip(rewards, successes, info)):
            if len(suc) > 0:
                success_rate = np.mean(suc)
            else:
                success_rate = 0  # no success rate to log

            evaluation_stats = {
                "duration": duration,
                "total_step": self.total_steps,
                "success_rate": success_rate,
                "average_return": np.mean(rew)
            }
            # Add additional evaluation information.
            evaluation_stats.update(info_i)

            if file_path is not None:
                # Add an evaluation number to the csv file in case of multiple
                # evaluation environments.
                eval_fp = file_path[:-4] + "_{}.csv".format(i)
                exists = os.path.exists(eval_fp)

                # Save evaluation statistics in a csv file.
                with open(eval_fp, "a") as f:
                    w = csv.DictWriter(f, fieldnames=evaluation_stats.keys())
                    if not exists:
                        w.writeheader()
                    w.writerow(evaluation_stats)
