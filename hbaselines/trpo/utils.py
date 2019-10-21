from gym.spaces import Box
import numpy as np

from stable_baselines.common.vec_env import VecEnv


def traj_segment_generator(policy,
                           env,
                           horizon,
                           reward_giver=None):
    """Compute target value using TD estimator, and advantage with GAE.

    Parameters
    ----------
    policy : MLPPolicy  FIXME
        the policy
    env : gym.Env
        the environment
    horizon : int
        the number of timesteps to run per batch
    reward_giver : TransitionClassifier  FIXME
        the reward predictor from observation and action

    Returns
    -------
    dict
        generator that returns a dict with the following keys:

        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards
        TODO: remove
        - true_rewards: (numpy float) if gail is used it is the original reward
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool) True if first timestep of an episode,
          used for GAE
        - actions: (np.ndarray) actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Initialize state variables
    step = 0
    # not used, just so we have the datatype
    action = env.action_space.sample()
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0  # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    true_rewards = np.zeros(horizon, 'float32')
    rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action for _ in range(horizon)])
    episode_start = True  # marks if we're on first timestep of an episode

    while True:
        action, vpred, _ = policy.step(
            observation.reshape(-1, *observation.shape))
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
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
            _, vpred, _ = policy.step(
                observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            # Reset current iteration length
            current_it_len = 0
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        episode_starts[i] = episode_start

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, Box):
            clipped_action = np.clip(action,
                                     a_min=env.action_space.low,
                                     a_max=env.action_space.high)

        observation, reward, done, info = env.step(clipped_action[0])
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
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """Compute target value using TD estimator, and advantage with GAE.

    Parameters
    ----------
    seg : dict
        the current segment of the trajectory (see traj_segment_generator
        return for more information)
    gamma : float
        Discount factor
    lam : float
        GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if
    # last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] \
            * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta \
            + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    """Flatten a python list of list.

    Parameters
    ----------
    listoflists : list of list
        the list of flatten

    Returns
    -------
    list
        flattened list
    """
    return [el for list_ in listoflists for el in list_]
