import os
import gym

from hbaselines.common.utils import ensure_dir
from hbaselines.trpo.policy import FeedForwardPolicy
from hbaselines.trpo.ppo import PPO

ENV_NAMES = [
    "HalfCheetah-v2",
    # "Hopper-v2",
    # "Walker2d-v2",
    # "Ant-v2",
    # "Reacher-v2",
    # "InvertedPendulum-v2",
    # "InvertedDoublePendulum-v2",
]

NUM_SEEDS = 1

IGNORE_DONES = False
NONSHARED_VF = False
DUEL_VF = False


def main():
    assert not (not NONSHARED_VF and DUEL_VF), \
        "Cannot use duel_vf without non-shared value functions."

    for env_name in ENV_NAMES:
        for seed in range(NUM_SEEDS):
            # Create the environment.
            env = gym.make(env_name)

            log_dir = "./{}/".format(env_name)

            # base log directory
            if not any([IGNORE_DONES, NONSHARED_VF, DUEL_VF]):
                log_dir = os.path.join(log_dir, "baseline")
            else:
                # log_dir = "./"
                if IGNORE_DONES:
                    log_dir += "ignore_done,"
                if NONSHARED_VF:
                    log_dir += "nonshared_vf,"
                if DUEL_VF:
                    log_dir += "duel_vf,"
                # remove the last comma
                log_dir = log_dir[:-1]

            # add seed to the log directory
            log_dir = os.path.join(log_dir, "{}".format(seed))
            ensure_dir(log_dir)

            # Create the model and perform the training operation.
            model = PPO(
                FeedForwardPolicy,
                env,
                verbose=1,
                # ignore_dones=IGNORE_DONES,
                policy_kwargs={
                    "shared": not NONSHARED_VF,
                    "duel_vf": DUEL_VF,
                }
            )
            model.learn(total_timesteps=1000000, log_dir=log_dir, seed=seed)

            # clear created variables
            del model, env


if __name__ == "__main__":
    main()
