import itertools
import random
import subprocess
import os
from absl import logging, flags, app
from multiprocessing import Queue, Manager
from pathos import multiprocessing
import traceback
import time
import sys
log_dir = sys.argv[1]
num_gpus = 2
max_worker_num = num_gpus * 2
nb_rollout_steps = 10 * 100
nb_train_steps = 100
meta_update_freq = 1
actor_update_freq = 1
batch_size = 1024
num_envs = 10
COMMAND = f"python3 experiments/run_hiro.py {log_dir} GoalTask --alg TD3 --evaluate --n_training 2 --total_steps 2500000 --verbose 1 --relative_goals --off_policy_corrections --eval_deterministic --num_envs {num_envs} --nb_rollout_steps {nb_rollout_steps} --actor_lr 3e-4 --critic_lr 3e-4 --use_huber --target_noise_clip 0.5 --batch_size {batch_size} --tau 0.05 --gamma 0.99 --meta_update_freq {meta_update_freq} --actor_update_freq {actor_update_freq} --intrinsic_reward_scale 1.0 --horizon 100"

meta_periods = (3, 5)
buffer_sizes = (500000, 2000000)
noises = (0.1, 0.3)
#nb_train_stepss = (100, 200, 400)
nb_train_stepss = (200, 400)

def _init_device_queue(max_worker_num):
    m = Manager()
    device_queue = m.Queue()
    for i in range(max_worker_num):
        idx = i % num_gpus
        device_queue.put(idx)
    return device_queue

def run():
    """Run trainings with all possible parameter combinations in
    the configured space.
    """

    process_pool = multiprocessing.Pool(
        processes=max_worker_num, maxtasksperchild=1)
    device_queue = _init_device_queue(max_worker_num)

    product = itertools.product(*(meta_periods, buffer_sizes, noises, nb_train_stepss))
    for task_count, values in enumerate(product):
        meta_period, buffer_size, noise, nb_train_steps = values
        command = "%s --meta_period %d --buffer_size %d --noise %0.2f --nb_train_steps %d" % (COMMAND, meta_period, buffer_size, noise, nb_train_steps)
        process_pool.apply_async(
            func=_worker,
            args=[command, device_queue],
            error_callback=lambda e: logging.error(e))

    process_pool.close()
    process_pool.join()

def _worker(command, device_queue):
    # sleep for random seconds to avoid crowded launching
    try:
        time.sleep(random.uniform(0, 60))

        device = device_queue.get()

        logging.set_verbosity(logging.INFO)

        logging.info("command %s" % command)
        os.system("CUDA_VISIBLE_DEVICES=%d " % device + command)

        device_queue.put(device)
    except Exception as e:
        logging.info(traceback.format_exc())
        raise e

run()
