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
num_gpus = 4
max_worker_num = num_gpus * 2
nb_rollout_steps = 38 * 50
nb_train_steps = 40
meta_update_freq = 1
actor_update_freq = 1
batch_size = 4864
num_envs = 12
COMMAND = f"python3 experiments/run_hiro.py {log_dir} FetchReach --alg TD3 --evaluate --n_training 1 --total_steps 4750000 --verbose 1 --relative_goals --off_policy_corrections --eval_deterministic --num_envs {num_envs} --nb_rollout_steps {nb_rollout_steps} --actor_lr 1e-3 --critic_lr 1e-3 --use_huber --target_noise_clip 0.5 --batch_size {batch_size} --tau 0.05 --gamma 0.98 --nb_train_steps {nb_train_steps} --meta_update_freq {meta_update_freq} --actor_update_freq {actor_update_freq} --intrinsic_reward_scale 1.0"

meta_periods = (3, 5)
buffer_sizes = (500000, 1500000)
noises = (0.1, 0.3)

#product = itertools.product(*(meta_periods, buffer_sizes, intrinsic_reward_scales, noises))
#for total, _ in enumerate(product):
#    continue
product = itertools.product(*(meta_periods, buffer_sizes, noises))
#for i, param_config in enumerate(product):
#    if i > int(total//2):
#        continue
#    meta_period, buffer_size, intrinsic_reward_scale, noise = param_config
#    command = "%s --meta_period %d --buffer_size %d --intrinsic_reward_scale %0.2f --noise %0.2f" % (COMMAND, meta_period, buffer_size, intrinsic_reward_scale, noise)
#    print(command + " &&")
    #os.system(command)
    #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    #output, error = process.communicate()
    #stream = os.popen(command)
    #print(stream.read())
    #print(output, error)

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

	product = itertools.product(*(meta_periods, buffer_sizes, noises))
	for task_count, values in enumerate(product):
		meta_period, buffer_size, noise = values
		command = "%s --meta_period %d --buffer_size %d --noise %0.2f" % (COMMAND, meta_period, buffer_size, noise)
		process_pool.apply_async(
			func=_worker,
			args=[command, device_queue],
			error_callback=lambda e: logging.error(e))

	process_pool.close()
	process_pool.join()

def _worker(command, device_queue):
	# sleep for random seconds to avoid crowded launching
	try:
		time.sleep(random.uniform(0, 3))

		device = device_queue.get()

		logging.set_verbosity(logging.INFO)

		logging.info("command %s" % command)
		os.system("CUDA_VISIBLE_DEVICES=%d " % device + command)

		device_queue.put(device)
	except Exception as e:
		logging.info(traceback.format_exc())
		raise e

run()
