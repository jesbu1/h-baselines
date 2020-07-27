import itertools
import subprocess
import os
nb_rollout_steps = 38 * 50
nb_train_steps = 40
meta_update_freq = 1
actor_update_freq = 1
batch_size = 4864
num_envs = 10
COMMAND = f"CUDA_VISIBLE_DEVICES=0 python experiments/run_hiro.py FetchPush --alg TD3 --evaluate --n_training 1 --total_steps 4750000 --verbose 1 --relative_goals --off_policy_corrections --eval_deterministic --num_envs {num_envs} --nb_rollout_steps {nb_rollout_steps} --actor_lr 1e-3 --critic_lr 3e-4 --use_huber --target_noise_clip 5.0 --batch_size {batch_size} --tau 0.05 --gamma 0.98 --nb_train_steps {nb_train_steps} --meta_update_freq {meta_update_freq} --actor_update_freq {actor_update_freq}"

meta_periods = (3, 5, 8)
buffer_sizes = (100000, 200000)
intrinsic_reward_scales = (0.1, 1)
noises = (0.5, 1.0, 2.0)

product = itertools.product(*(meta_periods, buffer_sizes, intrinsic_reward_scales, noises))
for total, _ in enumerate(product):
    continue
product = itertools.product(*(meta_periods, buffer_sizes, intrinsic_reward_scales, noises))
for i, param_config in enumerate(product):
    if i > int(total//2):
        continue
    meta_period, buffer_size, intrinsic_reward_scale, noise = param_config
    command = "%s --meta_period %d --buffer_size %d --intrinsic_reward_scale %0.2f --noise %0.2f" % (COMMAND, meta_period, buffer_size, intrinsic_reward_scale, noise)
    print(command + " &&")
    #os.system(command)
    #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    #output, error = process.communicate()
    #stream = os.popen(command)
    #print(stream.read())
    #print(output, error)
