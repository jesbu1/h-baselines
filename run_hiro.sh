echo "python experiments/run_hiro.py AntPush --alg TD3 --evaluate --n_training 2 --total_steps 10000000 --verbose 1 --relative_goals --off_policy_corrections --eval_deterministic --num_envs 4 --nb_rollout_steps 4 --actor_lr 1e-4 --critic_lr 1e-3 --use_huber --target_noise_clip 5.0 --batch_size 100"
