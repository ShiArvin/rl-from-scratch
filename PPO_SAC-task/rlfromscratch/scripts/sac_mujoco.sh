#!/usr/bin/env bash
set -e

cd /home/ubuntu/wangchenyang/RLfromscratch

source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate /home/ubuntu/wangchenyang/anaconda/envs/sac

export PYTHONUNBUFFERED=1

seeds=(0 1 2)
env_settings=(
  "Hopper-v5 0.2"
  "Walker2d-v5 0.2"
  "HalfCheetah-v5 0.2"
  "Ant-v5 0.2"
  "Humanoid-v5 0.05"
)

for seed in "${seeds[@]}"; do
  for setting in "${env_settings[@]}"; do
    read -r env_name init_temp <<< "$setting"
    experiment_name="SAC-lr3e-4-${env_name}-seed-${seed}"

    echo "Running ${experiment_name}  init_temp=${init_temp}"
    python run_sac.py \
      algorithm=sac \
      train_action_deterministic=false \
      env=mujoco \
      env.name="${env_name}" \
      env.num_training_envs=1 \
      seed="${seed}" \
      experiment_name="${experiment_name}" \
      use_swanlab=true \
      use_tensorboard=true \
      save_interval=100000 \
      test_interval=10000 \
      train_log_interval=10000 \
      total_epoch=1500000 \
      algorithm.buffer_size=1000000 \
      algorithm.learn_temp=false \
      algorithm.init_temp="${init_temp}" \
      algorithm.actor_lr=3e-4 \
      algorithm.critic_lr=3e-4 \
      interact_per_epoch=1 \
      "$@"
  done
done
