set -e

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"


for seed in 0 1 2; do 
  python run_ppo.py \
    algorithm=ppo \
    total_epoch=3000 \
    train_log_interval=10 \
    save_interval=1000 \
    test_interval=5 \
    train_action_deterministic=false \
    algorithm.collect_traj=false \
    env.name=Ant-v5 \
    algorithm.buffer_name=ReplayBuffer \
    interact_per_epoch=128 \
    algorithm.update_epochs=5 \
    algorithm.minibatch_size=128 \
    algorithm.use_grad_clip=true \
    algorithm.advan_norm=false \
    env.obs_norm=true \
    algorithm.rescale=false \
    algorithm.return_scaling=true \
    algorithm.lr_decay=true \
    seed=${seed}

  python run_ppo.py \
    algorithm=ppo \
    total_epoch=3000 \
    train_log_interval=10 \
    save_interval=1000 \
    test_interval=5 \
    train_action_deterministic=false \
    algorithm.collect_traj=false \
    env.name=HalfCheetah-v5 \
    algorithm.buffer_name=ReplayBuffer \
    interact_per_epoch=128 \
    algorithm.update_epochs=5 \
    algorithm.minibatch_size=256 \
    algorithm.use_grad_clip=true \
    algorithm.advan_norm=false \
    env.obs_norm=true \
    algorithm.rescale=true \
    algorithm.return_scaling=false \
    seed=${seed} \
    algorithm.initialize=true \

  python run_ppo.py \
    algorithm=ppo \
    total_epoch=1500 \
    train_log_interval=10 \
    save_interval=1000 \
    test_interval=5 \
    train_action_deterministic=false \
    algorithm.collect_traj=false \
    env.name=Hopper-v5 \
    algorithm.buffer_name=ReplayBuffer \
    interact_per_epoch=256 \
    algorithm.update_epochs=5 \
    algorithm.minibatch_size=256 \
    algorithm.use_grad_clip=true \
    algorithm.advan_norm=false \
    env.obs_norm=true \
    algorithm.rescale=false \
    algorithm.return_scaling=true \
    seed=${seed} \
    algorithm.lr_decay=true \

  python run_ppo.py \
    algorithm=ppo \
    total_epoch=1500 \
    train_log_interval=10 \
    save_interval=1000 \
    test_interval=5 \
    train_action_deterministic=false \
    algorithm.collect_traj=false \
    env.name=Walker2d-v5 \
    algorithm.buffer_name=ReplayBuffer \
    interact_per_epoch=256 \
    algorithm.update_epochs=5 \
    algorithm.minibatch_size=256 \
    algorithm.use_grad_clip=true \
    algorithm.advan_norm=false \
    env.obs_norm=true \
    algorithm.rescale=false \
    algorithm.return_scaling=true \
    seed=${seed} \
    algorithm.lr_decay=true \

  python run_ppo.py \
    algorithm=ppo \
    total_epoch=3000 \
    train_log_interval=10 \
    save_interval=1000 \
    test_interval=5 \
    train_action_deterministic=false \
    algorithm.collect_traj=false \
    env.name=Humanoid-v5 \
    algorithm.buffer_name=ReplayBuffer \
    interact_per_epoch=128 \
    algorithm.update_epochs=5 \
    algorithm.minibatch_size=128 \
    algorithm.use_grad_clip=true \
    algorithm.advan_norm=false \
    env.obs_norm=true \
    algorithm.rescale=false \
    algorithm.return_scaling=true \
    algorithm.lr_decay=true \
    seed=${seed}

done