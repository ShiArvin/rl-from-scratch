python run_trpo.py \
  total_epoch=5000 \
  train_log_interval=10 \
  save_interval=1000 \
  test_interval=50 \
  train_action_deterministic=false \
  algorithm.rescale=true \
  algorithm.collect_traj=true \
  env.name=HalfCheetah-v5