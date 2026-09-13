import os 
import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] ---------------%(message)s---------------',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import hydra 
from omegaconf import DictConfig, OmegaConf
import torch 
from datetime import datetime 
import numpy as np

from env.make_envs import make_vec_envs
from algorithm import TRPO, OnPolicyAlgorithm, ALGORITHM_DICT, PPO
from utils.utils import get_best_device, set_seed
from logger.logger import Logger 
from memory import BUFFER_DICT,ReplayBuffer, TrajectoryRollout
from agent import ProbabilityA2CAgent
from utils.networks import DiagGaussianActor, TanhGaussianActor, ValueFunction, MLPNetwork


def get_args(cfg: DictConfig):
    
    cfg.hydra_base_dir = os.getcwd() #get current working directory
    print("Parameters:", OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="config", config_name="config",version_base="1.3") #Hydra装饰器读取yaml配置并创建一个DictConfig对象并传参进入main()
def main(cfg: DictConfig):
    args = get_args(cfg)
    set_seed(args.seed)
    
    training_envs = make_vec_envs(args.env,True,seed=args.seed) # 创建并行训练环境
    
    logging.info("Checking for available GPUs...")
    device = get_best_device()

    logging.info("Creating the Logger...")
    runtime = datetime.now()
    runtime = runtime.strftime("%Y-%m-%d %H:%M:%S")
    logger = Logger(project_name=args.experiment_name, run_name=runtime, config=args.algorithm, log_dir=args.log_dir, use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard, use_swanlab=args.use_swanlab)

    logging.info("Creating the ReplayBuffer...")
    if args.algorithm.buffer_name=="TrajectoryRollout":
        buffer = TrajectoryRollout(training_envs.observation_space,training_envs.action_space, args.algorithm.trajnum, args.env.max_episode_length)
    else:
        buffer = ReplayBuffer(training_envs.observation_space, training_envs.action_space, args.interact_per_epoch, training_envs.num_envs,onpolicy=args.algorithm.onpolicy)

    logging.info("Creating the Agent...")
    if not args.algorithm.rescale:                         #创建actor
        actor = DiagGaussianActor(
            training_envs.observation_space, 
            training_envs.action_space, 
            MLPNetwork, 
            device=device,
            state_dependent_std=args.algorithm.state_dependent_std,
            hidden_sizes=args.algorithm.hidden_sizes, 
            activation=torch.nn.Tanh,
            initialize=args.algorithm.initialize
        )
    else:
        if args.algorithm.action_bound_method == "tanh":
            actor = TanhGaussianActor(
                training_envs.observation_space, 
                training_envs.action_space, 
                MLPNetwork, 
                device=device,
                state_dependent_std=args.algorithm.state_dependent_std,
                hidden_sizes=args.algorithm.hidden_sizes, 
                activation=torch.nn.Tanh,
                initialize=args.algorithm.initialize
            )
        else:
            raise ValueError(f"Action bound method {args.algorithm.action_bound_method} not supported")


    critic = ValueFunction(training_envs.observation_space, 
                           MLPNetwork, 
                           hidden_sizes=args.algorithm.hidden_sizes, 
                           activation=torch.nn.Tanh,
                           initialize=args.algorithm.initialize)

    agent = ProbabilityA2CAgent(training_envs.observation_space, training_envs.action_space, device, actor, critic)
    
        

    # create trainer 
    algorithm: OnPolicyAlgorithm | PPO
    
    algorithm = ALGORITHM_DICT[args.algorithm.name](training_envs=training_envs, testing_envs=None, 
                                               buffer=buffer, agent=agent, 
                                               logger=logger, device=device, 
                                               save_pth=os.path.join(args.log_dir, "newest_model.pth"), 
                                               best_pth=os.path.join(args.log_dir, "best_model.pth"),
                                               args=args)

    logging.info("Begin Training...")
    algorithm.run()
    

if __name__ == "__main__":
    main()