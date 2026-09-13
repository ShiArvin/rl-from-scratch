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

from env.make_envs import make_vec_envs
from algorithm.offpolicy_algorithm.baseoffpolicy import OffPolicyAlgorithm
from algorithm.offpolicy_algorithm import DQN
from algorithm.offpolicy_algorithm import ALGORITHM_DICT
from utils.utils import get_best_device, set_seed
from logger.logger import Logger 
from memory.memory import ReplayBuffer
from memory import BUFFER_DICT
from agent.atari_agent import AtariDQNAgent


def get_args(cfg: DictConfig):
    
    cfg.hydra_base_dir = os.getcwd()
    print("Parameters:", OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="config", config_name="config",version_base="1.3")
def main(cfg: DictConfig):
    args = get_args(cfg)
    set_seed(args.seed)
    
    training_envs = make_vec_envs(args.env,True,scale=False) # , make_vec_envs(args.env,False,scale=False)
    
    logging.info("Checking for available GPUs...")
    device = get_best_device()

    logging.info("Creating the Logger...")
    runtime = datetime.now()
    runtime = runtime.strftime("%Y-%m-%d %H:%M:%S")
    logger = Logger(project_name=args.experiment_name, run_name=runtime, log_dir=args.log_dir, use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard)

    logging.info("Creating the ReplayBuffer...")
    buffer = BUFFER_DICT[args.algorithm.buffer_name](training_envs.observation_space, training_envs.action_space, args.algorithm.buffer_size, training_envs.num_envs)

    logging.info("Creating the Agent...")
    agent = AtariDQNAgent(training_envs.observation_space, training_envs.action_space, device)
    target_agent = AtariDQNAgent(training_envs.observation_space, training_envs.action_space, device) if args.algorithm.use_target else None
    
        

    # create trainer 
    algorithm: OffPolicyAlgorithm | DQN
    
    algorithm = ALGORITHM_DICT[args.algorithm.name](training_envs=training_envs, testing_envs=None, 
                                               buffer=buffer, agent=agent, 
                                               logger=logger, device=device, 
                                               save_pth=os.path.join(args.log_dir, "newest_model.pth"), 
                                               args=args,target_agent=target_agent)

    logging.info("Begin Training...")
    algorithm.run()
    

if __name__ == "__main__":
    main()