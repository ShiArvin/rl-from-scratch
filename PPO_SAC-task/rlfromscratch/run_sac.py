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
from algorithm import OffPolicyAlgorithm, ALGORITHM_DICT, SAC
from utils.utils import get_best_device, set_seed
from logger.logger import Logger 
from memory import BUFFER_DICT,ReplayBuffer, TrajectoryRollout
from agent import A2CAgent, SACAgent
from utils.networks import QFunction, MLPNetwork, DoubleQFunction, DiagGaussianActor, TanhGaussianActor
from utils import ACTIVATION_DICT

def get_args(cfg: DictConfig):
    
    cfg.hydra_base_dir = os.getcwd()
    print("Parameters:", OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="config", config_name="config",version_base="1.3")
def main(cfg: DictConfig):
    args = get_args(cfg)
    set_seed(args.seed)
    
    training_envs = make_vec_envs(args.env,True,seed=args.seed) # , make_vec_envs(args.env,False,scale=False)
    
    logging.info("Checking for available GPUs...")
    device = get_best_device()

    logging.info("Creating the Logger...")
    runtime = datetime.now()
    runtime = runtime.strftime("%Y-%m-%d %H:%M:%S")
    logger = Logger(project_name=args.experiment_name, run_name=runtime, config=args.algorithm, log_dir=args.log_dir, use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard, use_swanlab=args.use_swanlab)

    logging.info("Creating the ReplayBuffer...")
    
    buffer = ReplayBuffer(training_envs.observation_space, training_envs.action_space, args.algorithm.buffer_size//training_envs.num_envs, training_envs.num_envs, onpolicy=args.algorithm.onpolicy)

    logging.info("Creating the Agent...")

    activation = ACTIVATION_DICT[args.algorithm.activation]
    if not args.algorithm.rescale:
        actor = DiagGaussianActor(
            training_envs.observation_space, 
            training_envs.action_space, 
            MLPNetwork, 
            device=device,
            state_dependent_std=args.algorithm.state_dependent_std,
            hidden_sizes=args.algorithm.hidden_sizes, 
            activation=torch.nn.Tanh
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
                activation=torch.nn.Tanh
            )
        else:
            raise ValueError(f"Action bound method {args.algorithm.action_bound_method} not supported")

    double_critic = args.algorithm.double_critic
    target_critic = None
    if double_critic:
        critic = DoubleQFunction(training_envs.observation_space, training_envs.action_space, MLPNetwork, hidden_sizes=args.algorithm.hidden_sizes, activation=activation)
        target_critic = DoubleQFunction(training_envs.observation_space, training_envs.action_space, MLPNetwork, hidden_sizes=args.algorithm.hidden_sizes, activation=activation) if args.algorithm.use_target else None

    else:
        critic = QFunction(training_envs.observation_space, training_envs.action_space, MLPNetwork, hidden_sizes=args.algorithm.hidden_sizes, activation=activation)
        target_critic = QFunction(training_envs.observation_space, training_envs.action_space, MLPNetwork, hidden_sizes=args.algorithm.hidden_sizes, activation=activation) if args.algorithm.use_target else None

    agent = SACAgent(
        training_envs.observation_space, 
        training_envs.action_space, 
        device, 
        actor, 
        critic,
        target_critic=target_critic,
        use_target=args.algorithm.use_target, 
        target_update_method=args.algorithm.target_update_method,
        target_update_tau=args.algorithm.target_update_tau,
        double_critic=double_critic)

    
    # create trainer 
    algorithm: OffPolicyAlgorithm | SAC
    
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