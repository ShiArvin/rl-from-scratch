from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
import torch 


from logger.logger import Logger
from memory.memory import ReplayBuffer, TrajectoryRollout
from utils.evaluate import evaluate
from agent.agent import AgentBase
from utils.result import Result
from utils.utils import to_useful_action, get_action_dim
from env.env_types import ENVS_NAME_TYPE


class BaseAlgorithm(ABC):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str,best_pth:str, args):
        super(BaseAlgorithm,self).__init__()
        self.training_envs = training_envs
        self.testing_envs = testing_envs
        self.buffer = buffer
        self.agent = agent.to(device)
        self.device = device
        self.logger = logger
        self.observation_space = training_envs.observation_space
        self.action_space = training_envs.action_space
        self.action_dim = get_action_dim(self.action_space)
        self.seed = args.seed 


        self.interaction_step = 0
        self.gradient_step = 0

        self.total_epoch = args.total_epoch
        self.num_training_envs = args.env.num_training_envs
        self.num_testing_envs = args.env.num_testing_envs
        self.test_interval = args.test_interval
        self.interact_per_epoch = args.interact_per_epoch
        self.update_step_per_epoch = args.update_step_per_epoch
        self.total_update_steps = self.update_step_per_epoch*self.total_epoch
        self.test_episodes = args.test_episodes
        self.train_action_deterministic = args.train_action_deterministic
        self.save_interval = args.save_interval
        self.train_log_interval = args.train_log_interval
        
        self.save_pth = save_pth
        self.args = args
        self.episode_reward_buffer = deque(maxlen=args.reward_buffer_size)

        self.best_score = -np.inf
        self.best_pth = best_pth
        self.env_name = args.env.name
        self.env_type = ENVS_NAME_TYPE[self.env_name]
        self.gamma = args.gamma 

        self.onpolicy=args.algorithm.onpolicy
        self.test_epsilon = args.algorithm.test_epsilon

    
    @abstractmethod
    def update(self, batch: dict, start_train: bool)-> Result:
        """
        update the policy and the buffer 
        """
        

    def random_choose_action(self):
        """You can use this function to implement epsilon-greedy exploration strategy"""
        return False

    def interact_with_envs(self):
        """
        interact with the environments and collect data, then return the collected batch of data
        """
        interact_steps_per_env = self.interact_per_epoch
        batch = dict(states=np.zeros(shape=(self.num_training_envs, interact_steps_per_env, *self.observation_space.shape),dtype=self.observation_space.dtype),
                    actions=np.zeros(shape=(self.num_training_envs, interact_steps_per_env, self.action_dim),dtype=self.action_space.dtype),
                    rewards=np.zeros(shape=(self.num_training_envs, interact_steps_per_env),dtype=np.float32),
                    next_states=np.zeros(shape=(self.num_training_envs, interact_steps_per_env, *self.observation_space.shape),dtype=self.observation_space.dtype),
                    dones=np.zeros(shape=(self.num_training_envs, interact_steps_per_env),dtype=np.float32),
                    truncateds=np.zeros(shape=(self.num_training_envs, interact_steps_per_env),dtype=np.float32)) 

        if self.onpolicy:
            batch['log_probs'] = np.zeros(shape=(self.num_training_envs, interact_steps_per_env),dtype=np.float32)
        with Result("interact") as result:
            for step in range(interact_steps_per_env):
                
                if self.random_choose_action():
                    actions = np.array([[self.training_envs.action_space.sample()] for _ in range(self.num_training_envs)],dtype=self.training_envs.action_space.dtype)
                    
                else:
                    with torch.no_grad():
                        actions, action_infos = self.agent.select_action(self.observations, self.train_action_deterministic)
                next_observations, rewards, terminateds, infos = self.training_envs.step(to_useful_action(self.action_space, self.action_dim, actions))
                

                batch['states'][:,step] = self.observations
                batch['actions'][:,step] = actions
                batch['rewards'][:,step] = rewards
                batch['next_states'][:,step] = next_observations 
                batch['dones'][:,step] = terminateds
                if self.onpolicy:
                    batch['log_probs'][:,step] = action_infos['log_probs']

                for i in range(self.num_training_envs):
                    info = infos[i]
                    if "truncated" in info:
                        # If the terminated is caused by truncation instead of termination, we consider it as not done and use the last observation as the next state for training.
                        batch['dones'][i,step] = False
                        batch['truncateds'][i,step] = True
                        batch['next_states'][i, step] = info['truncated_observation']
                        
                    if "episode" in info:
                        self.episode_reward_buffer.append(info['episode']['r'])
                self.observations = next_observations
                self.interaction_step += self.num_training_envs
        

        if len(self.episode_reward_buffer)>0:
            result.add_metric("reward_mean",np.mean(self.episode_reward_buffer))
            result.add_metric("reward_std",np.std(self.episode_reward_buffer))
        return batch, result
           
    def test_condition(self,epoch):
        """
        check if it's time to test the policy
        """
        return epoch % self.test_interval == 0

    def save_condition(self,epoch):
        """check if it's time to save the checkpoint"""
        return epoch % self.save_interval == 0
            

    def initialize(self):
        self.observations = self.training_envs.reset()
    
    def train_log_condition(self,epoch):
        return epoch%self.train_log_interval==0
        
    def start_train(self):
        return True
    
    def sace(self):
        pass

    def run(self):
        """
        run the training loop of the algorithm
        """
        
        self.initialize() # reset envs
        for epoch in tqdm(range(self.total_epoch), desc="Epoch", unit="epoch"):
            if self.test_condition(epoch):
                test_result, test_reward = evaluate(agent=self.agent, test_episodes=self.test_episodes,env_args=self.args.env, seed=self.seed+self.interaction_step,training_envs=self.training_envs, test_epsilon=self.test_epsilon)
                self.logger.log_test(epoch, self.interaction_step, self.gradient_step, test_result)
                if test_reward > self.best_score:
                    self.best_score = test_reward
                    self.agent.save(self.best_pth)
            
            collected_batch, interact_result = self.interact_with_envs()
            train_result = self.update(collected_batch, self.start_train())
            if self.train_log_condition(epoch):
                train_result.add(interact_result)
                self.logger.log_train(epoch, self.interaction_step, self.gradient_step, train_result)
                               
                              
            if self.save_condition(epoch):
                self.agent.save(self.save_pth)




