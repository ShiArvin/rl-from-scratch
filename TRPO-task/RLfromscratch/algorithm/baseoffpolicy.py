from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
import torch 


from logger.logger import Logger
from memory.memory import ReplayBuffer
from agent.agent import AgentBase, AtariDQNAgent
from utils.result import Result
from . import BaseAlgorithm

class OffPolicyAlgorithm(BaseAlgorithm, ABC):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer, agent: AgentBase, logger: Logger, device,  save_pth: str,best_pth:str, args):
        super(OffPolicyAlgorithm,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth,best_pth, args)
        

    
    
    def update(self, batch, start_train)-> Result:
        with Result("buffer") as result:
            self._update_buffer(batch)
        if start_train:
            update_policy_log = self._update_policy()
            result.add(update_policy_log)
            return result
        return result
        
    
    @abstractmethod
    def _update_buffer(self, batch):
        """
        update the replay buffer with given selected batch of data
        :param batch: the batch of data
        """
    
    @abstractmethod
    def _update_policy(self):
        """
        do gradient update to the policy with a batch of data sampled from replay buffer
        """

        

    def start_train(self):
        return self.interaction_step>=5000






