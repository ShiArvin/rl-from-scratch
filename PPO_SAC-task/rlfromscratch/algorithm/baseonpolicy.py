from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
import torch 
from typing import Tuple


from logger.logger import Logger
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase, AtariDQNAgent
from utils.result import Result
from utils.utils import to_useful_action
from .basealgorithm import BaseAlgorithm
from utils.utils import RunningMeanStd


class OnPolicyAlgorithm(BaseAlgorithm, ABC):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(OnPolicyAlgorithm,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth,best_pth, args)

        self.collect_traj = args.algorithm.collect_traj
        if self.collect_traj:
            self.traj_rollout = buffer
            self.trajnum = buffer.trajnum 
            self.max_episode_length = buffer.max_episode_length
            self.last_states = np.zeros((self.trajnum, *self.observation_space.shape), dtype=self.observation_space.dtype)
        else:
            self.last_states = np.zeros((self.num_training_envs, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.gae = args.algorithm.gae
        self.lambda_ = args.algorithm.lambda_
        self.ret_rms = RunningMeanStd()
        self.return_scaling = args.algorithm.return_scaling
        self._eps = 1e-8
           
            
    
    def update(self, batch, start_train)-> Result:
        # For on-policy training, it is defaulted to do the training 
        if not self.collect_traj: # If collect the trajector, the update of the buffer is done in the collect_trajectories function
            with Result("buffer") as result:
                self._update_buffer(batch)
            for _ in range(self.update_step_per_epoch):
                update_policy_log = self._update_policy()
            result.add(update_policy_log)
            self.buffer.reset()
        else:
            for _ in range(self.update_step_per_epoch):
                result = self._update_policy()
            self.traj_rollout.reset() # for on-policy training, we will reset the replay buffer after each update
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

    def collect_trajectories(self):
        traj_states = [[] for _ in range(self.num_training_envs)]
        traj_actions = [[] for _ in range(self.num_training_envs)]
        traj_rewards = [[] for _ in range(self.num_training_envs)]
        traj_log_probs = [[] for _ in range(self.num_training_envs)]
        traj_dones = [[] for _ in range(self.num_training_envs)]

        with Result("interact") as result:
            while not self.traj_rollout.full:
                with torch.no_grad():
                    actions, action_infos = self.agent.select_action(self.observations, self.train_action_deterministic)
                    log_probs = action_infos['log_probs']
                    
                    next_observations, rewards, terminateds, infos = self.training_envs.step(to_useful_action(self.action_space, self.action_dim, actions))

                    traj_states = [traj_states[i]+[self.observations[i]] for i in range(self.num_training_envs)]
                    traj_actions = [traj_actions[i]+[actions[i]] for i in range(self.num_training_envs)]
                    traj_rewards = [traj_rewards[i]+[rewards[i]] for i in range(self.num_training_envs)]
                    traj_log_probs = [traj_log_probs[i]+[log_probs[i]] for i in range(self.num_training_envs)]
                    traj_dones = [traj_dones[i]+[terminateds[i]] for i in range(self.num_training_envs)]
                    # traj_truncateds = [traj_truncateds[i]+[infos[i].get("TimeLimit.truncated",False)] for i in range(self.num_training_envs)]

                    for i in range(self.num_training_envs):
                        if self.traj_rollout.full:
                            break
                        info = infos[i]
                        
                        
                        if "episode" in info:
                            traj_last_observations = next_observations[i]
                            truncated = False
                            if "truncated" in info:
                                traj_dones[i][-1] = False 
                                traj_last_observations = info['truncated_observation']
                                truncated = True

                            self.episode_reward_buffer.append(info['episode']['r'])
                            self.traj_rollout.add_traj(
                                dict(
                                    states=np.array(traj_states[i]),
                                    actions=np.array(traj_actions[i]),
                                    rewards=np.array(traj_rewards[i]),
                                    dones=np.array(traj_dones[i]),
                                    log_probs=np.array(traj_log_probs[i]),
                                    last_states=np.array(traj_last_observations),
                                    truncateds=np.array(truncated)
                                )
                            )
                            traj_states[i] = []
                            traj_actions[i] = []
                            traj_rewards[i] = []
                            traj_dones[i] = []
                            traj_log_probs[i] = []
                            
                            
                    self.observations = next_observations
                    self.interaction_step += self.num_training_envs 
        # reset the environment after each collection
        self.initialize()
        if len(self.episode_reward_buffer)>0:
            result.add_metric("reward_mean",np.mean(self.episode_reward_buffer))
            result.add_metric("reward_std",np.std(self.episode_reward_buffer))
        return None, result 



    def start_train(self):
        return True
    
    def interact_with_envs(self):
        if self.collect_traj:
            return self.collect_trajectories()
        else:
            return super().interact_with_envs()
    
    def _compute_unnormalized_advantages_trajectory(self, rewards, dones, values, last_values):
        delta = np.zeros((self.trajnum,),dtype=np.float32)
        last_advans = np.zeros((self.trajnum,),dtype=np.float32)
        last_values = (1-dones[:,-1])*last_values + np.zeros((self.trajnum,),dtype=np.float32)
        last_returns = last_values
        advantages = np.zeros((self.trajnum,self.max_episode_length),dtype=np.float32)
        returns = np.zeros((self.trajnum,self.max_episode_length),dtype=np.float32)

        for i in range(self.max_episode_length-1,-1,-1):
            delta = rewards[:,i] + self.gamma*(1-dones[:,i])*last_values - values[:,i]
            advantages[:,i] = delta + self.gamma*self.lambda_*(1-dones[:,i])*last_advans
            returns[:,i] = rewards[:,i] + (1-dones[:,i])*self.gamma*last_returns
            
            
            last_values = values[:,i]
            last_advans = advantages[:,i]
            last_returns = returns[:,i]
        if self.gae:
            return returns, advantages
        else:
            return returns, returns - values
    
    def compute_advantages_from_traj(self)->Tuple[np.ndarray,np.ndarray, np.ndarray]:
        states = self.traj_rollout.states
        rewards = self.traj_rollout.rewards 
        dones = self.traj_rollout.dones 
        last_states = self.traj_rollout.last_states
        

        with torch.no_grad():
            values = self.agent.get_value(states.reshape((-1,*states.shape[2:]))).squeeze(1).reshape((self.trajnum, self.max_episode_length)).cpu().numpy()
            last_values = self.agent.get_value(last_states).squeeze(1).cpu().numpy()
        
        values_to_cal = values * np.sqrt(self.ret_rms.var + self._eps) if self.return_scaling else values 
        last_values_to_cal = last_values * np.sqrt(self.ret_rms.var + self._eps) if self.return_scaling else last_values 

        unnormalized_returns, advantages = self._compute_unnormalized_advantages_trajectory(rewards, dones, values_to_cal, last_values_to_cal)
        
        if self.return_scaling:
            returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns.flatten())
        else:
            returns = unnormalized_returns
        return returns, advantages, values
        
    
    def _compute_unnormalized_advantages_rollout(self, rewards, dones, truncateds, values, next_values):
        num_envs, rollout_length = rewards.shape[0], rewards.shape[1]
        advantages = np.zeros((num_envs,rollout_length),dtype=np.float32)
        returns = np.zeros((num_envs,rollout_length),dtype=np.float32)
        if self.gae:
            deltas = rewards + self.gamma*(1-dones)*next_values - values
            last_advan = np.zeros((num_envs,),dtype=np.float32)
            advan_mask = np.logical_or(dones, truncateds)

            for i in range(rollout_length-1,-1,-1):
                advantages[:,i] = deltas[:, i] + self.gamma*self.lambda_*(1-advan_mask[:, i])*last_advan
                last_advan = advantages[:,i]
            returns = advantages + values
        
        else:
            last_returns = next_values[:,-1]
            for i in range(rollout_length-1,-1,-1):
                returns[:,i] = rewards[:,i] + self.gamma*(1-dones[:,i])*((1-truncateds[:,i])*last_returns+truncateds[:,i]*next_values[:,i])
                last_returns = returns[:,i]
            advantages = returns - values 
        return returns, advantages


    def compute_advantages_from_rollout(self)->Tuple[np.ndarray,np.ndarray]:
        states = self.buffer.buffer['states']
        rewards = self.buffer.buffer['rewards']
        dones = self.buffer.buffer['dones']
        truncateds = self.buffer.buffer['truncateds']
        next_states = self.buffer.buffer['next_states']
        
        num_envs, rollout_length = states.shape[0], states.shape[1]

        with torch.no_grad():
            values = self.agent.get_value(states.reshape((-1,*states.shape[2:]))).squeeze(1).reshape((num_envs, rollout_length)).cpu().numpy()
            next_values = self.agent.get_value(next_states.reshape((-1,*states.shape[2:]))).squeeze(1).reshape((num_envs, rollout_length)).cpu().numpy()
 

        value_to_cal = values * np.sqrt(self.ret_rms.var + self._eps) if self.return_scaling else values
        next_value_to_cal = next_values * np.sqrt(self.ret_rms.var + self._eps) if self.return_scaling else next_values
        unnormalized_returns, advantages = self._compute_unnormalized_advantages_rollout(rewards, 
                                                                                 dones, 
                                                                                 truncateds,
                                                                                 value_to_cal,
                                                                                 next_value_to_cal)
        

        if self.return_scaling:
            returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns.flatten())
        else:
            returns = unnormalized_returns
        return returns, advantages, values







