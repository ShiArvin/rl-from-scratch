from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
import torch 


from logger.logger import Logger
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase, AtariDQNAgent
from utils.result import Result
from utils.utils import to_useful_action
from .basealgorithm import BaseAlgorithm


class OnPolicyAlgorithm(BaseAlgorithm, ABC):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(OnPolicyAlgorithm,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth,best_pth, args)

        self.collect_traj = args.algorithm.collect_traj
        if self.collect_traj:
            # Tips: if collect the trajectory, we will use the TrajectoryRollout as the replay buffer, and the update of the buffer is done in the collect_trajectories function
            self.traj_rollout = buffer
            self.trajnum = buffer.trajnum 
            self.max_episode_length = buffer.max_episode_length
           
            

    def update(self, batch, start_train)-> Result:
        # For on-policy training, it is defaulted to do the training 
        if not self.collect_traj: # If collect the trajector, the update of the buffer is done in the collect_trajectories function
            with Result("buffer") as result:
                self._update_buffer(batch)
            update_policy_log = self._update_policy()
            result.add(update_policy_log)
            self.buffer.reset()
        else:
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
        # traj_truncateds = [[] for _ in range(self.num_training_envs)]
        with Result("interact") as result:
            with torch.no_grad():
                trajectories=[
                    {
                        "states":[],
                        "actions":[],
                        "rewards":[],
                        "dones":[],
                        "log_probs":[],
                    }
                    for _ in range(self.num_training_envs)
                ]
                while not self.traj_rollout.full: #停止条件是收集满trajnum=10条trajectories，所以每个epoch收集transitions数量不确定，与rollout方法不同，但trajectory也会有并行环境
                    actions,action_infos=self.agent.select_action(self.observations,self.train_action_deterministic) #actions返回每个环境要执行的动作，action_infos返回当前策略下这些动作的log probability
                    next_observations,rewards,terminateds,infos=(
                        self.training_envs.step(           #.step(actions)所有并行环境各执行一步，返回....
                            to_useful_action(
                                self.action_space,
                                self.action_dim,
                                actions,
                            )
                        )
                    )
                    for i in range(self.num_training_envs):
                        if "episode" in infos[i]:
                            self.episode_reward_buffer.append(infos[i]["episode"]["r"])
                    for i in range(self.num_training_envs):
                        trajectories[i]["states"].append(self.observations[i])
                        trajectories[i]["actions"].append(actions[i])
                        trajectories[i]["rewards"].append(rewards[i])
                        trajectories[i]["dones"].append(terminateds[i])
                        trajectories[i]["log_probs"].append(action_infos["log_probs"][i])
                        if terminateds[i]:
                            traj_batch={
                                key: np.asarray(value)
                                for key,value in trajectories[i].items()
                            }
                            self.traj_rollout.add_traj(traj_batch) #最终trajectory模式取数据就从traj_rollout
                            trajectories[i]["states"]=[]
                            trajectories[i]["actions"]=[]
                            trajectories[i]["rewards"]=[]
                            trajectories[i]["dones"]=[]
                            trajectories[i]["log_probs"]=[]
                            if self.traj_rollout.full:
                                break
                    self.observations=next_observations
                    
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
            return self.collect_trajectories() #按trajectory收集
        else:
            return super().interact_with_envs() #回到basealgorithm.py中调用interact_with_envs(),按固定步数rollout收集






