from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym


from logger.logger import Logger
from memory.memory import ReplayBuffer
from utils.evaluate import evaluate_atari
from agent.atari_agent import AgentBase, AtariDQNAgent
from utils.result import Result
from utils.utils import atari_to_useful_action


class OffPolicyAlgorithm(ABC):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer, agent: AgentBase, logger: Logger, device,  save_pth: str, args):
        super(OffPolicyAlgorithm,self).__init__()
        self.training_envs = training_envs
        self.testing_envs = testing_envs
        self.buffer = buffer
        self.agent = agent.to(device)
        self.device = device
        self.logger = logger
        


        self.interaction_step = 0
        self.gradient_step = 0

        self.total_epoch = args.total_epoch
        self.num_training_envs = args.env.num_training_envs
        self.num_testing_envs = args.env.num_testing_envs
        self.test_interval = args.test_interval
        self.interact_per_epoch = args.interact_per_epoch
        self.test_episodes = args.test_episodes
        self.train_action_deterministic = args.train_action_deterministic
        self.save_interval = args.save_interval
        self.train_log_interval = args.train_log_interval
        
        self.save_pth = save_pth
        self.args = args
        self.episode_reward_buffer = deque(maxlen=args.reward_buffer_size)
        self.envs_rewards = np.zeros((self.num_training_envs,))

    
    
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

    def random_choose_action():
        """You can use this function to implement epsilon-greedy exploration strategy"""
        return False

    

    def interact_with_envs(self):
        """
        interact with the environments and collect data, then return the collected batch of data
        """
        interact_steps_per_env = self.interact_per_epoch
        batch = dict(states=[], actions=[], rewards=[], next_states=[], dones=[]) 
        with Result("interact") as result:
            for step in range(interact_steps_per_env):
                
                if self.random_choose_action():
                    actions=np.array(
                        [
                            [self.training_envs.action_space.sample()]
                            for _ in range(self.num_training_envs)
                        ]
                    )
                else:
                    actions=self.agent.select_action(self.observations,deterministic=True)
                next_observations,rewards,dones,infos=self.training_envs.step(actions.squeeze(1))
                batch["states"].append(self.observations)
                batch["actions"].append(actions)
                batch["rewards"].append(rewards)
                batch["next_states"].append(next_observations)
                batch["dones"].append(dones)
                self.observations=next_observations
                self.interaction_step+=self.num_training_envs

                for i in range(self.num_training_envs):
                    info = infos[i]
                    if "episode" in info:
                        self.episode_reward_buffer.append(info['episode']['r'])
                        # self.envs_rewards[i] = 0
                
        
        for k,v in batch.items():
            # if k=="infos":
            #     continue
            batch[k] = np.stack(v,axis=0)

        if len(self.episode_reward_buffer)>0:
            result.add_metric("reward_mean",np.mean(self.episode_reward_buffer))
            result.add_metric("reward_std",np.std(self.episode_reward_buffer))
        return batch, result
           
    def test_condition(self):
        """
        check if it's time to test the policy
        """
        return self.interaction_step % self.test_interval == 0

    def save_condition(self):
        """check if it's time to save the checkpoint"""
        if self.interaction_step % self.save_interval == 0:
            return True 
        return False

    def initialize(self):
        self.observations = self.training_envs.reset()
    
    def train_log_condition(self):
        if self.interaction_step%self.train_log_interval==0:
            return True 
        return False

    def start_train(self):
        return self.interaction_step>=5000

    def run(self):
        """
        run the training loop of the algorithm, this is a typical procedure of offpolicy algorithm
        """
        
        self.initialize() # reset envs
        for epoch in tqdm(range(self.total_epoch), desc="Epoch", unit="epoch"):
            if self.test_condition():
                test_result = evaluate_atari(self.agent, self.test_episodes,self.args.env)
                self.logger.log_test(epoch, self.interaction_step, self.gradient_step, test_result)
            
            collected_batch, interact_result = self.interact_with_envs()
            train_result = self.update(collected_batch, self.start_train())
            if self.train_log_condition():
                train_result.add(interact_result)
                self.logger.log_train(epoch, self.interaction_step, self.gradient_step, train_result)
                
            
            if self.save_condition():
                self.agent.save(self.save_pth)




