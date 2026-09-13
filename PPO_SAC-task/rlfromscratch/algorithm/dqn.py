from .baseoffpolicy import OffPolicyAlgorithm
from memory.memory import ReplayBuffer
import torch 
from utils.result import Result
from logger.logger import Logger
import numpy as np
from agent.agent import AtariDQNAgent
from utils import OPTIMIZER_DICT




class DQN(OffPolicyAlgorithm):
    
    """DQN algorithm implementation."""

    def __init__(self, training_envs, testing_envs, buffer: ReplayBuffer, agent: AtariDQNAgent, logger: Logger, device, save_pth: str,best_pth: str, args):
        super(DQN, self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth,best_pth, args)
        
        algo_args = args.algorithm
        assert algo_args.name=="DQN", "The method name in args must be 'dqn' for DQN algorithm."
        self.lr = algo_args.learning_rate
        self.start_epsilon = algo_args.start_epsilon
        self.epsilon = algo_args.start_epsilon
        self.end_epsilon = algo_args.end_epsilon
        self.epsilon_timestep = algo_args.epsilon_timestep
        self.epsilon_schedular = algo_args.epsilon_schedular
        
        self.batch_size = algo_args.batch_size
        self.device = device
        
        
        self.use_target = algo_args.use_target 
        self.target_update_interval = algo_args.target_update_interval

        self.dueling = algo_args.dueling_network
        self.use_grad_clip = algo_args.use_grad_clip
        

        
        if algo_args.optimizer == "RMSprop":
            self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.network.parameters(), lr=self.lr, alpha=0.95, eps=0.01)
        else:
            self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.network.parameters(), lr=self.lr)

        self.doubledqn = algo_args.doubledqn
        self.huber_loss = algo_args.huber_loss
        
        self.alpha = algo_args.alpha 
        self.beta = algo_args.beta
        self.beta_increment = (1-self.beta)/self.total_update_steps # here we use the default target 1 set in prioritized replay buffer paper
        self.buffer_name = algo_args.buffer_name

        self.nstep = algo_args.nstep
        
    


    def _update_buffer(self, batch):
        self.buffer.add(batch)


    def _update_policy(self):
        self.agent.train()
        with Result("train") as result:
            batch = self.buffer.sample(self.batch_size)
            states, actions, next_states, rewards, dones = batch['states'], batch['actions'], batch['next_states'], batch['rewards'], batch['dones']
            
            states = torch.from_numpy(states).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            next_states = torch.from_numpy(next_states).to(self.device)

            rewards = torch.from_numpy(rewards).float().to(self.device).unsqueeze(1)
            dones = torch.from_numpy(dones).float().to(self.device).unsqueeze(1)
            nstep_gamma = torch.from_numpy(batch['nstep_gamma']).float().to(self.device).unsqueeze(1)


            if self.doubledqn:
                # calculate the DDQN target
                with torch.no_grad():
                    targetv = self.agent.get_ddqn_target(next_states)
                    target = rewards + (1 - dones) * nstep_gamma * targetv

            else:
                # calculate the DQN target
                with torch.no_grad():
                    if self.use_target:
                        target = rewards + (1 - dones) * nstep_gamma * self.agent.get_max_q(next_states,target=True)
                    else:
                        target = rewards + (1 - dones) * nstep_gamma * self.agent.get_max_q(next_states,target=False)
            q = self.agent.get_q(states, actions)
            
            td_error = target - q


            if self.huber_loss:
                loss = torch.nn.functional.huber_loss(q, target, delta=1, reduction="none")
            else:
                loss = td_error**2
            
            if "weights" in batch:
                weights = torch.tensor(batch['weights'], device=self.device).unsqueeze(1)
                loss = loss*weights 
            
            loss = torch.mean(loss)
            
            # do gradient update to the agent 
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), max_norm=10.)

            self.optimizer.step()

            if self.buffer_name=="PrioritizedReplayBuffer":
                self.buffer.update_priority(batch['indices'], td_error.squeeze(1).detach().cpu().numpy())
                self.beta = min(1, self.beta + self.beta_increment)
                self.buffer.beta = self.beta 
                


            if self.use_target and self.gradient_step % self.target_update_interval == 0:
                self.agent.target_update()
            


        result.add_metric("loss", loss.item())
        result.add_metric("td_error", td_error.mean().item())
        result.add_metric("q_value_mean", q.mean().item())

        if self.buffer_name=="PrioritizedReplayBuffer":
            result.add_metric("beta", self.beta)
            result.add_metric("weight_mean", np.mean(batch['weights']))
       

        self.gradient_step += 1
        return result

    def random_choose_action(self):
        """You can use this function to implement epsilon-greedy exploration strategy"""
        if self.epsilon_schedular=="linear":
            self.epsilon = max(self.end_epsilon, self.start_epsilon - self.interaction_step / self.epsilon_timestep * (self.start_epsilon - self.end_epsilon))
        else:
            self.epsilon = self.start_epsilon
        
        return np.random.rand() < self.epsilon

    def interact_with_envs(self):
        batch, result = super().interact_with_envs()
        result.add_metric("epsilon",self.epsilon)
        return batch, result