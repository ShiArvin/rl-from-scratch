from .baseoffpolicy import OffPolicyAlgorithm
from memory.memory import ReplayBuffer
import torch 
from utils.result import Result
from logger.logger import Logger
import numpy as np
from agent import AgentBase, A2CAgent, TD3Agent
from utils import OPTIMIZER_DICT
import math


class TD3(OffPolicyAlgorithm):
    
    """DQN algorithm implementation."""

    def __init__(self, training_envs, testing_envs, buffer: ReplayBuffer, agent: TD3Agent, logger: Logger, device, save_pth: str,best_pth: str, args):
        super(TD3, self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth,best_pth, args)
        
        algo_args = args.algorithm

        self.critic_lr = algo_args.critic_lr
        self.actor_lr = algo_args.actor_lr
        
       
        self.batch_size = algo_args.batch_size
        

        self.agent = agent

        self.use_target = algo_args.use_target 
        self.target_update_interval = algo_args.target_update_interval

        
        self.critic_optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.critic_optimizer](self.agent.critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.actor_optimizer](self.agent.actor.parameters(), lr=self.actor_lr)

        self.noise_sigma = algo_args.noise_sigma
        self.noise_clip = algo_args.noise_clip


    def _update_buffer(self, batch):
        self.buffer.add(batch)
    
    def random_choose_action(self):
        return self.interaction_step<self.start_train_step


    def _update_policy(self):
        self.agent.train()
        with Result("train") as result:
            batch = self.buffer.sample(self.batch_size)
            states, actions, next_states, rewards, dones = batch['states'], batch['actions'], batch['next_states'], batch['rewards'], batch['dones']

            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device).unsqueeze(1)
            dones = torch.from_numpy(dones).float().to(self.device).unsqueeze(1)
            nstep_gamma = torch.from_numpy(batch['nstep_gamma']).float().to(self.device).unsqueeze(1)

            # Compute the target value
            with torch.no_grad():
                noise = (torch.randn_like(actions) * self.noise_sigma).clamp(-self.noise_clip, self.noise_clip)
                next_actions = self.agent.select_action_from_target(next_states,noise)
                target = rewards + (1-dones) * nstep_gamma * self.agent.get_q_function_from_target(next_states,next_actions)
                

            q1, q2 = self.agent.get_double_q_function(states, actions)
            
            td_error1 = target - q1
            td_error2 = target - q2
            

            q_loss = torch.mean(td_error1**2) + torch.mean(td_error2**2)
            
            # do gradient update to the critic
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()

            if self.gradient_step%self.target_update_interval==0:
                
                q1_new, _ = self.agent.get_double_q_function(states,self.agent.get_action_with_gradient(states))

                actor_loss = -q1_new.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.agent.update_target()

                result.add_metric("actor/loss", actor_loss.item())

        result.add_metric("critic1/q",q1.mean().item())
        result.add_metric("critic2/q",q2.mean().item())
        result.add_metric("critic1/td_error_abs", torch.abs(td_error1).mean().item())
        result.add_metric("critic2/td_error_abs", torch.abs(td_error2).mean().item())
        result.add_metric("critic/q_loss", q_loss.item())
        
        
        self.gradient_step += 1
        return result

