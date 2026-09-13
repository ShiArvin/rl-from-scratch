from .baseoffpolicy import OffPolicyAlgorithm
from memory.memory import ReplayBuffer
import torch 
from logger.logger import Logger
from agent import SACAgent
from utils import OPTIMIZER_DICT
import math
from utils.result import Result

class SAC(OffPolicyAlgorithm):
    
    """DQN algorithm implementation."""

    def __init__(self, training_envs, testing_envs, buffer: ReplayBuffer, agent: SACAgent, logger: Logger, device, save_pth: str,best_pth: str, args):
        super(SAC, self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth,best_pth, args)
        
        algo_args = args.algorithm

        self.critic_lr = algo_args.critic_lr
        self.actor_lr = algo_args.actor_lr
        self.temp_lr = algo_args.temp_lr
        
       
        self.batch_size = algo_args.batch_size
        

        self.agent = agent

        self.use_target = algo_args.use_target 
        self.target_update_interval = algo_args.target_update_interval

        
        self.critic_optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.critic_optimizer](self.agent.critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.actor_optimizer](self.agent.actor.parameters(), lr=self.actor_lr)

        self.init_temp = algo_args.init_temp
        self.log_temp = torch.nn.Parameter(torch.tensor(math.log(self.init_temp),device=self.device,dtype=torch.float32))
        self.learn_temp = algo_args.learn_temp
        if self.learn_temp:
            self.temp_optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.temp_optimizer]([self.log_temp], lr=self.temp_lr)
        self.update_log_temp = algo_args.update_log_temp
        
        self.target_entropy = -self.action_dim

    def _update_buffer(self, batch):
        self.buffer.add(batch)
    
    def random_choose_action(self):
        return self.interaction_step<self.start_train_step


    def _update_policy(self):
        # TODO(SAC-1): Set the agent to train mode and create Result("train") for metrics.
        self.agent.train()
        with Result("train") as result:
        # TODO(SAC-2): Sample a batch from replay buffer and move rewards/dones/nstep_gamma to self.device.
            batch=self.buffer.sample(self.batch_size)
            states=torch.from_numpy(batch["states"]).float().to(self.device)
            actions=torch.from_numpy(batch["actions"]).float().to(self.device)
            rewards=torch.from_numpy(batch["rewards"]).float().to(self.device).unsqueeze(-1)
            next_states=torch.from_numpy(batch["next_states"]).float().to(self.device)
            dones=torch.from_numpy(batch["dones"]).float().to(self.device).unsqueeze(-1)
            nstep_gamma=torch.from_numpy(batch["nstep_gamma"]).float().to(self.device).unsqueeze(-1)
            
        # TODO(SAC-3): Compute the soft Bellman target with target Q/value and temperature alpha.
            with torch.no_grad():
                critic_target=rewards+nstep_gamma*(1-dones)*self.agent.get_value(next_states,self.log_temp.detach().exp())
        # TODO(SAC-4): Evaluate both critics on (states, actions), compute TD errors, and optimize critic loss.
            q_1,q_2=self.agent.get_double_q_function(states,actions)
            td_error1=critic_target-q_1
            td_error2=critic_target-q_2
            critic_loss=(td_error1**2).mean()+(td_error2**2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        # TODO(SAC-5): Re-sample actions from the current policy with reparameterization.
            new_actions, log_probs = self.agent.resample_action(states)
            log_probs=log_probs.unsqueeze(-1)
        # TODO(SAC-6): Compute actor loss E[alpha * log_prob(action|state) - Q(state, action)] and optimize actor.
            new_q1, new_q2 = self.agent.get_double_q_function(states, new_actions)
            min_q = torch.min(new_q1, new_q2)
            actor_loss=(self.log_temp.detach().exp()*log_probs-min_q).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        # TODO(SAC-7): If self.learn_temp is enabled, optimize the temperature toward self.target_entropy.
            if self.learn_temp:
                entropy_target=self.target_entropy
                temp_loss=torch.mean(-self.log_temp*(log_probs.detach()+entropy_target))
                
                self.temp_optimizer.zero_grad()
                temp_loss.backward()
                self.temp_optimizer.step()
                
            if self.use_target and self.gradient_step % self.target_update_interval == 0:
                self.agent.update_target()
        # TODO(SAC-9): Add actor, critic, entropy, temperature, and TD-error metrics to Result.
            result.add_metric("actor/loss", actor_loss.item())

            result.add_metric("critic/loss", critic_loss.item())
            result.add_metric("critic1/q", q_1.mean().item())
            result.add_metric("critic2/q", q_2.mean().item())

            result.add_metric("entropy", (-log_probs.mean()).item())
            result.add_metric("temperature", self.log_temp.detach().exp().item())

            result.add_metric(
                "critic1/td_error_abs",
                td_error1.abs().mean().item()
            )
            result.add_metric(
                "critic2/td_error_abs",
                td_error2.abs().mean().item()
            )
            if self.learn_temp:
                result.add_metric("temperature/loss", temp_loss.item())
        # TODO(SAC-10): Increment self.gradient_step and return the Result.
            self.gradient_step+=1
            return result
        #raise NotImplementedError("TODO: complete SAC policy update.")
