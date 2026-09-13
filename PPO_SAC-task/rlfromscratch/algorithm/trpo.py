import gymnasium as gym
import torch 
from torch.distributions import kl_divergence
import numpy as np
from typing import Tuple
import math

from logger.logger import Logger
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase
from utils.result import Result
from utils.utils import get_flat_grad, conjugate_gradients, kl_product, get_model_flat_parameters, load_flat_parameters_to_model
from utils import OPTIMIZER_DICT
from .baseonpolicy import OnPolicyAlgorithm

class TRPO(OnPolicyAlgorithm):

    """Implementation for TRPO algorithm."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(TRPO,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, best_pth, args)

        algo_args = args.algorithm

        self.delta = algo_args.delta
        self.cg_steps = algo_args.cg_steps
        self.linesearch_coeffient = algo_args.linesearch_coeffient
        self.backtracking_steps = algo_args.backtracking_steps
        self.lr = algo_args.learning_rate
        self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.critic.parameters(), lr=self.lr)
        self.advan_norm = algo_args.advan_norm
        self.critic_update_steps = algo_args.critic_update_steps
        self.critic_batch_size = algo_args.critic_batch_size
        self.eigenvalue_reg = algo_args.eigenvalue_reg
        self.log_clip_stabilize = algo_args.log_clip_stabilize
        
        

    def _update_buffer(self, batch):
        self.buffer.add(batch)
    

    def _update_policy(self):
        with Result("train") as result:
            # get transitions
            if self.collect_traj:
                states, actions, masks, old_log_probs = self.traj_rollout.states, self.traj_rollout.actions, self.traj_rollout.masks, self.traj_rollout.log_probs
                returns, advantages, old_values = self.compute_advantages_from_traj()
            
                states = states.reshape((-1,*self.observation_space.shape))
                actions = actions.reshape((-1,actions.shape[-1]))
                masks = masks.reshape((-1))
                old_log_probs = old_log_probs.reshape((-1))
                advantages = advantages.reshape((-1))
                returns = returns.reshape((-1))

                # remove the padding
                masks = masks==1
                states = states[masks]
                actions = actions[masks]
                old_log_probs = old_log_probs[masks]
                advantages = advantages[masks]
                returns = returns[masks]
            else:
                states, actions, old_log_probs = self.buffer.buffer['states'], self.buffer.buffer['actions'], self.buffer.buffer['log_probs']

                returns, advantages, old_values = self.compute_advantages_from_rollout()
                states = states.reshape((-1,*self.observation_space.shape))
                actions = actions.reshape((-1,actions.shape[-1]))
                old_log_probs = old_log_probs.reshape((-1))
                advantages = advantages.reshape((-1))
                returns = returns.reshape((-1))



            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            old_log_probs = torch.from_numpy(old_log_probs).float().to(self.device)
            advantages = torch.from_numpy(advantages).float().to(self.device)
            returns = torch.from_numpy(returns).float().to(self.device)

            if self.advan_norm:
                advantages = (advantages-advantages.mean())/(advantages.std()+self._eps)



            log_probs = self.agent.log_prob(states,actions)
            if self.log_clip_stabilize:
            # the max clamp here is to avoid inf in ratios and advantages, which will cause the nan in gradient
                ratios = (torch.clamp(log_probs - old_log_probs,max=50)).exp()
            else:
                ratios = (log_probs - old_log_probs).exp()

            surrogate_target = torch.mean(ratios*advantages) 

            surrogate_gradient = get_flat_grad(surrogate_target, self.agent.actor, retain_graph=True).detach() # retain_graph=True

            dist = self.agent.dist(states)
            with torch.no_grad():
                old_dist = self.agent.dist(states)
            
            # TODO: here we need to consider other distributions which can't directly use the kl_divergence function
            kl = kl_divergence(old_dist, dist).mean()

            kl_grad = get_flat_grad(kl, self.agent.actor, create_graph=True)

            gradient_direction = conjugate_gradients(model=self.agent.actor,b=surrogate_gradient, flat_kl_grad=kl_grad, nsteps=self.cg_steps, eginvalue_reg=self.eigenvalue_reg)
            
            # TODO: if we replace the xHx with the calculation function we will find that the result is different, we need to figure out the reason
            xHx = torch.sum(gradient_direction*kl_product(gradient_direction,kl_grad,self.agent.actor, self.eigenvalue_reg))
            stepsize = torch.sqrt(2*self.delta / (xHx   +1e-8))

            
            



            # linesearch

            fail = False
            with torch.no_grad():
                backtrack_step = 0
                flat_params = get_model_flat_parameters(self.agent.actor)
                for i in range(self.backtracking_steps):
                    new_parameter = flat_params + stepsize*gradient_direction

                    load_flat_parameters_to_model(new_parameter, self.agent.actor)
                    new_dist = self.agent.dist(states)
                    new_log_prob = new_dist.log_prob(actions)
                    new_ratios = (new_log_prob-old_log_probs).exp()
                    new_surrogate_target = torch.mean(new_ratios*advantages)

                    new_kl = kl_divergence(old_dist, new_dist).mean()
                    
                    if new_surrogate_target>surrogate_target and new_kl <= self.delta:
                        backtrack_step = i
                        break
                    
                    stepsize *= self.linesearch_coeffient
                    if i==self.backtracking_steps-1:
                        backtrack_step = i
                        load_flat_parameters_to_model(flat_params,self.agent.actor)
                        fail = True
            
            # update the value function 
            dataset_size = states.size(0)
            batch_size = min(self.critic_batch_size, dataset_size)

            for _ in range(self.critic_update_steps):
                indices = torch.randperm(dataset_size, device=states.device)

                for start in range(0, dataset_size, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_states = states[batch_idx]
                    batch_returns = returns[batch_idx]

                    values = self.agent.get_value(batch_states).squeeze(1)
                    value_loss = ((values - batch_returns) ** 2).mean()

                    self.optimizer.zero_grad()
                    value_loss.backward()
                    self.optimizer.step()

            # # TODO:do the batch update or the epoch update
            # batch_size = states.shape[0]
            # critic_batch_size = min(self.critic_batch_size, batch_size)
            # value_loss = torch.tensor(0.0, device=self.device)
            # for _ in range(self.critic_update_steps):
            #     indices = torch.randperm(batch_size, device=self.device)[:critic_batch_size]
            #     values = self.agent.get_value(states[indices]).squeeze(-1)
            #     value_loss = torch.mean((values - returns[indices]) ** 2)

            #     self.optimizer.zero_grad()
            #     value_loss.backward()
            #     self.optimizer.step()


        self.gradient_step += 1

        result.add_metric("value/loss", value_loss.item())
        result.add_metric("actor/backtrack_step",backtrack_step)
        if not fail:
            result.add_metric("actor/new_surrogate_target",new_surrogate_target.item())
            result.add_metric("actor/new_kl",new_kl.item())
        else:
            result.add_metric("actor/new_surrogate_target",surrogate_target.item())
            result.add_metric("actor/new_kl",kl.item())
        if not self.advan_norm:
            result.add_metric("actor/adv_mean", advantages.mean().item())
            result.add_metric("actor/adv_max", torch.max(advantages).item())
        # result.add_metric("actor/xHx",torch.sum(gradient_direction*kl_product(gradient_direction,kl_grad,self.agent.actor,self.eigenvalue_reg)).item())
        result.add_metric("actor/stepsize",stepsize.item())
        result.add_metric("actor/gradient_direction_l2norm",torch.norm(gradient_direction,p=2).item())
        result.add_metric("actor/new_param_l2norm",torch.norm(new_parameter).item())
        return result

