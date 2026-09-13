import gymnasium as gym
import torch 
from torch.distributions import kl_divergence
import numpy as np
from typing import Tuple

from logger.logger import Logger
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase
from utils.result import Result
from utils.utils import ( #更新actor所需数学工具
    get_flat_grad, #把所有actor参数的梯度拼成一个长向量
    conjugate_gradients, #近似求解Hx=g
    kl_product, #计算kl hessian与向量乘积Hv
    get_model_flat_parameters, #把actor所有参数拼成一个长向量
    load_flat_parameters_to_model, #把长向量重新写回actor参数
    OPTIMIZER_DICT,
)
from .baseonpolicy import OnPolicyAlgorithm

class TRPO(OnPolicyAlgorithm):

    """Base class for on-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(TRPO,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, best_pth, args)

        algo_args = args.algorithm

        self.delta = algo_args.delta
        self.cg_steps = algo_args.cg_steps
        self.linesearch_coeffient = algo_args.linesearch_coeffient
        self.backtracking_steps = algo_args.backtracking_steps
        self.lr = algo_args.learning_rate
        self.lambda_ = algo_args.lambda_
        self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.critic.parameters(), lr=self.lr)
        self.advan_norm = algo_args.advan_norm
        self.critic_update_steps = algo_args.critic_update_steps
        self.critic_batch_size = algo_args.critic_batch_size
        

    def _update_buffer(self, batch):
        self.buffer.add(batch)
    
    def compute_advantages_from_traj(self)->Tuple[np.ndarray,np.ndarray]:
        states=self.traj_rollout.states
        rewards=self.traj_rollout.rewards
        dones=self.traj_rollout.dones
        masks=self.traj_rollout.masks #除了states有state_dim,shape都是(trajnum,max_episode_lenth)mask0表示是对齐用的padding,1表示是该t步是真实数据
        trajnum,max_episode_length,_=states.shape
        with torch.no_grad():
            flat_states=states.reshape(trajnum*max_episode_length,-1)
            flat_values=self.agent.get_value(flat_states)
            values=flat_values.squeeze(-1).cpu().numpy().reshape(trajnum,-1) #values.shape=(trajnum,max_episode_length)
            values=values*masks #防止padding位置经过critic变成非零
        advantages=np.zeros((trajnum,max_episode_length),dtype=np.float32) #advantages.shape=(trajnum,max_episode_length)
        for i in range(trajnum):
            traj_length=int(masks[i].sum())
            next_values=last_advantages=0.0
            for t in range(traj_length-1,-1,-1):
                nonterminal=1.0-dones[i,t]
                delta=rewards[i,t]+nonterminal*self.gamma*next_values-values[i,t]
                advantages[i,t]=delta+self.gamma*self.lambda_*nonterminal*last_advantages
                next_values=values[i,t]
                last_advantages=advantages[i,t]
        returns=advantages+values
        return advantages,returns

    def compute_advantages_from_rollout(self)->Tuple[np.ndarray,np.ndarray]:
        states=self.buffer.buffer["states"]
        rewards=self.buffer.buffer["rewards"]
        dones=self.buffer.buffer["dones"]
        flat_states=states.reshape(-1,states.shape[-1])
        with torch.no_grad():
            flat_values=self.agent.get_value(flat_states)
            num_envs,rollout_steps=rewards.shape
            values=flat_values.squeeze(-1).cpu().numpy().reshape(num_envs,rollout_steps) #values.shape=(num_envs,rollout_steps)
            last_values=self.agent.get_value(self.observations).squeeze(-1).cpu().numpy() #last_values.shape=(num_envs,)
        advantages=np.zeros_like(rewards,dtype=np.float32) #advantages.shape=(num_envs,rollout_steps)
        last_advantages=np.zeros(num_envs,dtype=np.float32) #last_advantages.shape=(num_envs,)
        next_values=last_values
        for t in range(rollout_steps-1,-1,-1):
            nonterminal=1.0-dones[:,t]
            delta=rewards[:,t]+nonterminal*self.gamma*next_values-values[:,t]
            advantages[:,t]=delta+self.gamma*self.lambda_*nonterminal*last_advantages
            next_values=values[:,t]
            last_advantages=advantages[:,t]
        returns=advantages+values
        return advantages,returns
        

        
    def _update_policy(self):
        # TODO: implement the updating procedure according to the TRPO algorithm 
        with Result("train") as result:

            if self.collect_traj:
                advantages,returns=self.compute_advantages_from_traj()
                states,actions,log_probs,masks=self.traj_rollout.states,self.traj_rollout.actions,self.traj_rollout.log_probs,self.traj_rollout.masks
                valid=masks.astype(bool)
                states,actions,log_probs,advantages,returns=states[valid],actions[valid],log_probs[valid],advantages[valid],returns[valid]
            else:
                advantages,returns=self.compute_advantages_from_rollout()
                states,actions,log_probs=self.buffer.buffer["states"],self.buffer.buffer["actions"],self.buffer.buffer["log_probs"]
                _,_,state_dim=states.shape
                _,_,action_dim=actions.shape
                states,actions=states.reshape(-1,state_dim),actions.reshape(-1,action_dim)
                advantages,returns,log_probs=advantages.reshape(-1,),returns.reshape(-1,),log_probs.reshape(-1,)
            states,actions,old_log_probs,advantages,returns=(torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in(states,actions,log_probs,advantages,returns))
            if self.advan_norm:
                advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)



            current_log_probs=self.agent.log_prob(states,actions)
            ratios=torch.exp(current_log_probs-old_log_probs)
            surrogate_target=torch.mean(ratios*advantages)
            surrogate_gradient=get_flat_grad(surrogate_target,self.agent.actor).detach() #surrogate_gradient.shape=(actor总参数，)
            with torch.no_grad():
                old_dist=self.agent.dist(states)
            current_dist=self.agent.dist(states)
            mean_kl=kl_divergence(old_dist,current_dist).mean()
            kl_grad=get_flat_grad(mean_kl,self.agent.actor,create_graph=True)
            gradient_direction=conjugate_gradients(self.agent.actor,surrogate_gradient,kl_grad,self.cg_steps)
            Hx=kl_product(gradient_direction,kl_grad,self.agent.actor)
            stepsize=torch.sqrt(2*self.delta/(1e-12+torch.sum(Hx*gradient_direction)))
        

            old_parameter=(                                       #linesearch主体
                get_model_flat_parameters(
                    self.agent.actor       
                )
                .detach()
                .clone()
            )
            full_step=stepsize*gradient_direction
            accepted=False
            for backtrack_step in range(self.backtracking_steps):
                step_fraction=self.linesearch_coeffient**backtrack_step
                candidate_parameter=step_fraction*full_step+old_parameter
                load_flat_parameters_to_model(candidate_parameter,self.agent.actor)
                with torch.no_grad():
                    current_ratios=torch.exp(
                        self.agent.log_prob(states,actions)-old_log_probs
                    )
                    new_surrogate_target=torch.mean(current_ratios*advantages)
                    new_kl=torch.mean(kl_divergence(old_dist,self.agent.dist(states)))
                    if new_surrogate_target>surrogate_target and new_kl<=self.delta:
                        new_parameter=candidate_parameter
                        accepted=True
                        break
            if not accepted:
                load_flat_parameters_to_model(old_parameter,self.agent.actor)
                new_parameter=old_parameter
                new_surrogate_target=surrogate_target
                new_kl=0
                
                
                
            #下面更新critic,当前states.shape=(sample_num,state_dim),returns.shape=(samples,)
            sample_num,_=states.shape
            for _ in range(self.critic_update_steps):
                random_indicies=torch.randperm(sample_num,device=self.device)
                for start in range(0,sample_num,self.critic_batch_size):
                    batch_indicies=random_indicies[start:start+self.critic_batch_size]
                    batch_states=states[batch_indicies]
                    batch_returns=returns[batch_indicies]
                    
                    predicted_values = self.agent.get_value(batch_states).squeeze(-1)
                    value_loss=torch.mean(predicted_values-batch_returns)**2
                
                    self.optimizer.zero_grad()
                    value_loss.backward()
                    self.optimizer.step()



        result.add_metric("value/loss", value_loss.item())
        result.add_metric("actor/backtrack_step",backtrack_step)
        result.add_metric("actor/new_surrogate_target",new_surrogate_target.item())
        result.add_metric("actor/new_kl",new_kl.item())
        if not self.advan_norm:
            result.add_metric("actor/adv_mean", advantages.mean().item())
            result.add_metric("actor/adv_max", torch.max(advantages).item())
        result.add_metric("actor/xHx",torch.sum(gradient_direction*kl_product(gradient_direction,kl_grad,self.agent.actor)).item())
        result.add_metric("actor/stepsize",stepsize.item())
        result.add_metric("actor/gradient_direction_l2norm",torch.norm(gradient_direction,p=2).item())
        result.add_metric("actor/new_param_l2norm",torch.norm(new_parameter).item())
        return result


                
