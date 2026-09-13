import gymnasium as gym 
import torch

from utils.result import Result
from utils import OPTIMIZER_DICT
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import ProbabilityA2CAgent
from logger.logger import Logger
from .baseonpolicy import OnPolicyAlgorithm


class PPO(OnPolicyAlgorithm):
    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: ProbabilityA2CAgent, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(PPO,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, best_pth, args)

        algo_args = args.algorithm
        self.eps_clip = algo_args.eps_clip
        self.value_coef = algo_args.value_coef
        self.use_entropy_loss= algo_args.use_entropy_loss
        self.entropy_coef = algo_args.entropy_coef
        self.actor_lr = algo_args.actor_lr
        self.critic_lr = algo_args.critic_lr 
        if algo_args.common_head:
            self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](
            [   {"params": self.agent.actor.encoder.parameters(), "lr": algo_args.encoder_lr},
                {"params": self.agent.actor.fc.parameters(), "lr": self.actor_lr},
                {"params": self.agent.critic.fc.parameters(), "lr": self.critic_lr}
                ]
            )
        else:
            self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](
                [
                    {"params": self.agent.actor.parameters(), "lr": self.actor_lr},
                    {"params": self.agent.critic.parameters(), "lr": self.critic_lr}
                    ]
                )
        self.lr_decay = algo_args.lr_decay
        if self.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(1.0 - step / self.total_update_steps, 0.0)
            )

        self.update_epochs = algo_args.update_epochs
        self.minibatch_size = algo_args.minibatch_size
        self.advan_norm = algo_args.advan_norm
        
        self.use_grad_clip = algo_args.use_grad_clip #是否启用梯度裁剪，防止某个minibatch产生过大梯度导致参数更新过猛
        self.max_grad_norm = algo_args.max_grad_norm #梯度裁剪阈值。若总梯度范数超过该值，就按比例缩小到这个范围内
        self.use_value_clip = algo_args.use_value_clip #是否启用 critic 的 value clipping。限制新的 V(s) 相比旧 V(s) 改变得太剧烈
        self.value_clip = algo_args.value_clip #value clipping 的截断范围，比如 0.2 表示限制 value prediction 相对旧值的变化幅度
        self.log_clip_stabilize = algo_args.log_clip_stabilize #是否对ratio的log进行截断，防止后续计算指数时数值太大溢出
        self.recompute_adv = algo_args.recompute_adv #是否在 PPO 多个 update_epoch 中，用更新后的 critic 重新计算 advantage；关闭时通常整批数据一直使用最开始算好的 advantage
        

    def _update_buffer(self, batch):
        self.buffer.add(batch)

   
    def _update_with_minibatch(self,states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor, old_values: torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        old_log_probs: torch.Tensor:(batch,)
        advantages: torch.Tensor:(batch,)
        returns: torch.Tensor:(batch,)
        old_values: torch.Tensor:(batch,)
        """
        # TODO(PPO-1): Use the current policy to recompute log_probs for states/actions.
        new_log_probs=self.agent.log_prob(states,actions)
        # TODO(PPO-2): Compute probability ratios exp(new_log_prob - old_log_prob).
        log_ratio=new_log_probs-old_log_probs
        if self.log_clip_stabilize:
            ratios=torch.clip(log_ratio,max=50).exp()
        else:
            ratios=torch.exp(log_ratio)
        # TODO(PPO-3): Build the clipped surrogate actor objective with self.eps_clip.
        clip_surrogate=torch.mean(torch.min(ratios*advantages,torch.clip(ratios,1-self.eps_clip,1+self.eps_clip)*advantages))
        actor_loss=-clip_surrogate
        # TODO(PPO-4): Use self.agent.get_value(states) to compute critic loss against returns.
        new_values=self.agent.get_value(states).squeeze(-1)
        critic_loss=(new_values-returns)**2
        # TODO(PPO-5): If self.use_value_clip is enabled, add PPO-style value clipping.
        if self.use_value_clip:
            clip_values=(old_values+torch.clip(new_values-old_values,-self.value_clip,self.value_clip))
            critic_loss_clip=(clip_values-returns)**2
            critic_loss=0.5*torch.mean(torch.max(critic_loss,critic_loss_clip))
        else:
            critic_loss=0.5*torch.mean(critic_loss)
        # TODO(PPO-6): If self.use_entropy_loss is enabled, subtract entropy regularization.
        if self.use_entropy_loss:
            policy_entr=self.agent.get_entropy(states)
        else:
            policy_entr=torch.tensor(0.0,device=states.device)
        # TODO(PPO-7): Backpropagate total loss, optionally apply grad clipping, and step optimizer.
        total_loss=actor_loss+self.value_coef*critic_loss-policy_entr*self.entropy_coef
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(),self.max_grad_norm)
        self.optimizer.step()
        # TODO(PPO-8): Return a dict of training metrics used by _update_policy.
        metrics={
            "actor/loss":actor_loss.item(),
            "value/loss":critic_loss.item(),
            "actor/entropy":policy_entr.item(),
            "loss/total":total_loss.item(),
            "actor/ratio_mean":ratios.mean().item(),
        }
        return metrics
       




    
    def _update_policy(self):
        # TODO(PPO-9): Create Result("train") to record update metrics.
        with Result("train") as result:
        # TODO(PPO-10): For each update epoch, compute returns/advantages from trajectory or rollout buffer.
            if not self.collect_traj:
                states,actions,log_probs=(self.buffer.buffer[k] for k in ["states","actions","log_probs"])
                returns,advantages,old_values=self.compute_advantages_from_rollout()
            else:
                states = self.traj_rollout.states
                actions = self.traj_rollout.actions
                log_probs = self.traj_rollout.log_probs
                masks = self.traj_rollout.masks
                returns,advantages,old_values=self.compute_advantages_from_traj()
        # TODO(PPO-11): Flatten env/time dimensions; when collect_traj=True, use masks to remove padding.
            if not self.collect_traj:
                envs,timestep,_=states.shape
                flat_states=states.reshape(envs*timestep,-1)
                flat_actions=actions.reshape(envs*timestep,-1)
                flat_log_probs=log_probs.reshape(envs*timestep)
                flat_returns=returns.reshape(envs*timestep)
                flat_advantages=advantages.reshape(envs*timestep)
                flat_old_values=old_values.reshape(envs*timestep)
            else:
                trajnum,maxstep,_=states.shape
                flat_states=states.reshape(trajnum*maxstep,-1)
                flat_actions=actions.reshape(trajnum*maxstep,-1)
                flat_log_probs=log_probs.reshape(trajnum*maxstep)
                flat_returns=returns.reshape(trajnum*maxstep)
                flat_advantages=advantages.reshape(trajnum*maxstep)
                flat_old_values=old_values.reshape(trajnum*maxstep)
                flat_masks=masks.reshape(trajnum*maxstep)
                
                valid_masks=flat_masks.astype(bool)
                
                flat_states=flat_states[valid_masks]
                flat_actions=flat_actions[valid_masks]
                flat_log_probs=flat_log_probs[valid_masks]
                flat_returns=flat_returns[valid_masks]
                flat_advantages=flat_advantages[valid_masks]
                flat_old_values=flat_old_values[valid_masks]
                
        # TODO(PPO-12): Convert numpy arrays to torch tensors on self.device.
            flat_states=torch.tensor(flat_states,dtype=torch.float32,device=self.device)
            flat_actions=torch.tensor(flat_actions,dtype=torch.float32,device=self.device)
            flat_log_probs=torch.tensor(flat_log_probs,dtype=torch.float32,device=self.device)
            flat_returns=torch.tensor(flat_returns,dtype=torch.float32,device=self.device)
            flat_advantages=torch.tensor(flat_advantages,dtype=torch.float32,device=self.device)
            flat_old_values=torch.tensor(flat_old_values,dtype=torch.float32,device=self.device)
        # TODO(PPO-13): If self.advan_norm is enabled, normalize advantages with self._eps.
            if self.advan_norm:
                flat_advantages=(flat_advantages-flat_advantages.mean())/(flat_advantages.std(unbiased=False)+self._eps)
        # TODO(PPO-14): Shuffle samples, split them into minibatches, and call _update_with_minibatch.
            dataset_size=flat_states.shape[0]
            metric_sums = {}
            metric_count = 0
            for _ in range(self.update_epochs):
                indices=torch.randperm(dataset_size,device=self.device)
                for start in range(0,dataset_size,self.minibatch_size):
                    
                    shuffle_idx=indices[start:start+self.minibatch_size]
                    batch_states=flat_states[shuffle_idx]
                    batch_actions=flat_actions[shuffle_idx]
                    batch_log_probs=flat_log_probs[shuffle_idx]
                    batch_advantages=flat_advantages[shuffle_idx]
                    batch_returns=flat_returns[shuffle_idx]
                    batch_old_values=flat_old_values[shuffle_idx]
                    metrics=self._update_with_minibatch(batch_states,batch_actions,batch_log_probs,batch_advantages,batch_returns,batch_old_values)
                    
                    
                    for key, value in metrics.items():
                        metric_sums[key] = (metric_sums.get(key, 0.0) + value)

                    metric_count += 1
        # TODO(PPO-15): Add returned metrics to Result, update lr scheduler if enabled, increment gradient_step.
        for key, value in metric_sums.items():
            result.add_metric(key, value / metric_count)

        if self.lr_decay:
                self.scheduler.step()

        self.gradient_step += 1
        return result
        #raise NotImplementedError("TODO: complete PPO policy update.")