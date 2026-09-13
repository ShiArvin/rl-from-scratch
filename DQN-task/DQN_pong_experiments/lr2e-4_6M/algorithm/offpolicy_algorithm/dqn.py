from .baseoffpolicy import OffPolicyAlgorithm
from memory.memory import ReplayBuffer
import torch 
from utils.result import Result
from logger.logger import Logger
import numpy as np
from agent.atari_agent import AgentBase, AtariDQNAgent


OPTIMIZER_DICT = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW,
}

LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss(),
}


class DQN(OffPolicyAlgorithm):
    
    """DQN algorithm implementation."""

    def __init__(self, training_envs, testing_envs, buffer: ReplayBuffer, agent:AgentBase, logger: Logger, device, save_pth: str, args, target_agent=None):
        super(DQN, self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, args)
        
        algo_args = args.algorithm
        assert algo_args.name=="DQN", "The method name in args must be 'dqn' for DQN algorithm."
        self.lr = algo_args.learning_rate
        self.gamma = algo_args.gamma 
        self.start_epsilon = algo_args.start_epsilon
        self.epsilon = algo_args.start_epsilon
        self.end_epsilon = algo_args.end_epsilon
        self.epsilon_timestep = algo_args.epsilon_timestep
        self.epsilon_schedular = algo_args.epsilon_schedular
        self.use_target = algo_args.use_target 
        self.batch_size = algo_args.batch_size
        self.device = device
        
        if self.use_target:
            self.target_agent = target_agent.to(self.device)
            self._target_hard_update()

        self.target_update_method = algo_args.target_update_method
        self.target_update_interval = algo_args.target_update_interval
        self.tau = algo_args.target_update_tau


        
        self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.parameters(), lr=self.lr)
        
        
    
    def _target_hard_update(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        return
    def _target_soft_update(self):
        with torch.no_grad():
            for target_parameters,online_parameters in zip(self.target_agent.parameters(),self.agent.parameters()):
                target_parameters.copy_(self.tau*online_parameters+(1-self.tau)*target_parameters) 
        return

    def _update_buffer(self, batch):
        self.buffer.add(batch)


    def _update_policy(self):
        samples=self.buffer.sample(self.batch_size)
        states,actions,next_states=samples["states"],samples["actions"],samples["next_states"]
        online_q=self.agent.get_q(states,actions)
        rewards=torch.as_tensor(samples["rewards"],dtype=online_q.dtype,device=online_q.device).unsqueeze(1)
        dones=torch.as_tensor(samples["dones"],dtype=online_q.dtype,device=online_q.device).unsqueeze(1)
        with torch.no_grad():
            target_q=self.target_agent.get_max_q(next_states)
            target_y=rewards+self.gamma*(1-dones)*target_q
        td_error=target_y-online_q
        loss=LOSS_DICT["mse"](online_q,target_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.gradient_step += 1
        if self.target_update_method=="hard":
            if (self.gradient_step%self.target_update_interval)==0:
                self._target_hard_update()
        else:
            self._target_soft_update()
        result=Result("network")
        result.add_metric("network/loss", loss.item())
        result.add_metric("td_error", td_error.mean().item())
        result.add_metric("q_value_mean", online_q.mean().item())
       

        return result

    def random_choose_action(self):
        """You can use this function to implement epsilon-greedy exploration strategy"""
        epsilon_t=max(self.end_epsilon,self.start_epsilon-(self.interaction_step/self.epsilon_timestep)*(self.start_epsilon-self.end_epsilon))
        random_num=np.random.random()
        self.epsilon=epsilon_t
        return random_num<epsilon_t

    def interact_with_envs(self):
        batch, result = super().interact_with_envs()
        result.add_metric("epsilon",self.epsilon)
        return batch, result