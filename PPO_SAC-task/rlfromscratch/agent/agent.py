import torch.nn as nn
import torch
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import AffineTransform, TanhTransform
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space 
from typing import Union, Tuple, Optional, Dict
from gymnasium.spaces import Box, Discrete


from utils.networks import AtariDQNNetwork, MLPNetwork, Actor, Critic, DeterministicActor, DoubleQFunction, QFunction, DiagGaussianActor, TanhGaussianActor, DiscreteProbabilityActor
from utils.utils import atari_state_preprocess_function

DOUBLE_CRITIC = [DoubleQFunction]
EPS = 1e-8


def to_correct_device_tensor(input, device, dtype=torch.float32)-> torch.Tensor:
    if isinstance(input, np.ndarray):
        return torch.tensor(input,dtype=dtype,device=device)
    elif isinstance(input, torch.Tensor):
        input = input.to(device)
        return input
    else:
        raise TypeError("input must be np array or torch tensor")


class AgentBase(nn.Module, ABC):
    # config_attrs is a tuple of strings that are the names of the attributes that are used to configure the agent
    _config_attrs: tuple[str, ...] = ("observation_space", "action_space") 
    def __init__(self,observation_space: Space, action_space: Space, device):
        super(AgentBase, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space 
        self.device = device
        
    
    @abstractmethod
    def select_action(self, states:np.ndarray, deterministic:bool) -> np.ndarray:
        """Choose the action according to the states agent observed. """
        pass 
    
   
    def _get_config_dict(self) -> dict:
        return {name: getattr(self, name) for name in self._config_attrs}


    def save(self, pth_file: str):
        """
        Save model to a given location.

        :param path:
        """
        # we need to save the states_dict and other parameters that are used to configure the agent, 
        # but notice: 1. device should not be saved 2. the extra parameters in actor, critic are not saved. These parameters should be appointed by the user when creating the model.
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self._get_config_dict()
            },
            pth_file
        )
        
    def load(self,pth_file:str):
        ckpt = torch.load(pth_file, weights_only=False, map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        for name, value in ckpt["config"].items():
            setattr(self, name, value)
        

class AtariDQNAgent(AgentBase):
    _config_attrs = AgentBase._config_attrs + ("num_actions", "observation_shape")
    def __init__(self, observation_space: Space, action_space: Space, device, network:AtariDQNNetwork, use_target=False, target_network=None, target_update_tau=0.005,target_update_method="hard"):
        super(AtariDQNAgent, self).__init__(observation_space, action_space, device)
        self.num_actions = action_space.n 
        self.observation_shape = observation_space.shape
        self.network = network
        self.use_target = use_target
        if self.use_target:
            self.target_network = target_network
            self._target_hard_update()
        self.target_update_tau = target_update_tau
        self.target_update_method = target_update_method
    
    def _target_hard_update(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def _target_soft_update(self):
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(
                self.target_update_tau * param.data + (1 - self.target_update_tau) * target_param.data
            )
    
    def target_update(self):
        if self.target_update_method=="hard":
            self._target_hard_update()
        elif self.target_update_method=="soft":
            self._target_soft_update()
        else:
            raise ValueError("target_update_method must be either 'hard' or 'soft'")

    

    def select_action(self, state:np.ndarray, deterministic=False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        params: state: (batch, channel, 84,84) or (channel, 84,84)
        output: action: (batch, )
        """
        
        state = atari_state_preprocess_function(self.observation_space, state)
        state = to_correct_device_tensor(state, self.device)

        if len(state.shape)==3: # state's shape is (channel, 84, 84)
            state = state.unsqueeze(0)

        all_q_values = self.network(state)
        
        if deterministic:
            action = torch.argmax(all_q_values, dim=1,keepdim=True)
        else:
            action = torch.multinomial(torch.softmax(all_q_values, dim=1), num_samples=1)

        return action.cpu().numpy(), None

    def get_q(self, states:np.ndarray, actions:Union[np.ndarray, torch.Tensor],target=False):
        states = atari_state_preprocess_function(self.observation_space, states)
        states, actions = to_correct_device_tensor(states, self.device) ,to_correct_device_tensor(actions, self.device, torch.long)
        if target and self.use_target:
            all_q_values = self.target_network(states)
        else:
            all_q_values = self.network(states)
        q_values = all_q_values.gather(1, actions)
        return q_values 

    def get_max_q(self, states:np.ndarray,target=False):
        states = atari_state_preprocess_function(self.observation_space, states)
        states = to_correct_device_tensor(states, self.device)
        if target and self.use_target:
            all_q_values = self.target_network(states)
        else:
            all_q_values = self.network(states)
        q_max = torch.max(all_q_values, dim=1, keepdim=True).values
        return q_max 

    def get_ddqn_target(self, states):
        states = atari_state_preprocess_function(self.observation_space, states)
        states = to_correct_device_tensor(states, self.device)
        all_q_values_target = self.target_network(states)
        all_q_values_online = self.network(states)
        chosen_actions = torch.argmax(all_q_values_online, dim=1, keepdim=True)
        return all_q_values_target.gather(1,chosen_actions)




class A2CAgent(AgentBase):
    _config_attrs = AgentBase._config_attrs
    def __init__(self, 
        observation_space: Space, 
        action_space: Space, 
        device:torch.device, 
        actor:Actor, 
        critic:Critic
    ):
        super(A2CAgent, self).__init__(observation_space, action_space, device)
        
        self.actor = actor
        self.critic = critic
        if self.critic.__class__ in DOUBLE_CRITIC:
            self.double_critic = True
        else:
            self.double_critic = False
        
        
        
        if isinstance(self.action_space, Box):
            self.low, self.high = torch.tensor(self.action_space.low,device=self.device), torch.tensor(self.action_space.high, device=self.device)
            self.loc, self.scale = (self.low+self.high)/2, (self.high-self.low)/2
        
    
    def select_action(self, states:Union[np.ndarray, torch.Tensor], deterministic=False)->Tuple[np.ndarray, Dict]:
        """
        select_action 的 Docstring
        :param states: (batch, state_dim)
        :return: actions: (batch, action_dim)
        """
        raise NotImplementedError


    def get_value(self,states:np.ndarray):
        """
        states: np.ndarray:(batch, state_dim)
        output: (batch,1)
        """
        # output: (batch,1)
        states = to_correct_device_tensor(states, self.device)
        return self.critic(states)
    
    def get_double_q_function(self,states:np.ndarray,actions:np.ndarray):
        """
        states: np.ndarray:(batch, state_dim)
        actions: np.ndarray:(batch, action_dim)
        output: (batch,1)
        """
        assert self.double_critic
        states, actions = to_correct_device_tensor(states, self.device), to_correct_device_tensor(actions, self.device)
        q1, q2 = self.critic(states, actions)
        return q1, q2
    
    def get_q_function(self,states:np.ndarray,actions:np.ndarray):
        """
        states: np.ndarray:(batch, state_dim)
        actions: np.ndarray:(batch, action_dim)
        output: (batch,1)
        """
        states, actions = to_correct_device_tensor(states, self.device), to_correct_device_tensor(actions, self.device)
        if self.double_critic:
            q1, q2 = self.critic(states, actions)
            return torch.min(q1,q2)
        else:
            return self.critic(states, actions)
    
 

class ProbabilityA2CAgent(A2CAgent):
    _config_attrs = A2CAgent._config_attrs
    def __init__(self, 
        observation_space: Space, 
        action_space: Space, 
        device:torch.device, 
        actor:Union[DiagGaussianActor, TanhGaussianActor, DiscreteProbabilityActor], 
        critic:Critic
        ):
        super(ProbabilityA2CAgent, self).__init__(observation_space, action_space, device, actor, critic)

    def select_action(self, states:Union[np.ndarray, torch.Tensor], deterministic=False)->Tuple[np.ndarray, Dict]:
        """
        select_action 的 Docstring
        :param states: (batch, state_dim)
        :return: actions: (batch, action_dim)
        """
        states = to_correct_device_tensor(states, self.device)
        actions, log_probs = self.actor.get_action(states, deterministic)
        return actions.detach().cpu().numpy(), {"log_probs":log_probs.detach().cpu().numpy()}
        

    def dist(self,states):
        assert len(states.shape)<=2
        states = to_correct_device_tensor(states, self.device)
        return self.actor.get_dist(states)
    
    def log_prob(self, states:np.ndarray, actions:np.ndarray):
        """
        states: np.ndarray:(batch, state_dim)
        actions: np.ndarray:(batch, action_dim)
        output: (batch,)
        """
        # output: (batch,)
        states, actions = to_correct_device_tensor(states, self.device), to_correct_device_tensor(actions, self.device)
        return self.actor.get_log_prob(states, actions)
    
    def get_entropy(self, states:np.ndarray):
        """
        states: np.ndarray:(batch, state_dim)
        output: (batch,)
        """
        states = to_correct_device_tensor(states, self.device)
        return self.actor.get_entropy(states)

class DeterminiticA2CAgent(A2CAgent):
    _config_attrs = A2CAgent._config_attrs
    def __init__(self, 
        observation_space: Space, 
        action_space: Space, 
        device:torch.device, 
        actor:DeterministicActor, 
        critic:Critic
        ):
        super(DeterminiticA2CAgent, self).__init__(observation_space, action_space, device, actor, critic)

    def select_action(self, states:Union[np.ndarray, torch.Tensor], deterministic=False)->Tuple[np.ndarray, Dict]:
        """
        states: (batch, state_dim)
        return: actions: (batch, action_dim)
        """
        states = to_correct_device_tensor(states, self.device)
        actions = self.actor.get_action(states, deterministic)
        return actions.detach().cpu().numpy(), None
        

    def dist(self,states):
        raise NotImplementedError
    
    def log_prob(self, states:np.ndarray, actions:np.ndarray):
        """
        states: np.ndarray:(batch, state_dim)
        actions: np.ndarray:(batch, action_dim)
        output: (batch,)
        """
        # output: (batch,)
        raise NotImplementedError    


class PolicyAgent(AgentBase):
    _config_attrs = AgentBase._config_attrs

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        device: torch.device,
        actor: Actor,
    ):
        super(PolicyAgent, self).__init__(observation_space, action_space, device)
        self.actor = actor

    def dist(self, states):
        states = to_correct_device_tensor(states, self.device)
        return self.actor.get_dist(states)

    def select_action(self, states: Union[np.ndarray, torch.Tensor], deterministic=False) -> Tuple[np.ndarray, Dict]:
        states = to_correct_device_tensor(states, self.device)
        return self.actor.get_action(states, deterministic)

    def log_prob(self, states: Union[np.ndarray, torch.Tensor], actions: Union[np.ndarray, torch.Tensor]):
        states = to_correct_device_tensor(states, self.device)
        actions = to_correct_device_tensor(actions, self.device)
        return self.actor.get_log_prob(states, actions)



class SACAgent(ProbabilityA2CAgent):
    _config_attrs = A2CAgent._config_attrs + (
        "use_target",
        "target_update_method",
        "target_update_tau",
        "double_critic",
    )
    def __init__(self, 
        observation_space: Space, 
        action_space: Space, 
        device:torch.device, 
        actor:Actor, 
        critic:Critic, 
        target_critic:nn.Module=None, 
        use_target:bool=False,
        target_update_method="soft",
        target_update_tau=0.005,
        double_critic=True
    ):
        super(SACAgent, self).__init__(observation_space, action_space, device, actor, critic)
        
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.use_target = use_target
        self.target_update_method = target_update_method
        self.target_update_tau = target_update_tau
        if self.use_target:
            self._target_hard_update()
        self.double_critic = double_critic

    def _target_hard_update(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
    def _target_soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.target_update_tau * param.data + (1 - self.target_update_tau) * target_param.data
            )
    def update_target(self):
        if self.target_update_method == "soft":
            self._target_soft_update()
        else:
            self._target_hard_update()

    def get_q_function_from_target(self, states:Union[np.ndarray, torch.Tensor], actions:Union[np.ndarray, torch.Tensor])->torch.Tensor:
        states = to_correct_device_tensor(states, self.device)
        actions = to_correct_device_tensor(actions, self.device)
        if self.double_critic:
            q1, q2 = self.target_critic(states, actions)
            return torch.min(q1, q2)
        else:
            return self.target_critic(states, actions)

    def get_value(self, states:np.ndarray,temperature:torch.Tensor):
        """
        states: np.ndarray:(batch, state_dim)
        temperature: tensor:(1,)
        output: (batch,1)
        """
        # In SAC, get_value is used to get the value of the state-action pair, so we don't need to do gradient update here
        assert len(states.shape)<=2
        states = to_correct_device_tensor(states, self.device)
        
        actions, log_probs = self.actor.get_action(states, deterministic=False)

        log_probs = log_probs.unsqueeze(1)
        if self.use_target:
            q = self.get_q_function_from_target(states, actions)
        else:
            q = self.get_q_function(states, actions)
        
        values = q - temperature*log_probs
        
        return values

    def resample_action(self,states:np.ndarray):
        """
        states: np.ndarray:(batch, state_dim)
        output: (batch, action_dim)
        """
        states = to_correct_device_tensor(states, self.device)
        return self.actor.get_action(states,deterministic=False)


class TD3Agent(DeterminiticA2CAgent):
    _config_attrs = A2CAgent._config_attrs + (
        "use_target",
        "target_update_method",
        "target_update_tau",
        "double_critic",
    )
    """
    TD3 Agent uses target network for actor and critic. 
    """
    def __init__(self, 
        observation_space: Space, 
        action_space: Space, 
        device:torch.device, 
        actor:DeterministicActor, 
        critic:DoubleQFunction, 
        target_actor:nn.Module=None,
        target_critic:nn.Module=None, 
        use_target:bool=False,
        target_update_method="soft",
        target_update_tau=0.005,
        double_critic=True
    ):
        super(TD3Agent, self).__init__(observation_space, action_space, device, actor, critic)
        
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.use_target = use_target
        self.target_update_method = target_update_method
        self.target_update_tau = target_update_tau
        if self.use_target:
            self._target_hard_update()
        self.double_critic = double_critic

    def _target_hard_update(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
    
    def _target_soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.target_update_tau * param.data + (1 - self.target_update_tau) * target_param.data
            )
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.target_update_tau * param.data + (1 - self.target_update_tau) * target_param.data
            )

    def update_target(self):
        if self.target_update_method == "soft":
            self._target_soft_update()
        else:
            self._target_hard_update()
    
    def select_action(self, states:Union[np.ndarray, torch.Tensor], deterministic=False)->Tuple[np.ndarray, Dict]:
        """
        select_action 的 Docstring
        :param states: (batch, state_dim)
        :return: actions: (batch, action_dim)
        """
        states = to_correct_device_tensor(states, self.device)
        actions = torch.clamp(self.actor.get_action(states, deterministic), min=self.low,max=self.high)
        return actions.detach().cpu().numpy(), None

    def select_action_from_target(self, states:Union[np.ndarray, torch.Tensor], noise: torch.Tensor)->torch.Tensor:
        assert len(states.shape)<=2
        states = to_correct_device_tensor(states, self.device)
        actions = self.target_actor.get_action(states, deterministic=False) + noise
        actions = torch.clamp(actions, min=self.low,max=self.high)
        return actions
    
    def get_q_function_from_target(self, states:Union[np.ndarray, torch.Tensor], actions:Union[np.ndarray, torch.Tensor])->torch.Tensor:
        states = to_correct_device_tensor(states, self.device)
        actions = to_correct_device_tensor(actions, self.device)
        if self.double_critic:
            q1, q2 = self.target_critic(states, actions)
            return torch.min(q1, q2)
        else:
            return self.target_critic(states, actions)

    def get_action_with_gradient(self, states)->torch.Tensor:
        states = to_correct_device_tensor(states, self.device)
        return self.actor.get_action(states, deterministic=True)
        



