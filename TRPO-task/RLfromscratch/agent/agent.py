import torch.nn as nn
import torch
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import AffineTransform, TanhTransform
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space 
from typing import Union, Tuple, Optional, Dict


from utils.networks import AtariDQNNetwork, MLPNetwork, Actor
from utils.utils import atari_state_preprocess_function


def to_correct_device_tensor(input, device, dtype=torch.float32)-> torch.Tensor:
    if isinstance(input, np.ndarray):
        return torch.tensor(input,dtype=dtype,device=device)
    elif isinstance(input, torch.Tensor):
        input = input.to(device)
        return input
    else:
        raise TypeError("input must be np array or torch tensor")


class AgentBase(nn.Module, ABC):
    def __init__(self,observation_space: Space, action_space: Space, device):
        super(AgentBase, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space 
        self.device = device
        
    
    @abstractmethod
    def select_action(self, states:np.ndarray, deterministic:bool) -> np.ndarray:
        """Choose the action according to the states agent observed. """
        pass 
    
    def _get_other_parameters(self):
        return dict(
            observation_space = self.observation_space,
            action_space = self.action_space,
            device = self.device 
        )

    def save(self, pth_file: str):
        """
        Save model to a given location.

        :param path:
        """
        torch.save({"state_dict": self.state_dict(), "data": self._get_other_parameters()}, pth_file)
        
    def load(self,path:str):
        pth_file = torch.load(path,weights_only=False)
        self.load_state_dict(pth_file['state_dict'])
        # TODO: check when other parameters are added to Agent
        data = pth_file['data']
        self.observation_space = data['observation_space']
        self.action_space = data['action_space']
        self.device = data['device']
        


class AtariDQNAgent(AgentBase):
    def __init__(self, observation_space: Space, action_space: Space, device):
        super(AtariDQNAgent, self).__init__(observation_space, action_space, device)
        self.num_actions = action_space.n 
        self.observation_shape = observation_space.shape
        self.network = AtariDQNNetwork(observation_space.shape, action_space.n)

    

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

    def get_q(self, states:np.ndarray, actions:Union[np.ndarray, torch.Tensor]):
        states = atari_state_preprocess_function(self.observation_space, states)
        states, actions = to_correct_device_tensor(states, self.device) ,to_correct_device_tensor(actions, self.device, torch.long)
        all_q_values = self.network(states)
        q_values = all_q_values.gather(1, actions)
        return q_values 

    def get_max_q(self, states:np.ndarray):
        states = atari_state_preprocess_function(self.observation_space, states)
        states = to_correct_device_tensor(states, self.device)
        all_q_values = self.network(states)
        q_max = torch.max(all_q_values, dim=1, keepdim=True).values
        return q_max 





class GaussianAgent(AgentBase):

    # all the tensor returned could do the gradient propogation, except the select_action
    def __init__(self, observation_space: Space, action_space: Space, device, hidden_sizes=[64,64], activation=nn.ReLU, rescale=True, action_bound_method="tanh", action_eps=1e-6):
        super(GaussianAgent, self).__init__(observation_space, action_space, device)
        self.action_dim = action_space.shape[0]
        
        self.observation_shape = observation_space.shape
        
        self.actor = Actor(observation_space.shape[0], self.action_dim, hidden_sizes, activation)
        self.critic = MLPNetwork(observation_space.shape[0], 1, hidden_sizes, activation)
        
        self.rescale = rescale 
        self.action_bound_method = action_bound_method
        self.low, self.high = torch.tensor(self.action_space.low,device=self.device), torch.tensor(self.action_space.high, device=self.device)
        self.loc, self.scale = (self.low+self.high)/2, (self.high-self.low)/2
        self.action_eps = action_eps
    
    def dist(self,states):
        assert len(states.shape)<=2
        states = to_correct_device_tensor(states, self.device)
        mu,std = self.actor(states)
        dist = Independent(Normal(mu,std), 1) # this will be seen as (batch,) action_dim-dimension distributions instead of batch*action_dim 1-dimension distributions
        if self.rescale:
            if self.action_bound_method=="tanh":
                dist = TransformedDistribution(dist, [TanhTransform(cache_size=1), AffineTransform(loc=self.loc, scale=self.scale)]) # The notation wrote by torch is wrong: the order of the transformation is tanh->affine; but the notation is affine->tanh
            else:
                raise NotImplementedError 
        return dist

    def select_action(self, states, deterministic=False)->Tuple[np.ndarray, Dict]:
        """
        select_action 的 Docstring
        :param states: (batch, state_dim)
        :return: actions: (batch, action_dim)
        """
        assert len(states.shape)<=2
        states = to_correct_device_tensor(states, self.device)
        dist = self.dist(states) 
        
        if deterministic:
            # take the mean of the original Gaussian distribution and transform it 
            actions = self.scale*torch.tanh(dist.base_dist.mean) + self.loc
        else:
            actions = dist.sample()
        
        # clamp the action to avoid the log_prob=nan
        actions = torch.clamp(actions,min=self.low+self.action_eps,max=self.high-self.action_eps)
        log_probs = dist.log_prob(actions)
        return actions.detach().cpu().numpy(), {"log_probs":log_probs.detach().cpu().numpy()}

    def get_value(self,states):
        assert len(states.shape)<=2
        # output: (batch,1)
        states = to_correct_device_tensor(states, self.device)
        values = self.critic(states)
        return values
    
    def log_prob(self, states, actions):
        # output: (batch,)
        assert len(states.shape)<=2
        assert len(actions.shape)<=2
        states, actions = to_correct_device_tensor(states, self.device), to_correct_device_tensor(actions, self.device)
        dist = self.dist(states)
        
        log_prob = dist.log_prob(actions)
        return log_prob
