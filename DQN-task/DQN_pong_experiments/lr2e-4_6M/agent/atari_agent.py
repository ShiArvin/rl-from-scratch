import torch.nn as nn
import torch
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space 
from typing import Union

from utils.networks import AtariDQNNetwork
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

    

    def select_action(self, state:np.ndarray, deterministic=False) -> np.ndarray:
       
        new_state=atari_state_preprocess_function(self.observation_space,state) 
        new_state=to_correct_device_tensor(new_state,self.device)
        with torch.no_grad():
            q_values=self.network(new_state)
            if deterministic:
                self.actions=torch.argmax(q_values,dim=1,keepdim=True)
            else:
                self.actions=torch.multinomial(torch.softmax(q_values,dim=1),num_samples=1)
        return self.actions.cpu().numpy()

    def get_q(self, states:np.ndarray, actions:Union[np.ndarray, torch.Tensor]):
        """
        params: state: (batch, channel, 84,84)
        output: action: (batch, 1)
        """
        new_states=atari_state_preprocess_function(self.observation_space,states)
        new_states=to_correct_device_tensor(new_states,self.device)
        new_actions=torch.tensor(actions,dtype=torch.long,device=self.device)
        all_q=self.network(new_states)
        if new_actions.dim()==1:
            new_actions=new_actions.unsqueeze(1)
        selected_q=torch.gather(all_q,dim=1,index=new_actions)
        return selected_q


    def get_max_q(self, states:np.ndarray):
        new_states=atari_state_preprocess_function(self.observation_space,states)
        new_states=to_correct_device_tensor(new_states,self.device)
        all_q=self.network(new_states)
        max_q=torch.max(all_q,dim=1,keepdim=True).values
        return max_q