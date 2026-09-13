import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class AtariDQNNetwork(nn.Module):
    def __init__(self, input_shape:Union[tuple, list], num_actions):
        """
        input_shape: the shape of the input figure (C, H, W), usually (4, 84, 84)
        num_actions: Atari's action space is Discrete
        """
        super(AtariDQNNetwork, self).__init__()
        self.stack=nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],out_channels=16,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592,256),
            nn.ReLU(),
            nn.Linear(256,num_actions)
        )


    def forward(self, x):
        # x: (batch, 4, 84, 84)
        q_values=self.stack(x)
        return q_values