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
        C, H, W = input_shape

        self.conv1 = nn.Conv2d(C, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        
        with torch.no_grad():  
            dummy = torch.zeros(1, *input_shape)  
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            linear_input_size = x.view(1, -1).size(1)  

        self.fc1 = nn.Linear(linear_input_size, 256)
        
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x


# This network is used in TRPO
class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[30,], activation=nn.ReLU):
        super(MLPNetwork, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(activation())
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, input_dim)
        return self.net(x)
    

class Actor(nn.Module): # This is designed in TRPO, in other algorithm there are more to implement
    def __init__(self, state_dim, action_dim, hidden_sizes=[30,], activation=nn.ReLU):
        super(Actor, self).__init__()
        self.mu = MLPNetwork(state_dim, action_dim, hidden_sizes, activation)
        self.log_sigma =  nn.Parameter(torch.zeros(size=(1,action_dim))) 
    
    def forward(self,x):
        mu = self.mu(x)
        sigma = torch.exp(self.log_sigma).expand_as(mu)
        return mu, sigma