import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np 
from gymnasium.spaces import Space 
from torch.distributions import Normal, TransformedDistribution, Independent, Categorical
from torch.distributions.transforms import AffineTransform, TanhTransform
import math

EPS = 1e-8


def layer_init(layer, std=np.sqrt(2), bias_const=0.0, initialize=True):
    if not initialize:
        return layer
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class AtariDQNNetwork(nn.Module):
    def __init__(self, input_shape:Union[tuple, list], num_actions, initialize=False, dueling_network=False, conv_gradient_rescale=False):
        """
        input_shape: the shape of the input figure (C, H, W), usually (4, 84, 84)
        num_actions: Atari's action space is Discrete
        """
        super(AtariDQNNetwork, self).__init__()
        C, H, W = input_shape

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(C, 32, kernel_size=8, stride=4),initialize=initialize),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2),initialize=initialize),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1),initialize=initialize),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():  
            dummy = torch.zeros(1, *input_shape)  
            x = self.conv(dummy)
            linear_input_size = x.size(1)  

        self.fc = nn.Sequential(
            layer_init(nn.Linear(linear_input_size, 512),initialize=initialize),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions),initialize=initialize, std=1)
        )
        self.dueling = dueling_network 
        self.conv_gradient_rescale = conv_gradient_rescale
        if self.dueling:
            self.v = nn.Sequential(
                layer_init(nn.Linear(linear_input_size, 512),initialize=initialize),
                nn.ReLU(),
                layer_init(nn.Linear(512, 1),initialize=initialize, std=1)
            )



    def forward(self, x: torch.Tensor):
        # x: (batch, 4, 84, 84)
        x = self.conv(x)

        if self.training and x.requires_grad and self.conv_gradient_rescale:
            x.register_hook(lambda grad: grad / math.sqrt(2.0))

        if self.dueling:
            advan = self.fc(x)
            v = self.v(x)
            q = v + advan - torch.mean(advan, dim=1, keepdim=True)
        else:
            q = self.fc(x)

        return q

class AtariCNNEncoder(nn.Module):
    def __init__(self, input_shape:Union[tuple, list], output_dim:int, initialize=False):
        """
        input_shape: the shape of the input figure (C, H, W), usually (4, 84, 84)
        num_actions: Atari's action space is Discrete
        """
        super(AtariCNNEncoder, self).__init__()
        C, H, W = input_shape

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(C, 32, kernel_size=8, stride=4),initialize=initialize),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2),initialize=initialize),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1),initialize=initialize),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():  
            dummy = torch.zeros(1, *input_shape)  
            x = self.conv(dummy)
            linear_input_size = x.size(1)  

        self.fc = nn.Sequential(
            layer_init(nn.Linear(linear_input_size, output_dim),initialize=initialize),
            nn.ReLU()
        )
        self.embedding_dim = output_dim


    def forward(self, x):
        # x: (batch, 4, 84, 84)
        x = self.fc(self.conv(x))
        return x 



class MLPNetwork(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_sizes=[30,], activation=nn.ReLU,initialize=False, last_std=0.01):
        super(MLPNetwork, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_dim, hidden_size),initialize=initialize))
            layers.append(activation())
            last_dim = hidden_size
        
        layers.append(layer_init(nn.Linear(last_dim, output_dim),initialize=initialize, std=last_std))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, input_dim)
        return self.net(x)


    

class Actor(nn.Module): # This is designed for TRPO, in other algorithm there are more or less to implement
    def __init__(self, observation_space:Space, action_space:Space):
        super(Actor, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def forward(self,x):
        raise NotImplementedError

    def get_action(self, states:np.ndarray,deterministic:bool):
        raise NotImplementedError

    def get_dist(self, states:np.ndarray):
        raise NotImplementedError

    def get_log_prob(self, states:np.ndarray, actions:np.ndarray):
        raise NotImplementedError
    

class DiscreteProbabilityActor(Actor):
    """This actor use encoder and mlp decoder, and outputs discrete action from Categorical Distribution"""
    def __init__(self, observation_space:Space, action_space:Space, encoder:nn.Module,initialize=False):
        super(DiscreteProbabilityActor, self).__init__(observation_space, action_space)

        self.encoder = encoder 
        self.fc = MLPNetwork(
            input_dim=encoder.embedding_dim,
            output_dim=action_space.n,
            hidden_sizes=[],
            initialize=initialize,
            last_std=0.01
            )
    
    def forward(self, x:torch.Tensor):
        return self.fc(self.encoder(x))

    def get_dist(self, states:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        output: torch.distributions.Independent(Normal(mu,std), 1)
        """
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        return dist 
    
        
    def get_action(self, states:torch.Tensor,deterministic:bool):
        """
        states: torch.Tensor:(batch, state_dim)
        deterministic: bool
        output: actions:torch.Tensor:(batch, action_dim), log_probs:torch.Tensor:(batch,)
        note: the actions is not clipped, each of the output can compute the gradient
        """
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.unsqueeze(1), log_probs

    
    def get_log_prob(self, states:torch.Tensor, actions:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        output: torch.Tensor:(batch,)
        """
        assert actions.shape[1]==1
        actions = actions.squeeze(1)
        dist = self.get_dist(states)
        log_probs = dist.log_prob(actions)
        return log_probs
    
    def get_entropy(self, states:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        output: torch.Tensor:(batch,)
        """
        dist = self.get_dist(states)
        return dist.entropy().mean()



class DiagGaussianActor(Actor):
    # TODO: add image input class
    """
    use network to compute the mean and standard deviation of the Gaussian distribution
    """
    def __init__(self, observation_space:Space, 
                 action_space:Space, 
                 net_architecture:nn.Module,
                 device="cpu",
                 state_dependent_std=False, 
                 clip_sigma=True,
                 hidden_sizes=[64, 64], 
                 activation=torch.nn.Tanh, 
                 initialize=False):
        super(DiagGaussianActor, self).__init__(observation_space, action_space)

        self.mu = net_architecture(observation_space.shape[0], action_space.shape[0], hidden_sizes,activation,initialize, 0.01)
        # you can use network to compute the log_sigma according to the state
        self.state_dependent_std = state_dependent_std
        if state_dependent_std:
            self.log_sigma = net_architecture(observation_space.shape[0], action_space.shape[0], hidden_sizes,activation,initialize,0.01)
        else:
            self.log_sigma =  nn.Parameter(torch.zeros(size=(1,action_space.shape[0]))) 
        self.clip_sigma = clip_sigma
        self.device = device
        self.low, self.high = torch.tensor(self.action_space.low,device=self.device), torch.tensor(self.action_space.high, device=self.device)
        self.loc, self.scale = (self.low+self.high)/2, (self.high-self.low)/2


    def forward(self,x:torch.Tensor):
        mu = self.mu(x)
        if self.state_dependent_std:
            log_sigma = self.log_sigma(x)
        else:
            log_sigma = self.log_sigma
        if self.clip_sigma:
            log_sigma = log_sigma.clamp(min=-10, max=2)
        sigma = torch.exp(log_sigma)
        if not self.state_dependent_std:
            sigma = sigma.expand_as(mu)
        return mu, sigma
    
    def get_dist(self, states:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        output: torch.distributions.Independent(Normal(mu,std), 1)
        """
        mu,std = self.forward(states)
        dist = Independent(Normal(mu,std), 1) # this will be seen as (batch,) action_dim-dimension distributions instead of batch*action_dim 1-dimension distributions
        return dist
    
        
    def get_action(self, states:torch.Tensor,deterministic:bool):
        """
        states: torch.Tensor:(batch, state_dim)
        deterministic: bool
        output: actions:torch.Tensor:(batch, action_dim), log_probs:torch.Tensor:(batch,)
        note: the actions is not clipped, each of the output can compute the gradient
        """
        dist = self.get_dist(states) 
        
        if deterministic:
            actions = dist.base_dist.mean
        else:
            actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    
    def get_log_prob(self, states:torch.Tensor, actions:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        output: torch.Tensor:(batch,)
        """
        dist = self.get_dist(states)
        log_probs = dist.log_prob(actions)
        return log_probs
    
    def get_entropy(self, states:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        output: torch.Tensor:(batch,)
        """
        dist = self.get_dist(states)
        return dist.entropy().mean()

class TanhGaussianActor(Actor):
    def __init__(self, observation_space:Space, 
                 action_space:Space, 
                 net_architecture:nn.Module,
                 device="cpu",
                 state_dependent_std=False, 
                 clip_sigma=True, 
                 hidden_sizes=[64, 64], 
                 activation=torch.nn.Tanh, 
                 initialize=False):
        super(TanhGaussianActor, self).__init__(observation_space, action_space)

        self.mu = net_architecture(observation_space.shape[0], action_space.shape[0], hidden_sizes,activation,initialize, 0.01)
        # you can use network to compute the log_sigma according to the state
        self.state_dependent_std = state_dependent_std
        if state_dependent_std:
            self.log_sigma = net_architecture(observation_space.shape[0], action_space.shape[0], hidden_sizes,activation,initialize,0.01)
        else:
            self.log_sigma =  nn.Parameter(torch.zeros(size=(1,action_space.shape[0]))) 
        self.clip_sigma = clip_sigma
        self.device = device

        self.low, self.high = torch.tensor(self.action_space.low,device=self.device), torch.tensor(self.action_space.high, device=self.device)
        self.loc, self.scale = (self.low+self.high)/2, (self.high-self.low)/2


    def forward(self,x:torch.Tensor):
        mu = self.mu(x)
        if self.state_dependent_std:
            log_sigma = self.log_sigma(x)
        else:
            log_sigma = self.log_sigma
        if self.clip_sigma:
            log_sigma = log_sigma.clamp(min=-10, max=2)
        sigma = torch.exp(log_sigma)
        if not self.state_dependent_std:
            sigma = sigma.expand_as(mu)
        return mu, sigma
    
    def get_dist(self, states:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        output: torch.distributions.TransformedDistribution(Independent(Normal(mu,std), 1), [TanhTransform(cache_size=1), AffineTransform(loc=self.loc, scale=self.scale)])
        """
        mu,std = self.forward(states)
        raw_dist = Independent(Normal(mu,std), 1) # this will be seen as (batch,) action_dim-dimension distributions instead of batch*action_dim 1-dimension distributions
        dist = TransformedDistribution(raw_dist, [TanhTransform(cache_size=1), AffineTransform(loc=self.loc, scale=self.scale)]) # The notation wrote by torch is wrong: the order of the transformation is tanh->affine; but the notation is affine->tanh
        return dist

    
    def get_action(self, states:torch.Tensor,deterministic:bool):
        """
        states: torch.Tensor:(batch, state_dim)
        deterministic: bool
        output: actions:torch.Tensor:(batch, action_dim), log_probs:torch.Tensor:(batch,)
        note: the actions is clipped, each of the output can not compute the gradient
        """
        mu,std = self.forward(states)
        raw_dist = Independent(Normal(mu,std), 1)
        
        if deterministic:
            u = raw_dist.mean

        else:
            u = raw_dist.rsample()
        
        # clamp the action within action range and avoid the log_prob=nan
        actions = torch.tanh(u)*self.scale + self.loc
        log_probs = raw_dist.log_prob(u) - torch.log(1-torch.tanh(u).pow(2)+EPS).sum(-1, keepdim=False)-torch.log(self.scale).sum()
        return actions, log_probs
    
    
    def get_log_prob(self, states:torch.Tensor, actions:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        output: torch.Tensor:(batch,)
        """
        actions = torch.clamp(actions, self.low+EPS, self.high-EPS) # this is to avoid log_prob=nan
        dist = self.get_dist(states)
        log_probs = dist.log_prob(actions)
        return log_probs
    
    def get_entropy(self, states:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        output: torch.Tensor:(batch,)
        """
        mu,std = self.forward(states)
        raw_dist = Independent(Normal(mu,std), 1)
        return raw_dist.entropy().mean()


class DeterministicActor(Actor):
    def __init__(self, 
                 observation_space:Space, 
                 action_space:Space,
                 net_architecture:nn.Module,
                 device="cpu",

                 add_noise="normal",
                 explore_noise_sigma=0.1,

                 hidden_sizes=[64, 64], 
                 activation=torch.nn.ReLU, 
                 initialize=False):
        super(DeterministicActor, self).__init__(observation_space, action_space)
        
        self.net = net_architecture(observation_space.shape[0], action_space.shape[0], hidden_sizes,activation,initialize, 0.01)
 
        self.device = device

        self.add_noise = add_noise
        self.explore_noise_sigma = explore_noise_sigma



    def forward(self, x:torch.Tensor):
        return self.net(x)


    def get_action(self,states:np.ndarray,deterministic:bool):
        # return: (actions, action_info)
        actions = self.forward(states)
        if not deterministic:
            if self.add_noise=="normal":
                noise = torch.randn_like(actions) * self.explore_noise_sigma
            else:
                raise NotImplementedError
            actions = actions + noise

        return actions 

class TanhDeterministicActor(Actor):
    def __init__(self, 
                 observation_space:Space, 
                 action_space:Space,
                 net_architecture:nn.Module,
                 device="cpu",
                 add_noise="normal",
                 explore_noise_sigma=0.1,

                 hidden_sizes=[64, 64], 
                 activation=torch.nn.ReLU, 
                 initialize=False):
        super(TanhDeterministicActor, self).__init__(observation_space, action_space)
        
        self.net = net_architecture(observation_space.shape[0], action_space.shape[0], hidden_sizes,activation,initialize, 0.01)
 
        self.device = device

        self.add_noise = add_noise
        self.explore_noise_sigma = explore_noise_sigma

        self.low, self.high = torch.tensor(self.action_space.low,device=self.device), torch.tensor(self.action_space.high, device=self.device)
        self.loc, self.scale = (self.low+self.high)/2, (self.high-self.low)/2


    def forward(self, x:torch.Tensor):
        return self.net(x)


    def get_action(self,states:np.ndarray,deterministic:bool):
        # return: (actions, action_info)
        actions = self.scale*torch.tanh(self.forward(states)) + self.loc 
        if not deterministic:
            if self.add_noise=="normal":
                noise = torch.randn_like(actions) * self.explore_noise_sigma
            else:
                raise NotImplementedError
        
            actions = actions + noise
        return actions

        

class Critic(nn.Module):
    def __init__(self,observation_space:Space, action_space:Space=None):
        super(Critic, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def forward(self,x):
        raise NotImplementedError


class ValueFunction(Critic):
    def __init__(self, 
                 observation_space:Space, 
                 net_architecture:nn.Module,
                 hidden_sizes=[64, 64], 
                 activation=torch.nn.Tanh, 
                 initialize=False):
        super(ValueFunction, self).__init__(observation_space)

        self.net = net_architecture(observation_space.shape[0],1, hidden_sizes,activation,initialize, 1)
    
    def forward(self,states:torch.Tensor):
        return self.net(states)


class QFunction(Critic):
    def __init__(self, observation_space:Space, action_space:Space, net_architecture:nn.Module, **kwargs):
        super(QFunction, self).__init__(observation_space, action_space)

        self.net = net_architecture(observation_space.shape[0]+action_space.shape[0], 1, **kwargs)
    
    def forward(self,states:torch.Tensor, actions:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        output: (batch,1)
        """
        inputs = torch.concat([states, actions], dim=1)
        return self.net(inputs)

class DoubleQFunction(Critic):
    def __init__(self, observation_space:Space, action_space:Space, net_architecture:nn.Module, **kwargs):
        super(DoubleQFunction, self).__init__(observation_space, action_space)

        self.net1 = net_architecture(observation_space.shape[0]+action_space.shape[0], 1, **kwargs)
        self.net2 = net_architecture(observation_space.shape[0]+action_space.shape[0], 1, **kwargs)

    def forward(self,states:torch.Tensor, actions:torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        output: (2,batch,1)
        """
        inputs = torch.cat([states, actions], dim=1)
        return self.net1(inputs), self.net2(inputs)

class DiscreteVFunction(Critic):
    def __init__(self, observation_space:Space, action_space:Space, encoder:nn.Module,initialize=False):
        super(DiscreteVFunction, self).__init__(observation_space, action_space)

        self.encoder = encoder 
        self.fc = MLPNetwork(
            input_dim=encoder.embedding_dim,
            output_dim=1,
            hidden_sizes=[],
            initialize=initialize,
            last_std=1
            )
    
    def forward(self, x:torch.Tensor):
        return self.fc(self.encoder(x))
