
import numpy as np
import gymnasium as gym 
from utils.utils import atari_state_preprocess_function

def get_action_dim(action_space: gym.spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        # Number of discrete actions
        return len(action_space.nvec)
    elif isinstance(action_space, gym.spaces.MultiBinary):
        # Number of binary actions
        return action_space.n
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


class ReplayBuffer:
    """Replay buffer used to store and sample transitions when training an off-policy model.
    Here we use the implementation of memory efficient replay buffer."""
    def __init__(self,observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 buffer_size: int,
                 num_envs: int):
        
        self.buffer_size = buffer_size//num_envs
        self.action_dim = get_action_dim(action_space)
        self.num_envs = num_envs

        self.observation_space = observation_space 
        self.action_space = action_space
        self.states = np.zeros((buffer_size, self.num_envs, *observation_space.shape), dtype=observation_space.dtype) #TODO : check the type of observation space, if it's not gym.spaces.Box, it may raise an error.
        self.actions = np.zeros((buffer_size, self.num_envs, self.action_dim), dtype=action_space.dtype)
        
        self.rewards = np.zeros((buffer_size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        # TODO: try to understand why there is no next_states buffer
        # since the replay buffer is a circular buffer, (pos+1)%self.buffer_size is the next_state, so we don't need to store next_state separately.
        # in fact, this may introduce one transition can't be used, so the real buffer size is buffer_size-1
        self.pos = 0
        self.full = False 


    def add(self,batch: dict[str, np.ndarray]):
        """Add a batch of transitions to the replay buffer."""
        
        assert "states" in batch and "actions" in batch and "rewards" in batch and "next_states" in batch and "dones" in batch, "Batch must contain states, actions, rewards, next_states and dones."
        # unpack the batch
        s, a , ns, r, d = batch['states'], batch['actions'], batch['next_states'], batch['rewards'], batch['dones']

        batch_size,num_envs = s.shape[0],s.shape[1]   # the first dimension is the batch size, the second dimension is the number of envs
        
        assert batch_size>=0, "Batch size must be non-negative."
        
        end = self.pos + batch_size

        if end <= self.buffer_size:
            # not crossing the boundary
            self.states[self.pos:end] = s
            self.actions[self.pos:end] = a
            self.rewards[self.pos:end] = r
            self.dones[self.pos:end] = d
        else:
            # crossing the boundary, we need to split the batch into two parts
            first_part = self.buffer_size - self.pos
            second_part = batch_size - first_part

            self.states[self.pos:] = s[:first_part]
            self.states[:second_part] = s[first_part:]

            self.actions[self.pos:] = a[:first_part]
            self.actions[:second_part] = a[first_part:]

            self.rewards[self.pos:] = r[:first_part]
            self.rewards[:second_part] = r[first_part:]

            self.dones[self.pos:] = d[:first_part]
            self.dones[:second_part] = d[first_part:]

            self.full = True
        if end>=self.buffer_size:
            self.full = True
        self.pos = end % self.buffer_size
        
        self.states[self.pos] = ns[-1] # set the next state of the last transition in the batch to the current position, so that we can get the next state when sampling.
    
    def _sample_from_indices(self, batch_indices: np.ndarray) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer.
        """
        # TODO: Try to implement n-step sample
        env_indices = np.random.randint(0, high=self.num_envs, size=(len(batch_indices),))
        
        return dict(
            states=self.states[batch_indices, env_indices, :],
            actions=self.actions[batch_indices, env_indices, :],
            next_states=self.states[(batch_indices + 1) % self.buffer_size, env_indices, :], # get the next state by adding 1 to the index, and taking modulo buffer size to handle the circular buffer.
            rewards=self.rewards[batch_indices, env_indices],
            dones=self.dones[batch_indices, env_indices],
        )

    def _get_indices(self, batch_size: int) -> np.ndarray:
        """Get a batch of indices to sample from the replay buffer.
        You may change this method to use prioritized experience replay or other sampling strategies."""
        # TODO: Try to implement n-step sample
        if self.full:
            batch_indices = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_indices = np.random.randint(0, self.pos, size=batch_size)
        
        return batch_indices

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer."""
        
        batch_indices = self._get_indices(batch_size)
        return self._sample_from_indices(batch_indices)
