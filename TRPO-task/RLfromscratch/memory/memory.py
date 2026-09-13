
import numpy as np
import gymnasium as gym 
from utils.utils import get_action_dim



class ReplayBuffer:
    """Replay buffer used to store and sample transitions when training an off-policy model.
    Here we use the implementation of memory efficient replay buffer."""
    def __init__(self,observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 buffer_size: int,
                 num_envs: int, 
                 onpolicy=False):
        
        self.buffer_size = buffer_size
        action_dim = get_action_dim(action_space)
        self.num_envs = num_envs
        self.buffer_message = {
            "states": {"shape":(num_envs, buffer_size, *observation_space.shape), "dtype":observation_space.dtype} ,
            "actions": {"shape": (self.num_envs, buffer_size, action_dim), "dtype": action_space.dtype},
            "dones": {"shape": (self.num_envs, buffer_size), "dtype": np.float32},
            "rewards": {"shape": (self.num_envs, buffer_size), "dtype": np.float32},
        }
        self.onpolicy = onpolicy
        if onpolicy:
            self.buffer_message['log_probs'] = {"shape": (self.num_envs, buffer_size), "dtype": np.float32}
        
        self.buffer = {}
    
        # since the replay buffer is a circular buffer, (pos+1)%self.buffer_size is the next_state, so we don't need to store next_state separately.
        # in fact, this may introduce one transition can't be used, so the real buffer size is buffer_size-1
        self.pos = 0
        self.full = False 
        self.reset()


    def add(self,batch: dict[str, np.ndarray]):
        """Add a batch of transitions to the replay buffer."""
        # off-policy 
        if self.onpolicy:
            assert not self.full

        s = batch['states']

        num_envs,batch_size = s.shape[0],s.shape[1]   # the first dimension is the batch size, the second dimension is the number of envs
    
        end = self.pos + batch_size

        if end <= self.buffer_size:
            # not crossing the boundary
            for key in self.buffer:
                self.buffer[key][:,self.pos:end] = batch[key]
        else:
            # crossing the boundary, we need to split the batch into two parts
            first_part = self.buffer_size - self.pos
            second_part = batch_size - first_part
            for key in self.buffer:
                self.buffer[key][:,self.pos:] = batch[key][:,:first_part]
                self.buffer[key][:,:second_part] = batch[key][:,first_part:]

        if end>=self.buffer_size:
            self.full = True
        self.pos = end % self.buffer_size
        if not self.onpolicy:
            self.buffer['states'][:,self.pos] = batch['next_states'][:,-1] # set the next state of the last transition in the batch to the current position, so that we can get the next state when sampling.
       
       

    def _sample_from_indices(self, batch_indices: np.ndarray) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer.
        TODO: add more states preprocess to other environments 
        """
        env_indices = np.random.randint(0, high=self.num_envs, size=(len(batch_indices),))
        # this sample method is based one the dot product of env_indices and batch_indices 
        batch = {}
        for key in self.buffer:
            batch[key] = self.buffer[key][env_indices, batch_indices]
        batch['next_states'] = self.buffer['states'][env_indices, (batch_indices + 1) % self.buffer_size]

        return batch

    def _get_indices(self, batch_size: int) -> np.ndarray:
        """Get a batch of indices to sample from the replay buffer.
        You may change this method to use prioritized experience replay or other sampling strategies."""
        if self.full:
            batch_indices = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_indices = np.random.randint(0, self.pos, size=batch_size)
        
        return batch_indices

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer."""
        # This is only used for off-policy
        
        batch_indices = self._get_indices(batch_size)
        return self._sample_from_indices(batch_indices)


    def reset(self):
        """ clear the replay buffer."""
        for key, message in self.buffer_message.items():
            if key != "dones":
                self.buffer[key] = np.zeros(shape=message['shape'],dtype=message['dtype'])
            else:
                self.buffer[key] = np.ones(shape=message['shape'],dtype=message['dtype'])

        self.pos = 0
        self.full = False 


class TrajectoryRollout:
    def __init__(self, observation_space: gym.spaces.Box, 
                 action_space: gym.spaces.Space,
                 trajnum: int,
                 max_episode_length: int,
                 gamma: float):
        self.trajnum = trajnum 
        self.max_episode_length = max_episode_length
        self.action_dim = get_action_dim(action_space)
        self.gamma = gamma

        self.observation_space = observation_space 
        self.action_space = action_space
        # TODO: use dict to manage these values to avoid miss them in the reset
        self.states = np.zeros((trajnum, max_episode_length, *observation_space.shape), dtype=observation_space.dtype) 
        self.actions = np.zeros((trajnum, max_episode_length, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((trajnum, max_episode_length), dtype=np.float32)
        self.dones = np.ones((trajnum, max_episode_length), dtype=np.float32) # This is specifically designed to calculate the rewards to go 
        self.masks = np.zeros((trajnum,max_episode_length),dtype=np.float32)
        self.log_probs = np.zeros((trajnum,max_episode_length),dtype=np.float32)
        self.pos = 0
        self.full = False 
    
    def add_traj(self, traj_batch: dict):
        if self.full:
            raise ValueError("Trajectory Rollout is full, you need to clear the rollout first.")
        s, a, r, d, lp = traj_batch['states'], traj_batch['actions'], traj_batch['rewards'], traj_batch['dones'], traj_batch['log_probs']
        traj_length = len(s)
    
        
        self.states[self.pos,:traj_length] = s
        self.actions[self.pos,:traj_length] = a
        self.rewards[self.pos,:traj_length] = r 
        self.dones[self.pos,:traj_length] = d
        self.log_probs[self.pos, :traj_length] = lp
        self.masks[self.pos,:traj_length] = 1
        self.pos += 1
        if self.pos>=self.trajnum:
            self.full = True

    
    def reset(self):
        """ clear the replay buffer."""
        trajnum = self.trajnum 
        max_episode_length = self.max_episode_length
        observation_space = self.observation_space
        action_space = self.action_space
        self.states = np.zeros((trajnum, max_episode_length, *observation_space.shape), dtype=observation_space.dtype) 
        self.actions = np.zeros((trajnum, max_episode_length, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((trajnum, max_episode_length), dtype=np.float32)
        self.dones = np.ones((trajnum, max_episode_length), dtype=np.float32)
        self.masks = np.zeros((trajnum,max_episode_length),dtype=np.float32)
        self.log_probs = np.zeros((trajnum,max_episode_length),dtype=np.float32)
        
        self.pos = 0
        self.full = False 
    

            



