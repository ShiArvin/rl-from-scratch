
import numpy as np
import random
import gymnasium as gym 
from utils.utils import get_action_dim
from .datastructure import SumTree



class ReplayBuffer:
    """Replay buffer used to store and sample transitions when training an off-policy model.
    Here we use the implementation of memory efficient replay buffer."""
    def __init__(self,observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 buffer_size: int,
                 num_envs: int, 
                 gamma:float=0.99,
                 nstep:int=1,
                 onpolicy=False
                 ):
        
        self.buffer_size = buffer_size
        action_dim = get_action_dim(action_space)
        self.num_envs = num_envs
        self.buffer_message = {
            "states": {"shape":(num_envs, buffer_size, *observation_space.shape), "dtype":observation_space.dtype} ,
            "actions": {"shape": (self.num_envs, buffer_size, action_dim), "dtype": action_space.dtype},
            "dones": {"shape": (self.num_envs, buffer_size), "dtype": np.float32},
            "rewards": {"shape": (self.num_envs, buffer_size), "dtype": np.float32},
            "next_states": {"shape": (self.num_envs, buffer_size, *observation_space.shape), "dtype": observation_space.dtype}
        }
        self.buffer_message['truncateds'] = {"shape": (self.num_envs, buffer_size), "dtype": np.float32}
        self.onpolicy = onpolicy
        if onpolicy:
            self.buffer_message['log_probs'] = {"shape": (self.num_envs, buffer_size), "dtype": np.float32}
            
        
        self.buffer = {}
    
        # we think the replay buffer as a big circle 
        self.pos = 0
        self.full = False 

        self.nstep = nstep
        self.gamma = gamma

        self.reset()


    def add(self,batch: dict[str, np.ndarray]):
        """Add a batch of transitions to the replay buffer."""
        # for onpolicy algorithm, we should not allow the cover of the old data
        if self.onpolicy:
            assert not self.full

        s = batch['states']

        num_envs,batch_size = s.shape[0],s.shape[1]   # the first dimension is the batch size, the second dimension is the number of envs
    
        end = self.pos + batch_size


        indices = (self.pos + np.arange(batch_size))%self.buffer_size

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

        return indices 

    def _is_outof_range(self, indices:np.ndarray):
        if self.full:
            return indices>=self.buffer_size 
        else:
            return indices>=self.pos
    
    def _loop_to_real(self, loop_indices:np.ndarray):
        if self.full:
            return (loop_indices+self.pos)%self.buffer_size 
        else:
            return loop_indices 
    
    def _compute_nstep(self,env_indices:np.ndarray, indices: np.ndarray, batch: dict):
        if self.full:
            loop_indices = indices - self.pos 
            loop_indices[loop_indices<0] += self.buffer_size 
        else:
            loop_indices = np.copy(indices)


        end_loop_indices = np.copy(loop_indices)
        returns = np.copy(batch['rewards'])
        
        for i in range(1,self.nstep):

            outof_range = self._is_outof_range(end_loop_indices+1)

            end_indices = self._loop_to_real(end_loop_indices)
            done = np.logical_or(self.buffer['dones'][env_indices, end_indices], \
                           self.buffer['truncateds'][env_indices, end_indices])
            
            end_loop_indices += 1
            end_loop_indices[np.logical_or(outof_range, done)] -= 1

            cur_indices = loop_indices + i
            outof_end = cur_indices > end_loop_indices
            cur_indices[outof_end] = 0 # random choose an index 
            returns += self.gamma**i * (1-outof_end) * self.buffer['rewards'][env_indices, self._loop_to_real(cur_indices)]

        end_real_indices = self._loop_to_real(end_loop_indices)
        dones = self.buffer['dones'][env_indices, end_real_indices]

        
        batch['rewards'] = returns 
        batch['dones'] = dones
        batch['next_states'] = self.buffer['next_states'][env_indices, end_real_indices]
        batch['nstep_gamma'] = self.gamma ** (end_loop_indices - loop_indices + 1)
        return batch 
       

    def _sample_from_indices(self, batch_indices: np.ndarray) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer.
        TODO: add more states preprocess to other environments 
        """
        env_indices = np.random.randint(0, high=self.num_envs, size=(len(batch_indices),))
        # this sample method is based one the dot product of env_indices and batch_indices 
        batch = {}
        for key in self.buffer:
            batch[key] = self.buffer[key][env_indices, batch_indices]
        
        batch = self._compute_nstep(env_indices, batch_indices, batch)
        

        return batch

    def _get_indices(self, batch_size: int) -> np.ndarray:
        """Get a batch of indices to sample from the replay buffer.
        You may change this method to use prioritized experience replay or other sampling strategies."""
        if self.full:
            batch_indices = np.random.randint(0, self.buffer_size, size=batch_size)
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


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    This implementation is based on the proportional prioritized replay buffer.
    """
    def __init__(self,observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 buffer_size: int,
                 num_envs: int,
                 alpha: float,
                 beta: float,
                 epsilon=1e-6,
                 batch_norm=False,
                 gamma=0.99,
                 nstep=1,
                 onpolicy=False):

        assert num_envs==1, "We only implement Prioritized Replaybuffer for num_envs==1"
        super(PrioritizedReplayBuffer, self).__init__(observation_space, action_space, buffer_size, num_envs, gamma, nstep, onpolicy)

        self.alpha = alpha 
        self.beta = beta 
        self.epsilon = epsilon
        self.priority = SumTree(buffer_size)

        self.max_td = 1.

        self.batch_norm = batch_norm

    def add(self,batch: dict[str, np.ndarray]):
        indices = super().add(batch)
        initial_priority = np.ones_like(indices,dtype=np.float64)*(self.max_td ** self.alpha)
        self.priority[indices] = initial_priority
        return indices

    
    def _get_indices(self,batch_size:int):
        # Note: Here we use the stratified sampling according to the implementation in RainBow
        
        p_sum = self.priority.tree[1]
        segment = p_sum / batch_size
        ratios = np.arange(batch_size,dtype=np.float64) * segment 
        ratios += np.random.uniform(0.0, segment, [batch_size])
        
        indices = self.priority.get_prefix_idx(ratios)
        return indices

    def _get_weight(self, indices:np.ndarray):
        # Note: We do the normalization inside the batch
        weights = (self.priority[indices])**(-self.beta)
        weights = weights / weights.max()
        return weights

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer."""
        # This is only used for off-policy
        
        batch_indices = self._get_indices(batch_size)
        batch = self._sample_from_indices(batch_indices)
        weights = self._get_weight(batch_indices)
        batch['weights'] = weights 
        batch['indices'] = batch_indices
        return batch 


    def update_priority(self, indices:np.ndarray, td_error:np.ndarray):
        assert indices.shape==td_error.shape
        td_abs = np.abs(td_error) + self.epsilon
        priorities = td_abs**self.alpha
        self.max_td = max(self.max_td, td_abs.max())
        self.priority[indices] = priorities
        


    def reset(self):
        super().reset()
        self.priority = SumTree(self.buffer_size)
        self.max_td = 1.
    

    

        


class TrajectoryRollout:
    def __init__(self, observation_space: gym.spaces.Box, 
                 action_space: gym.spaces.Space,
                 trajnum: int,
                 max_episode_length: int):
        self.trajnum = trajnum 
        self.max_episode_length = max_episode_length
        self.action_dim = get_action_dim(action_space)

        self.observation_space = observation_space 
        self.action_space = action_space
        # TODO: use dict to manage these values to avoid miss them in the reset
        self.states = np.zeros((trajnum, max_episode_length, *observation_space.shape), dtype=observation_space.dtype) 
        self.actions = np.zeros((trajnum, max_episode_length, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((trajnum, max_episode_length), dtype=np.float32)
        self.dones = np.ones((trajnum, max_episode_length), dtype=np.float32) # This is specifically designed to calculate the rewards to go 
        self.masks = np.zeros((trajnum, max_episode_length),dtype=np.float32)
        self.log_probs = np.zeros((trajnum,max_episode_length),dtype=np.float32)
        self.last_states = np.zeros((trajnum, *observation_space.shape), dtype=observation_space.dtype) # store the last states of each trajectory for calculating the value of the last states in the GAE calculation
        self.truncateds = np.zeros((trajnum, ), dtype=np.float32) # store the truncateds of each step for calculating the returns in the case of truncation
        self.pos = 0
        self.full = False 
    
    def add_traj(self, traj_batch: dict):
        if self.full:
            raise ValueError("Trajectory Rollout is full, you need to clear the rollout first.")
        s, a, r, d, lp, ls, tr = traj_batch['states'], traj_batch['actions'], traj_batch['rewards'], traj_batch['dones'], traj_batch['log_probs'], traj_batch['last_states'], traj_batch['truncateds']
        traj_length = len(s)
    
        
        self.states[self.pos,-traj_length:] = s
        self.actions[self.pos,-traj_length:] = a
        self.rewards[self.pos,-traj_length:] = r 
        self.dones[self.pos, -traj_length:] = d
        self.log_probs[self.pos, -traj_length:] = lp
        self.masks[self.pos,-traj_length:] = 1
        self.last_states[self.pos] = ls
        self.truncateds[self.pos] = tr
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
        self.dones = np.ones((trajnum, max_episode_length), dtype=np.float32) # This is specifically designed to calculate the rewards to go 
        self.masks = np.zeros((trajnum, max_episode_length),dtype=np.float32)
        self.log_probs = np.zeros((trajnum,max_episode_length),dtype=np.float32)
        self.last_states = np.zeros((trajnum, *observation_space.shape), dtype=observation_space.dtype) # store the last states of each trajectory for calculating the value of the last states in the GAE calculation
        self.truncateds = np.zeros((trajnum, ), dtype=np.float32) # store the truncateds of each step for calculating the returns in the case of truncation
        self.pos = 0
        self.full = False 
    




if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/ubuntu/wangchenyang/RLfromscratch")

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    passed, failed = 0, 0
    def check(cond, msg):
        global passed, failed
        if cond:
            passed += 1; print(f"  [PASS] {msg}")
        else:
            failed += 1; print(f"  [FAIL] {msg}")

    # ====== Test 1: nstep=3, no done/truncated, normal case ======
    print("=" * 60)
    print("Test 1: nstep=3, continuous trajectory (no done/truncated)")
    print("=" * 60)
    gamma = 0.99
    buf = ReplayBuffer(obs_space, act_space, buffer_size=10, num_envs=1, gamma=gamma, nstep=3)
    batch = {
        "states": np.arange(10).reshape(1, 10, 1).repeat(4, axis=2).astype(np.float32),
        "actions": np.zeros((1, 10, 1), dtype=np.int64),
        "rewards": np.array([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]], dtype=np.float32),
        "dones": np.zeros((1, 10), dtype=np.float32),
        "truncateds": np.zeros((1, 10), dtype=np.float32),
        "next_states": np.arange(1, 11).reshape(1, 10, 1).repeat(4, axis=2).astype(np.float32),
    }
    buf.add(batch)

    env_idx = np.array([0, 0, 0])
    sample_idx = np.array([0, 3, 5])
    sample_batch = {key: buf.buffer[key][env_idx, sample_idx] for key in buf.buffer}
    result = buf._compute_nstep(env_idx, sample_idx, sample_batch)

    # index 0: r0 + gamma*r1 + gamma^2*r2, next_state=state[2]
    exp0 = 1 + gamma*2 + gamma**2*3
    check(abs(result['rewards'][0] - exp0) < 1e-5, f"idx0 return: expected {exp0:.4f}, got {result['rewards'][0]:.4f}")
    check(result['next_states'][0, 0] == 3.0, f"idx0 next_state should be state[2]'s next=3, got {result['next_states'][0, 0]}")
    check(abs(result['nstep_gamma'][0] - gamma**3) < 1e-5, f"idx0 nstep_gamma should be gamma^3")

    # index 3: r3 + gamma*r4 + gamma^2*r5
    exp3 = 4 + gamma*5 + gamma**2*6
    check(abs(result['rewards'][1] - exp3) < 1e-5, f"idx3 return: expected {exp3:.4f}, got {result['rewards'][1]:.4f}")

    # ====== Test 2: nstep=3, done in the middle ======
    print("\n" + "=" * 60)
    print("Test 2: nstep=3, done truncates the n-step lookahead")
    print("=" * 60)
    buf2 = ReplayBuffer(obs_space, act_space, buffer_size=10, num_envs=1, gamma=gamma, nstep=3)
    batch2 = {
        "states": np.arange(10).reshape(1, 10, 1).repeat(4, axis=2).astype(np.float32),
        "actions": np.zeros((1, 10, 1), dtype=np.int64),
        "rewards": np.array([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]], dtype=np.float32),
        "dones": np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),  # done at index 1
        "truncateds": np.zeros((1, 10), dtype=np.float32),
        "next_states": np.arange(1, 11).reshape(1, 10, 1).repeat(4, axis=2).astype(np.float32),
    }
    buf2.add(batch2)

    env_idx2 = np.array([0])
    sample_idx2 = np.array([0])
    sample_batch2 = {key: buf2.buffer[key][env_idx2, sample_idx2] for key in buf2.buffer}
    result2 = buf2._compute_nstep(env_idx2, sample_idx2, sample_batch2)

    # index 0: done at index 1, so lookahead stops at index 1
    # return = r0 + gamma*r1, next_state = next_states[1]
    exp_done = 1 + gamma * 2
    check(abs(result2['rewards'][0] - exp_done) < 1e-5,
          f"done@1: return from idx0 should be {exp_done:.4f}, got {result2['rewards'][0]:.4f}")
    check(result2['next_states'][0, 0] == 2.0,
          f"done@1: next_state should be next_states[1]=2, got {result2['next_states'][0, 0]}")
    check(result2['dones'][0] == 1.0, f"done@1: final done flag should be 1.0, got {result2['dones'][0]}")
    check(abs(result2['nstep_gamma'][0] - gamma**2) < 1e-5,
          f"done@1: nstep_gamma should be gamma^2, got {result2['nstep_gamma'][0]}")

    # ====== Test 3: nstep=3, truncated in the middle ======
    print("\n" + "=" * 60)
    print("Test 3: nstep=3, truncated truncates the n-step lookahead")
    print("=" * 60)
    buf3 = ReplayBuffer(obs_space, act_space, buffer_size=10, num_envs=1, gamma=gamma, nstep=3)
    batch3 = {
        "states": np.arange(10).reshape(1, 10, 1).repeat(4, axis=2).astype(np.float32),
        "actions": np.zeros((1, 10, 1), dtype=np.int64),
        "rewards": np.array([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]], dtype=np.float32),
        "dones": np.zeros((1, 10), dtype=np.float32),
        "truncateds": np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),  # truncated at index 2
        "next_states": np.arange(1, 11).reshape(1, 10, 1).repeat(4, axis=2).astype(np.float32),
    }
    buf3.add(batch3)

    env_idx3 = np.array([0])
    sample_idx3 = np.array([1])
    sample_batch3 = {key: buf3.buffer[key][env_idx3, sample_idx3] for key in buf3.buffer}
    result3 = buf3._compute_nstep(env_idx3, sample_idx3, sample_batch3)

    # index 1: truncated at index 2, so end_index stops at 2
    # return = r1 + gamma*r2, next_state = next_states[2]
    exp_trunc = 2 + gamma * 3
   
    check(abs(result3['rewards'][0] - exp_trunc) < 1e-5,
          f"trunc@2: return from idx1 should be {exp_trunc:.4f}, got {result3['rewards'][0]:.4f}")
    check(result3['next_states'][0, 0] == 3.0,
          f"trunc@2: next_state should be next_states[2]=3, got {result3['next_states'][0, 0]}")
    check(result3['dones'][0] == 0.0,
          f"trunc@2: done flag should remain 0 (truncated != done), got {result3['dones'][0]}")

    # ====== Test 4: nstep=3, boundary (buffer not full, near end) ======
    print("\n" + "=" * 60)
    print("Test 4: nstep=3, near buffer boundary (pos < buffer_size)")
    print("=" * 60)
    buf4 = ReplayBuffer(obs_space, act_space, buffer_size=10, num_envs=1, gamma=gamma, nstep=3)
    short_batch = {
        "states": np.arange(5).reshape(1, 5, 1).repeat(4, axis=2).astype(np.float32),
        "actions": np.zeros((1, 5, 1), dtype=np.int64),
        "rewards": np.array([[1., 2., 3., 4., 5.]], dtype=np.float32),
        "dones": np.zeros((1, 5), dtype=np.float32),
        "truncateds": np.zeros((1, 5), dtype=np.float32),
        "next_states": np.arange(1, 6).reshape(1, 5, 1).repeat(4, axis=2).astype(np.float32),
    }
    buf4.add(short_batch)
    # buf4.pos = 5, buf4.full = False

    env_idx4 = np.array([0])
    sample_idx4 = np.array([4])  # last valid index
    sample_batch4 = {key: buf4.buffer[key][env_idx4, sample_idx4] for key in buf4.buffer}
    result4 = buf4._compute_nstep(env_idx4, sample_idx4, sample_batch4)

    # index 4: next index 5 is out of range (pos=5), so end_index stays at 4
    # return = r4 only (no lookahead), nstep_gamma = gamma^1
    check(abs(result4['rewards'][0] - 5.0) < 1e-5,
          f"boundary: return from idx4 should be 5.0 (no lookahead), got {result4['rewards'][0]:.4f}")
    check(abs(result4['nstep_gamma'][0] - gamma) < 1e-5,
          f"boundary: nstep_gamma should be gamma^1, got {result4['nstep_gamma'][0]}")

    # index 3: can look 1 step ahead (index 4 valid, index 5 out of range)
    sample_idx4b = np.array([3])
    env_idx4b = np.array([0])
    sample_batch4b = {key: buf4.buffer[key][env_idx4b, sample_idx4b] for key in buf4.buffer}
    result4b = buf4._compute_nstep(env_idx4b, sample_idx4b, sample_batch4b)

    exp_boundary = 4 + gamma * 5
    check(abs(result4b['rewards'][0] - exp_boundary) < 1e-5,
          f"boundary: return from idx3 should be {exp_boundary:.4f}, got {result4b['rewards'][0]:.4f}")
    check(abs(result4b['nstep_gamma'][0] - gamma**2) < 1e-5,
          f"boundary: nstep_gamma should be gamma^2, got {result4b['nstep_gamma'][0]}")

    # ====== Test 5: nstep=1, should just return original reward ======
    print("\n" + "=" * 60)
    print("Test 5: nstep=1, no lookahead (baseline check)")
    print("=" * 60)
    buf5 = ReplayBuffer(obs_space, act_space, buffer_size=10, num_envs=1, gamma=gamma, nstep=1)
    buf5.add(short_batch)

    env_idx5 = np.array([0, 0, 0])
    sample_idx5 = np.array([0, 2, 4])
    sample_batch5 = {key: buf5.buffer[key][env_idx5, sample_idx5] for key in buf5.buffer}
    result5 = buf5._compute_nstep(env_idx5, sample_idx5, sample_batch5)

    check(abs(result5['rewards'][0] - 1.0) < 1e-5, f"nstep=1: idx0 reward should be 1.0, got {result5['rewards'][0]}")
    check(abs(result5['rewards'][1] - 3.0) < 1e-5, f"nstep=1: idx2 reward should be 3.0, got {result5['rewards'][1]}")
    check(abs(result5['rewards'][2] - 5.0) < 1e-5, f"nstep=1: idx4 reward should be 5.0, got {result5['rewards'][2]}")

    # ====== Test 6: done at start index itself ======
    print("\n" + "=" * 60)
    print("Test 6: nstep=3, done at the sampled index itself")
    print("=" * 60)
    buf6 = ReplayBuffer(obs_space, act_space, buffer_size=10, num_envs=1, gamma=gamma, nstep=3)
    batch6 = {
        "states": np.arange(5).reshape(1, 5, 1).repeat(4, axis=2).astype(np.float32),
        "actions": np.zeros((1, 5, 1), dtype=np.int64),
        "rewards": np.array([[1., 2., 3., 4., 5.]], dtype=np.float32),
        "dones": np.array([[1, 0, 0, 0, 0]], dtype=np.float32),  # done at index 0
        "truncateds": np.zeros((1, 5), dtype=np.float32),
        "next_states": np.arange(1, 6).reshape(1, 5, 1).repeat(4, axis=2).astype(np.float32),
    }
    buf6.add(batch6)

    env_idx6 = np.array([0])
    sample_idx6 = np.array([0])
    sample_batch6 = {key: buf6.buffer[key][env_idx6, sample_idx6] for key in buf6.buffer}
    result6 = buf6._compute_nstep(env_idx6, sample_idx6, sample_batch6)

    # done at index 0 means end_index stays at 0
    check(abs(result6['rewards'][0] - 1.0) < 1e-5,
          f"done@start: return should be 1.0 (no lookahead), got {result6['rewards'][0]}")
    check(result6['dones'][0] == 1.0, f"done@start: done flag should be 1.0, got {result6['dones'][0]}")
    check(abs(result6['nstep_gamma'][0] - gamma) < 1e-5,
          f"done@start: nstep_gamma should be gamma^1, got {result6['nstep_gamma'][0]}")

    # ====== Summary ======
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    print("=" * 60)
