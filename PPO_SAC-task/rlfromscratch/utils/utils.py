import torch
import numpy as np
import gymnasium as gym
import random
from typing import Union, Any 
import torch.nn as nn
from gymnasium.spaces import Discrete, Space




class RunningMeanStd:
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    :param mean: the initial mean estimation for data array. Default to 0.
    :param std: the initial standard error estimation for data array.
    :param clip_max: the maximum absolute value for data array. Default to
        10.0.
    :param epsilon: To avoid division by zero.
    """

    def __init__(
        self,
        mean: float | np.ndarray = 0.0,
        std: float | np.ndarray = 1.0,
        clip_max: float | None = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        self.mean, self.var = mean, std
        self.clip_max = clip_max
        self.count = 0
        self.eps = epsilon

    def norm(self, data_array: float | np.ndarray) -> float | np.ndarray:
        data_array = (data_array - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max:
            data_array = np.clip(data_array, -self.clip_max, self.clip_max)
        return data_array

    def update(self, data_array: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(data_array, axis=0), np.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count





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

def get_gpu_free_memory_mb(gpu_id: int) -> tuple[float, float]:
    """返回 (free_mb, total_mb)，基于整卡 NVML 统计。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = mem.free / (1024 ** 2)
        total_mb = mem.total / (1024 ** 2)
        return free_mb, total_mb
    except Exception:
        # 回退：nvidia-smi
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            text=True,
        )
        free_mb, total_mb = map(float, out.strip().split(","))
        return free_mb, total_mb

def get_best_device():
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")
    else:
        num_gpus = torch.cuda.device_count()
        max_free = -1
        best_gpu = 0

        for i in range(num_gpus):
            free_memory, total_memory = get_gpu_free_memory_mb(i)

            print(f"GPU {i}: total {total_memory:.0f}MB, free {free_memory:.0f}MB")

            if free_memory > max_free:
                max_free = free_memory
                best_gpu = i

        device = torch.device(f"cuda:{best_gpu}")
        print(f"Using GPU {best_gpu} with approx {max_free:.0f} MB free memory.")
    return device

def set_seed(seed):
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    




def atari_state_preprocess_function(observation_space:gym.spaces.Space, states:Union[np.ndarray,torch.Tensor]):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """
    if isinstance(states, np.ndarray):
        if isinstance(observation_space,gym.spaces.Box) and len(observation_space.shape)==3 and states.dtype==np.uint8: #  and observation_space.low==0 and observation_space.high==255: this condition is for atari environment (4,84,84) unit8
            return states.astype(np.float32) / 255.0
        else:
            return states
    elif isinstance(states, torch.Tensor):
        if isinstance(observation_space,gym.spaces.Box) and len(observation_space.shape)==3 and states.dtype==torch.uint8: #  and observation_space.low==0 and observation_space.high==255: this condition is for atari environment (4,84,84) unit8
            return states.float() / 255.0
        else:
            return states
    else:
        raise ValueError("states must be np.ndarray or torch.Tensor")


def to_useful_action(action_space:Space, action_dim:int, actions: np.ndarray):
    # this is used to transform the actions into envs' required action shape 
    if isinstance(action_space, Discrete):
        assert action_dim==1 
        if len(actions.shape)==1:
            return actions 
        elif len(actions.shape)==2:
            assert actions.shape[1]==1
            return actions.squeeze(1)
        else:
            raise ValueError("actions' shape is more than 3 dims")
    else:
        if len(actions.shape)==1:
            assert action_dim==actions.shape[0], "Actions shape mismatches with the Action dimension"
            return actions
        elif len(actions.shape)==2:
            assert action_dim==actions.shape[1], "Actions shape mismatches with the Action dimension"
            return actions
        elif len(actions.shape)==3:
            assert action_dim==actions.shape[2] and actions.shape[1]==1, "Actions shape mismatches with the Action dimension"
            return actions.reshape(-1,action_dim)
        else:
            raise ValueError("actions' shape is more than 3 dims")
        
def get_flat_grad(y: torch.Tensor, model: nn.Module, **kwargs: Any) -> torch.Tensor:
    grads = torch.autograd.grad(y, model.parameters(), **kwargs) 
    
    return torch.cat([grad.reshape(-1) for grad in grads])


def kl_product(vector: torch.Tensor, kl_grad:torch.Tensor, model: nn.Module, egienvalue_reg: float = 0.1):
    """ We compute \delta^2 kl@vector = \delta(\delta kl@vector) instead of calculate the \delta^2 kl because its complexity is O(n^2) """
    _sum = torch.sum(vector*kl_grad)
    product =  get_flat_grad(_sum, model, retain_graph=True).detach()
    return product + vector* egienvalue_reg # This is equal to add 0.1 to the eigenvalue of Hessian matrix





def conjugate_gradients(
        model:nn.Module,
        b: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-10,
        eginvalue_reg: float = 0.1
        
    ) -> torch.Tensor:
    x = torch.zeros_like(b)
    r, p = b.clone(), b.clone()

    rdotr = r.dot(r)
    for _ in range(nsteps):
        z = kl_product(p, flat_kl_grad,model,egienvalue_reg=eginvalue_reg)
        alpha = rdotr / p.dot(z)
        x += alpha * p
        r -= alpha * z
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x


def get_model_flat_parameters(model:nn.Module):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

def load_flat_parameters_to_model(flat_params:torch.Tensor, model:nn.Module):
    start_idx = 0
    for param in model.parameters():
        length = len(param.view(-1))
        end_idx = start_idx + length 

        param.data.copy_(flat_params[start_idx: end_idx].reshape(param.shape))
        start_idx = end_idx 

        





if __name__ == "__main__":
    print(get_best_device())