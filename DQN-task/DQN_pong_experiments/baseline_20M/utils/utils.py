import torch
import numpy as np
import gymnasium as gym
import random
from typing import Union

def get_best_device():

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")
    else:
        num_gpus = torch.cuda.device_count()
        max_free = -1
        best_gpu = 0

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
        
            total_memory = props.total_memory
            
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
           
            free_memory = total_memory - (allocated + reserved)

            print(f"GPU {i}: total {total_memory/1e6:.0f}MB, free {free_memory/1e6:.0f}MB")

            if free_memory > max_free:
                max_free = free_memory
                best_gpu = i

        device = torch.device(f"cuda:{best_gpu}")
        print(f"Using GPU {best_gpu} with approx {max_free/1e6:.0f} MB free memory.")
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
    if isinstance(observation_space,gym.spaces.Box) and len(observation_space.shape)==3 and states.dtype==np.uint8: #  and observation_space.low==0 and observation_space.high==255: this condition is for atari environment (4,84,84) unit8
        return states.astype(np.float32) / 255.0
    else:
        return states


def atari_to_useful_action(actions: np.ndarray):
        if len(actions.shape)==1:
            return actions 
        elif len(actions.shape)==2:
            return actions.squeeze(1)
        else:
            raise ValueError("actions' shape is more than 3 dims")

if __name__ == "__main__":
    print(get_best_device())