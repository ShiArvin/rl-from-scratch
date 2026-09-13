from .basealgorithm import BaseAlgorithm
from .baseoffpolicy import OffPolicyAlgorithm
from .baseonpolicy import OnPolicyAlgorithm


from .dqn import DQN 
from .trpo import TRPO 


ALGORITHM_DICT = {
    "DQN": DQN,
    "TRPO":TRPO 
}