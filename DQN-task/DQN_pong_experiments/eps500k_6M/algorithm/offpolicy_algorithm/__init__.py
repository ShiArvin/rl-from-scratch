from .dqn import DQN 
from .baseoffpolicy import OffPolicyAlgorithm
ALGORITHM_DICT = {
    "OffPolicy": OffPolicyAlgorithm,
    "DQN": DQN,
}