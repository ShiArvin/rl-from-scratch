from .basealgorithm import BaseAlgorithm
from .baseoffpolicy import OffPolicyAlgorithm
from .baseonpolicy import OnPolicyAlgorithm
from .offline import OfflineAlgorithm


from .dqn import DQN 
from .trpo import TRPO 
from .ppo import PPO
from .sac import SAC
from .bc import BehaviorCloning
from .td3 import TD3


ALGORITHM_DICT = {
    "DQN": DQN,
    "TRPO":TRPO,
    "PPO": PPO,
    "SAC": SAC,
    "BC": BehaviorCloning,
    "TD3": TD3
}
