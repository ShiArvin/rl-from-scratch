from utils.result import Result
import torch
import numpy as np 
from omegaconf import DictConfig

from agent.atari_agent import AgentBase
from .utils import atari_to_useful_action
from env.make_envs import make_vec_envs

def evaluate_atari(agent:AgentBase, test_episodes:int, env_args:DictConfig):
    """
    Evaluate the policy in the given environments.
    
    Args:
        agent: The RL agent to be evaluated.
        envs: Vec environments to evaluate on.
        test_episodes: Number of episodes to run for each environment.
    """
    envs = make_vec_envs(env_args,False,scale=False)
    agent.eval()
    states = envs.reset()
    episode_rewards = []
    with Result(head="test") as result:
        while len(episode_rewards)<test_episodes:
            with torch.no_grad():
                actions = agent.select_action(states,deterministic=True) # test with deterministic policy
            next_states, rewards, terminateds, infos = envs.step(atari_to_useful_action(actions))
            
            for i in range(envs.num_envs):
                if 'episode' in infos[i]:
                    episode_rewards.append(infos[i]['episode']['r'])
            states = next_states

    result.add_metric("episode_rewards_mean", np.mean(episode_rewards))
    result.add_metric("episode_rewards_std", np.std(episode_rewards))
    envs.close()
    return result
        
    