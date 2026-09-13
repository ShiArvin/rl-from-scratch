from utils.result import Result
import torch
import numpy as np 
from omegaconf import DictConfig

from agent.agent import AgentBase
from .utils import to_useful_action, get_action_dim
from env.make_envs import make_vec_envs


def evaluate(agent:AgentBase, test_episodes:int, env_args:DictConfig, seed:int):
    envs = make_vec_envs(env_args,False,seed=seed, scale=False)
    agent.eval()
    states = envs.reset()
    episode_rewards = []
    action_dim = get_action_dim(envs.action_space)
    with Result(head="test") as result:
        while len(episode_rewards)<test_episodes:
            with torch.no_grad():
                actions, action_infos = agent.select_action(states,deterministic=True) # test with deterministic policy
            next_states, rewards, terminateds, infos = envs.step(to_useful_action(envs.action_space, action_dim, actions))
            
            for i in range(envs.num_envs):
                if len(episode_rewards)>=test_episodes:
                    break
                if 'episode' in infos[i]:
                    episode_rewards.append(infos[i]['episode']['r'])
            states = next_states

    result.add_metric("episode_rewards_mean", np.mean(episode_rewards))
    result.add_metric("episode_rewards_std", np.std(episode_rewards))
    envs.close()
    return result, np.mean(episode_rewards)
        

# def evaluate_mujoco(agent:AgentBase, test_episodes:int, env_args:DictConfig, seed:int):
#     # TODO