from src.wrappers.gym import make_env
from importlib import import_module
import gymnasium as gym
import numpy as np

def get_agent(name, use_jax):
    path = 'src.algorithms.jax' if use_jax else 'src.algorithms.torch'
    if name in ['DQN', 'SAC', 'PPO']:
        mod = import_module(f'{path}.{name}')
    else:
        raise ValueError(f'Unknown agent: {name}')

    return getattr(mod, name)


class RLAgent:
    def __init__(self, exp, seed, env_config, device, collector, render=False):
        self.env_config = env_config

        self.agent = exp.agent
        self.hypers = exp.get_hypers(seed)

        self.gamma = self.hypers['gamma']
        self.norm_obs = self.hypers['norm_obs']
        self.norm_reward = self.hypers['norm_reward']
        self.env = gym.make(exp.env_name, **env_config)
        self.env.seed(seed)

        self.agent = get_agent(self.agent, exp.use_jax)
        self.agent = self.agent(env=self.env, seed=seed, device=device, hypers=self.hypers, collector=collector,
                                total_timesteps=exp.total_steps, eval_steps=exp.eval_steps)

    def train(self):
        return self.agent.train()

    def eval(self, obs: np.ndarray = None, save_path: str = None):
        return self.agent.eval(obs, save_path)

    def load_checkpoint(self, path: str):
        self.agent.load_checkpoint(path)

    def save_checkpoint(self, path: str):
        self.agent.save_checkpoint(path)
