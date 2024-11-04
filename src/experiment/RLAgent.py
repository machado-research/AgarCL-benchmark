from src.wrappers.gym import make_env
from importlib import import_module

def get_agent(name, use_jax):
    path = 'src.algorithms.jax' if use_jax else 'src.algorithms.torch'
    if name == 'SAC':
        mod = import_module(f'{path}.SAC')
    elif name == 'PPO':
        mod = import_module(f'{path}.PPO')
    else:
        raise ValueError(f'Unknown agent: {name}')
    
    return getattr(mod, name)

class RLAgent:
    def __init__(self, exp, seed, env_config, device, collector_config, render=False):
        self.env_config = env_config
        
        self.agent = exp.agent
        self.hypers = exp.get_hypers(seed)

        self.gamma = self.hypers['gamma']
        self.norm_obs = self.hypers['norm_obs']
        self.norm_reward = self.hypers['norm_reward']
        self.env = make_env(exp.env_name, env_config, self.gamma,
                            self.norm_obs, self.norm_reward)
        
        self.agent = get_agent(self.agent, exp.use_jax)
        self.agent = self.agent(env=self.env, seed=seed, device=device, hypers=self.hypers, collector_config=collector_config,
                                total_timesteps=exp.total_steps, render=render)
        
        
    def train(self):
        return self.agent.train()

    def eval(self, obs):
        return self.agent.eval(obs)
    
    def get_collector(self):
        return self.agent.collector

    def save_collector(self, exp, save_path):
        self.agent.save_collector(exp, save_path)

    def load_checkpoint(self, path):
        self.agent.load_checkpoint(path)
    
    def save_checkpoint(self, path):
        self.agent.save_checkpoint(path)
    