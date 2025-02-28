import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    def __init__(self, collector, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.collector = collector
        
        self.total_steps = 0
        self.cumulative_reward = 0
        self.last_obs = None
        self.sub_reward = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        #TODO: add the measurements here
        reward = float(self.locals['rewards'])
        self.last_obs = self.locals['new_obs'].squeeze(1)
        done = bool(self.locals['dones'])
        
        if done:
            self.collector.next_frame()
            self.collector.collect('episode_reward', self.episode_reward)
            self.episode_reward = 0

        if self.total_steps % 100 == 0:
            self.collector.next_frame()
            self.collector.collect('steps', 100)
            self.collector.collect('reward', self.sub_reward)
            self.sub_reward = 0

        self.cumulative_reward += reward
        self.sub_reward += reward
        self.episode_reward += reward
        self.total_steps += 1

        return True

    def get_avg_reward(self):
        return self.cumulative_reward / self.total_steps
    
    def get_last_obs(self):
        return self.last_obs
