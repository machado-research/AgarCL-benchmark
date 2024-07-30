# import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # deal with dm_control's Dict observation space
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))
        self.action_dim = action_shape[0]

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_limits=None):
        action_mean = self.actor_mean(x).reshape(-1, self.action_dim)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if action_limits:
            action = np.clip(action, action_limits[0], action_limits[1])

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class HybridAgent(nn.Module):
    def __init__(self, obs_shape, cont_action_shape, dis_action_shape):
        print(obs_shape, cont_action_shape, dis_action_shape)
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(cont_action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(cont_action_shape)))
        self.action_dim = cont_action_shape
        
        self.dis_actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, dis_action_shape), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_limits=None):
        action_mean = self.actor_mean(x).reshape(-1, self.action_dim)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cont_probs = Normal(action_mean, action_std)
        
        disc_action_mean = self.dis_actor_mean(x)
        disc_probs = Categorical(logits=disc_action_mean)
        
        if action is None:
            cont_action = cont_probs.sample()
            dis_action = disc_probs.sample((1, 1))
        else:
            cont_action, dis_action = action[:, :-1], action[:, -1]
            
        if action_limits:
            cont_action = np.clip(cont_action, action_limits[0], action_limits[1])
        
        cont_action_log_prob = cont_probs.log_prob(cont_action).sum(-1, keepdim=True)
        cont_action_entropy = cont_probs.entropy().sum(-1, keepdim=True)
        
        dis_action_log_prob = disc_probs.log_prob(dis_action).sum(-1, keepdim=True)
        dis_action_entropy = disc_probs.entropy().sum(-1, keepdim=True)
        
        if action is None:
            action = torch.cat([cont_action, dis_action], 1)
        
        log_prob = cont_action_log_prob + dis_action_log_prob
        entropy = cont_action_entropy + dis_action_entropy
        
        return action, log_prob, entropy, self.critic(x)
