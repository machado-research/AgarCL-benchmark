import gymnasium as gym
import numpy as np
from PyExpUtils.collection.Collector import Collector
from src.wrappers.gym import make_env

import torch
import torch.nn as nn
import torch.optim as optim
import tyro
# from torch.distributions.normal import Normal
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
import gym_agario 
import json
from cleanRL_PPO.custom_cleanrl_utils.evals.ppo_eval import evaluate
from cleanRL_PPO.custom_cleanrl_utils.huggingface import push_to_hub

class PPO_CleanRL:
    def __init__(self,
                 env: gym.Env,
                 seed: int,
                 device: str,
                 hypers: dict,
                 collector: Collector = None,
                 total_timesteps: int = 1e6,
                 eval_steps: int = 3000,
                 ) -> None:

        self.collector = collector
        self.total_timesteps = total_timesteps
        self.eval_steps = eval_steps

        # Hyperparameters
        self.num_steps = hypers['num_steps']
        self.anneal_lr = hypers['anneal_lr']
        self.learning_rate = hypers['learning_rate']
        self.gamma = hypers['gamma']
        self.gae_lambda = hypers['gae_lambda']
        self.num_minibatches = hypers['num_minibatches']
        self.update_epochs = hypers['update_epochs']
        self.norm_adv = hypers['norm_adv']
        self.clip_coef = hypers['clip_coef']
        #Need to be added in hypers
        self.hybrid_action = hypers['hybrid_action']
        self.clip_vloss = hypers['clip_vloss']
        self.ent_coef = hypers['ent_coef']
        self.vf_coef = hypers['vf_coef']
        self.max_grad_norm = hypers['max_grad_norm']
        self.target_kl = hypers.get('target_kl', None)

        self.env = env
        self.device = device


    def get_action_and_value(self, x, action=None):
        features = self.actor_base(x)
        
        # Continuous actions
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return torch.tanh(action), probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    
    def get_hybrid_action_and_value(self, x, action=None):
        features = self.actor_base(x)
        
        # Continuous actions
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)  # Ensure std is positive
        continuous_dist = Normal(action_mean, action_std)
        
        # Discrete actions
        action_logits = self.actor_discrete(features)
        discrete_dist = Categorical(logits=action_logits)

        if action is None:
            continuous_action = continuous_dist.sample()
            discrete_action = discrete_dist.sample()
        else:
            action = action.reshape(-1,action.shape[-1])
            continuous_action, discrete_action = action[:, :2], action[:, 2]
        
        #Clip continuous action to the valid range [-1,1]
        continuous_action = torch.tanh(continuous_action)
        # Compute log probabilities
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(1)
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        log_prob = continuous_log_prob + discrete_log_prob  # Sum log probabilities
        
        # Compute entropy for PPO updates
        entropy = continuous_dist.entropy().sum(1) + discrete_dist.entropy()

        return (continuous_action, discrete_action), log_prob, entropy, self.critic(x)



    def train(self, time_steps: int = None):
        raise NotImplementedError

