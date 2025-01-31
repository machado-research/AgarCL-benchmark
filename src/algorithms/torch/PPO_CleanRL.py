import gymnasium as gym
import numpy as np
from PyExpUtils.collection.Collector import Collector

import torch
import torch.nn as nn
import torch.optim as optim
import tyro
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

    def train(self, time_steps: int = None):
        raise NotImplementedError

