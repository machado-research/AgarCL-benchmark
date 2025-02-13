import argparse
import json
import os

import numpy as np
import torch.nn as nn
import gymnasium as gym
import pfrl
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
import gym_agario

import torch
from PyExpUtils.collection.Collector import Collector

from src.wrappers.gym import ModifyObservationWrapper, ModifyDiscreteActionWrapper
from src.utils.torch.networks import CustomCNN, phi




class PFRL_DQN:
    def __init__(self,
                 env: gym.Env,
                 seed: int,
                 device: str,
                 hypers: dict,
                 collector: Collector = None,
                 total_timesteps: int = 2 * 1e6,
                 eval_steps: int = 500,
                 ) -> None:

        self.collector = collector
        self.total_timesteps = total_timesteps
        self.eval_steps = eval_steps
        
        # self.save_path = hypers['save_path']
        self.lr = hypers['lr']
        self.buffer_size = int(hypers['buffer_size'])
        self.learning_starts = int(hypers['learning_starts'])
        self.batch_size = hypers['batch_size']
        self.hidden_size = hypers['hidden_size']
        self.gamma = hypers['gamma']
        self.eval_interval = hypers['eval_interval']
        self.device = device # 'cuda' if torch.cuda.is_available() else 'cpu'

        self.env = ModifyDiscreteActionWrapper(env)
        self.env = ModifyObservationWrapper(self.env)
        self.env.seed(seed)
        self.eval_env = pfrl.wrappers.RandomizeAction(env, 0.5)
        n_actions = env.action_space.n

        self.q_function = nn.Sequential(
            CustomCNN(n_input_channels=4, n_output_channels=256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            DiscreteActionValueHead(),
        )

        self.opt = torch.optim.Adam(self.q_func.parameters(), self.lr, eps=1.5 * 10**-4)
        self.rbuf = replay_buffers.ReplayBuffer(self.buffer_size)#1e5
        
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0,
            end_epsilon=0.1,
            decay_steps=total_timesteps,
            random_action_func=lambda: np.random.randint(n_actions),
        )

        self.agent = agents.DQN(
        self.q_func,
        self.opt,
        self.rbuf,
        gpu=self.device,
        gamma=self.gamma,
        explorer=self.explorer,
        replay_start_size=self.learning_starts,
        target_update_interval=10**4,
        clip_delta=True,
        update_interval=4,
        batch_accumulator="sum",
        phi=phi,
        )

        

    def train(self, time_steps: int = None, save_path: str = None):
        experiments.train_agent_with_evaluation(
            agent=self.agent,
            env=self.env,
            steps=time_steps,
            eval_n_steps=self.eval_steps,
            eval_n_episodes=None,
            eval_interval=self.eval_interval,
            outdir=save_path,
            save_best_so_far_agent=True,
            eval_env=self.eval_env,
            checkpoint_freq = 100000,
        )

    def eval(self, obs: np.ndarray, save_path: str):
        stats = experiments.evaluator.eval_performance(
            env=self.env,
            agent=self.agent,
            n_steps=None,
            n_episodes=5,
            max_episode_len=4500,
            logger=None,
        )

        with open(os.path.join(self.save_path, "bestscores.json"), "w") as f:
            json.dump(stats, f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))

    def save_checkpoint(self, path: str):
        pass

    def load_checkpoint(self, path: str):
        self. agent.load(path)
