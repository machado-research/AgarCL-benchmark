import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pfrl
from pfrl import agents, experiments, explorers, replay_buffers, utils
from pfrl.q_functions import DiscreteActionValueHead
from goBigUtils import GoBiggerObsFlatten, CustomMLP
import gym_agario
import wandb

class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 8 directions × 3 codes = 24 discrete actions
        dirs = [
            np.array([0, 1], dtype=np.float32),   # up
            np.array([1, 1], dtype=np.float32),   # up-right
            np.array([1, 0], dtype=np.float32),   # right
            np.array([1, -1], dtype=np.float32),  # down-right
            np.array([0, -1], dtype=np.float32),  # down
            np.array([-1, -1], dtype=np.float32), # down-left
            np.array([-1, 0], dtype=np.float32),  # left
            np.array([-1, 1], dtype=np.float32),  # up-left
        ]
        self.action_mappings = [(d, code) for d in dirs for code in range(3)]
        self.action_space = gym.spaces.Discrete(len(self.action_mappings))

    def action(self, action):
        assert 0 <= action < len(self.action_mappings)
        return self.action_mappings[action]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="agario-screen-v0")
    parser.add_argument("--outdir", type=str, default="./Results/DQN_gobigger")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--steps", type=int, default=5 * 10**6)
    parser.add_argument("--replay-start-size", type=int, default=10**4)
    parser.add_argument("--eval-n-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=50000)
    parser.add_argument("--n-best-episodes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--target-update-interval", type=int, default=10**4)
    parser.add_argument("--batch-accumulator", type=str, default="sum")
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="DQN_gobigger", config=args)

    utils.set_random_seed(args.seed)
    train_seed = args.seed
    test_seed = 2**31 - 1 - args.seed
    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    # load config for GoBigger
    env_config = json.load(open('env_config.json', 'r'))



    def make_env_(test):
        # Use different random seeds for train and test envs
        
        env_seed = test_seed if test else train_seed
        env_name = "agario-screen-v0"

        print( env_name)

        # env = make_env(env_name, env_config, gamma, norm_obs, norm_reward)
        env = gym.make(env_name, **env_config)
        env = DiscreteActions(env)
        env = GoBiggerObsFlatten(env,
                            max_food=env_config['num_pellets'],
                            max_virus=env_config['num_viruses'],
                            max_spore=env_config.get('num_spores', 0),
                            max_clone=env_config.get('num_bots', 0))
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, 0.5)
        
       
        return env


    def make_env(test):
        seed = test_seed if test else train_seed
        print( args.env)
        env = gym.make(args.env, **env_config)
        # wrap action & observation
        env = DiscreteActions(env)
        env = GoBiggerObsFlatten(env,
                                 max_food=env_config['num_pellets'],
                                 max_virus=env_config['num_viruses'],
                                 max_spore=env_config.get('num_spores', 0),
                                 max_clone=env_config.get('num_bots', 0))
        env = pfrl.wrappers.CastObservationToFloat32(env)
        env.reset(seed=seed)
        if test:
            env = pfrl.wrappers.RandomizeAction(env, 0.5)
        return env

    # instantiate envs and get dims
    env = make_env_(test=False)
    eval_env = make_env_(test=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # build Q-function with CustomMLP
    hidden_sizes = [256, 128]
    emb_dim = 64
    q_body = CustomMLP(obs_dim, hidden_sizes, emb_dim)
    q_func = nn.Sequential(
        q_body,
        nn.ReLU(),
        nn.Linear(emb_dim, n_actions),
        DiscreteActionValueHead(),
    )

    opt = torch.optim.Adam(q_func.parameters(), lr=args.lr, eps=1.5e-4)
    rbuf = replay_buffers.ReplayBuffer(100000)
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,
        end_epsilon=0.1,
        decay_steps=10**6,
        random_action_func=lambda: np.random.randint(n_actions),
    )

    # identity phi (already float32)
    def phi(x):
        return x

    agent = agents.DQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        clip_delta=True,
        update_interval=4,
        batch_accumulator=args.batch_accumulator,
        minibatch_size=args.minibatch_size,
        soft_update_tau=args.tau,
        n_times_update=args.epochs,
        phi=phi,
    )

    # load/demo/train logic follows unchanged...
    # ...

if __name__ == "__main__":
    main()
