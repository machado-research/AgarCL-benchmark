import argparse
import json
import os

import numpy as np
import torch
from torch import nn
import gymnasium as gym

import pfrl
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.q_functions import DistributionalDuelingDQN
from pfrl.wrappers import atari_wrappers
from src.wrappers.gym import make_env

# from gym import spaces

class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)
        self.action_mappings = [
            np.array([0, 1], dtype=np.float32),   # Up
            np.array([1, 1], dtype=np.float32),   # Up-Right
            np.array([1, 0], dtype=np.float32),   # Right
            np.array([1, -1], dtype=np.float32),  # Down-Right
            np.array([0, -1], dtype=np.float32),  # Down
            np.array([-1, -1], dtype=np.float32), # Down-Left
            np.array([-1, 0], dtype=np.float32),  # Left
            np.array([-1, 1], dtype=np.float32),  # Up-Left
        ]

    def action(self, action):
        # Map the discrete action to (dx, dy)
        
        # Ensure action is within valid range
        assert 0 <= action < len(self.action_mappings)
        dx_dy = self.action_mappings[action]
        # Discrete action is always 0 (noop)
        return (dx_dy, # In the `DiscreteActions` class, the `action` method is mapping the discrete
        # action index to a corresponding (dx, dy) movement. The `0` in the line
        # `return (dx_dy, 0)` is setting the second component of the movement to `0`.
        # This means that the agent will not move vertically (dy=0) when the discrete
        # action is taken. The first component `dx_dy` determines the horizontal
        # movement based on the discrete action index.
        # In the `DiscreteActions` class, the `action` method is mapping the discrete
        # action index to a corresponding (dx, dy) movement. The `0` in the return
        # statement `(dx_dy, 0)` is indicating that the discrete action is always 0,
        # which means it corresponds to a "noop" action (no operation). This is a
        # placeholder value as the movement is determined solely by the `dx_dy` value
        # in this case.
        0)

class ModifyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Modify observation space if needed
        self.observation_space = env.observation_space

    def observation(self, observation):
        # Modify the observation here
        # Normalize the observation
        modified_observation = observation[0].transpose(2, 0, 1)
        modified_observation = modified_observation/255.0        
        # Normalize the observation
        return modified_observation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs[0].transpose(2,0,1)
        obs = obs/255.0
        return obs, info




class DistributionalDuelingHead(nn.Module):
    """Head module for defining a distributional dueling network.

    This module expects a (batch_size, in_size)-shaped `torch.Tensor` as input
    and returns `pfrl.action_value.DistributionalDiscreteActionValue`.

    Args:
        in_size (int): Input size.
        n_actions (int): Number of actions.
        n_atoms (int): Number of atoms.
        v_min (float): Minimum value represented by atoms.
        v_max (float): Maximum value represented by atoms.
    """

    def __init__(self, in_size, n_actions, n_atoms, v_min, v_max):
        super().__init__()
        assert in_size % 2 == 0
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer(
            "z_values", torch.linspace(v_min, v_max, n_atoms, dtype=torch.float)
        )
        self.a_stream = nn.Linear(in_size // 2, n_actions * n_atoms)
        self.v_stream = nn.Linear(in_size // 2, n_atoms)

    def forward(self, h):
        h_a, h_v = torch.chunk(h, 2, dim=1)
        a_logits = self.a_stream(h_a).reshape((-1, self.n_actions, self.n_atoms))
        a_logits = a_logits - a_logits.mean(dim=1, keepdim=True)
        v_logits = self.v_stream(h_v).reshape((-1, 1, self.n_atoms))
        probs = nn.functional.softmax(a_logits + v_logits, dim=2)
        return pfrl.action_value.DistributionalDiscreteActionValue(probs, self.z_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="agario-screen-v0")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--noisy-net-sigma", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=10**6)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--eval-n-steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=9000)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of hidden layers."
    )

    parser.add_argument("--n-best-episodes", type=int, default=5)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2**31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    
    def make_env_(test):
        # Use different random seeds for train and test envs
        
        env_seed = test_seed if test else train_seed
        env_name = "agario-screen-v0"
        gamma = 0.99
        norm_obs = True
        norm_reward = False
        env_config = json.load(open('env_config.json', 'r'))

        # env = make_env(env_name, env_config, gamma, norm_obs, norm_reward)
        env = gym.make(env_name, **env_config)
        env = DiscreteActions(env)
        env = ModifyObservationWrapper(env)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        
       
        return env

    env = make_env_(test=False)
    eval_env = make_env_(test=True)

    obs_size = env.observation_space.low.size
    n_actions = env.action_space.n
    n_atoms = 51
    v_max = 10
    v_min = -10

    #Two Layers 
    if args.num_layers == 2:
        q_func = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=16, stride=1),
            nn.LayerNorm([64, 113, 113]),  # Add LayerNorm after the first Conv2d layer
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=8, stride=1),
            nn.LayerNorm([32, 106, 106]),  # Add LayerNorm after the second Conv2d layer
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 106 * 106, 128),  # Adjust the input size according to the output of Conv2d
            nn.LayerNorm(128),  # Add LayerNorm after the first Linear layer
            nn.ReLU(),
            DistributionalDuelingHead(128, n_actions, n_atoms, v_min, v_max),
        )
    elif args.num_layers == 1: 
    #One Layer
        q_func = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=16, stride=1),
            nn.LayerNorm([64, 113, 113]),  # Add LayerNorm after the first Conv2d layer
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 113 * 113, 256),  # Adjust the input size according to the output of Conv2d
            nn.LayerNorm(256),  # Add LayerNorm after the first Linear layer, 
            nn.ReLU(),
            DistributionalDuelingHead(256, n_actions, n_atoms, v_min, v_max),
        )
    else: 
        raise ValueError("Invalid number of layers")
    
    # Noisy nets
    pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
    # Turn off explorer
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,end_epsilon = 0.001, random_action_func=env.action_space.sample, decay_steps = 850000
    )

    # Use the same hyper parameters as https://arxiv.org/abs/1710.02298
    opt = torch.optim.Adam(q_func.parameters(), 6.25e-5, eps=1.5 * 10**-4)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 4
    betasteps = args.steps / update_interval
    rbuf = replay_buffers.ReplayBuffer(10**5)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = agents.CategoricalDoubleDQN
    agent = Agent(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        minibatch_size=32,
        replay_start_size=args.replay_start_size,
        target_update_interval=32000,
        update_interval=update_interval,
        batch_accumulator="mean",
        phi=phi,
    )

    if args.load or args.load_pretrained:
        # either load_ or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model(
                    "Rainbow", args.env, model_type=args.pretrained_type
                )[0]
            )

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=args.eval_n_steps, n_episodes=None
        )
        print(
            "n_episodes: {} mean: {} median: {} stdev {}".format(
                eval_stats["episodes"],
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )

    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
            checkpoint_freq = 50000,
        )

        # run 200 evaluation episodes, each capped at 30 mins of play
        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=args.max_frames / 4,
            logger=None,
        )
        with open(os.path.join(args.outdir, "bestscores.json"), "w") as f:
            json.dump(stats, f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))


if __name__ == "__main__":
    main()