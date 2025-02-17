"""A training script of PPO on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1709.06560 as much
as possible.
"""
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
# from src.wrappers.gym import make_env
from pfrl.initializers import init_chainer_default

from pfrl.agents import PPO
# from gym import spaces
import functools
import gym_agario
import os

class MultiActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))  # (dx, dy) movement vector

    def action(self, action):
        return (action, 0)  # no-op on the second action

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.observation_space.shape[3], self.observation_space.shape[1], self.observation_space.shape[2]), dtype=np.uint8)

    def observation(self, observation):
        return observation[0].transpose(2, 0, 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[0].transpose(2, 0, 1), info


def main():
    import logging
    assert torch.cuda.is_available()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="agario-screen-v0",
        help="AgarIO",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=10, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="PPO_results_Exp2",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default= 2 * 10**6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=5,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=5000,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to update model for per PPO iteration.",
    )

    parser.add_argument(
        "--clip-eps", type=float, default=0.4, help="Clipping parameter for PPO.")

    parser.add_argument(
        "--entropy-coef", type=float, default=0.001, help="Entropy coefficient for PPO.")

    parser.add_argument(
        "--clip-eps-vf", type=float, default=0.2, help="Clipping parameter for the value function.")

    parser.add_argument(
        "--value-func-coef", type=float, default=0.7, help="Value function coefficient for PPO.")

    parser.add_argument(
        "--max-grad-norm", type=float, default=0.9, help="Maximum norm of gradients.")

    parser.add_argument(
        "--lr", type=float, default=1e-5, help="The learning rate of the optimizer.")


    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2**32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env_config = json.load(open('env_config.json', 'r'))
        env = gym.make(args.env, **env_config)
        gamma  = 0.99
        # Use different random seeds for train and test envs
        # env_seed = (2**32 - 1 - process_seed if test else process_seed) % (2**32)
        env.seed(args.seed)
        env = MultiActionWrapper(env)
        env = ObservationWrapper(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.flatten_observation.FlattenObservation(env)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        #Scaling Rewards
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, 0, 1))
        # if args.monitor:
        #     env = pfrl.wrappers.Monitor(env, args.outdir)
        # if args.render:
        #     env = pfrl.wrappers.Render(env)

        return env

    def make_batch_env(test):
        return make_env(0, test)
        # return pfrl.envs.MultiprocessVectorEnv(
        #     [
        #         functools.partial(make_env, idx, test)
        #         for idx, env in enumerate(range(args.num_envs))
        #     ]
        # )
    env_config = json.load(open('env_config.json', 'r'))
    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env, **env_config)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    # assert isinstance(action_space, gym.spaces.Box)

    # Normalize observations based on their empirical mean and variance
    obs_shape = (obs_space.shape[3], obs_space.shape[1], obs_space.shape[2])
    
    
    class CustomCNN(nn.Module):
        def __init__(self, n_input_channels, n_output_channels, activation=nn.ReLU(), bias=0.1):
            super().__init__()
            self.n_input_channels = n_input_channels
            self.activation = activation
            self.n_output_channels = n_output_channels
            self.layers = nn.ModuleList(
                [
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
                    nn.LayerNorm([32, 31, 31]),
                    nn.Conv2d(32, 64, 4, stride=2),
                    nn.LayerNorm([64, 14, 14]),
                    nn.Conv2d(64, 32, 3, stride=1),
                    nn.LayerNorm([32, 12, 12]),
                ]
            )
            self.output = nn.Linear(4608, n_output_channels)  # Adjusted for 3x84x84 input

            self.apply(init_chainer_default)
            self.apply(self.constant_bias_initializer(bias=bias))

        def constant_bias_initializer(self, bias=0.1):
            def init(m):
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, bias)
            return init

        def forward(self, state):
            h = state
            for layer in self.layers:
                h = self.activation(layer(h))
            h_flat = h.view(h.size(0), -1)
            return self.activation(self.output(h_flat))
        
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_shape, clip_threshold=5
    )

    obs_size = obs_space.low.size
    # action_size = action_space.low.size
    action_size = 2
    policy = nn.Sequential(
        CustomCNN(n_input_channels=4, n_output_channels=256),
        nn.ReLU(),
        init_chainer_default(nn.Linear(256, 2)),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
    )

    vf = torch.nn.Sequential(
        CustomCNN(n_input_channels=4, n_output_channels=256),
        nn.ReLU(),
        init_chainer_default(nn.Linear(256, 1)),
    )

    # While the original paper initialized weights by normal distribution,
    # we use orthogonal initialization as the latest openai/baselines does.
    # def ortho_init(layer, gain):
    #     nn.init.orthogonal_(layer.weight, gain=gain)
    #     nn.init.zeros_(layer.bias)

    # ortho_init(policy[0], gain=1)
    # ortho_init(policy[3], gain=1)
    # ortho_init(policy[7], gain=1e-2)
    # ortho_init(vf[0], gain=1)
    # ortho_init(vf[3], gain=1)
    # ortho_init(vf[7], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)

    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        entropy_coef=args.entropy_coef,
        clip_eps_vf=args.clip_eps_vf,
        value_func_coef=args.value_func_coef,
        max_grad_norm=args.max_grad_norm,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
    )

    if args.load or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(utils.download_model("PPO", args.env, model_type="final")[0])

    if args.demo:
        env = make_batch_env(True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )

        with open(os.path.join(args.outdir, "demo_scores.json"), "w") as f:
            json.dump(eval_stats, f)
    else:
        # experiments.train_agent_batch_with_evaluation(
        #     agent=agent,
        #     env=make_batch_env(False),
        #     eval_env=make_batch_env(True),
        #     outdir=args.outdir,
        #     steps=args.steps,
        #     eval_n_steps=None,
        #     eval_n_episodes=args.eval_n_runs,
        #     eval_interval=args.eval_interval,
        #     log_interval=args.log_interval,
        #     max_episode_len=timestep_limit,
        #     save_best_so_far_agent=False,
        # )
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            checkpoint_freq = 50000,
            # log_interval=args.log_interval,
             train_max_episode_len=timestep_limit,
             eval_max_episode_len=timestep_limit,
        )


if __name__ == "__main__":
    main()
