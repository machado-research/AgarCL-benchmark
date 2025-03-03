"""A training script of DDPG on OpenAI Gym Mujoco environments.

This script follows the settings of http://arxiv.org/abs/1802.09477 as much
as possible.
"""

import argparse
import logging
import sys

import gymnasium as gym
import gym_agario

import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
# from pfrl.agents.ddpg import DDPG
from pfrl.agents.hybrid_ddpg import HybridDDPG
from pfrl.nn import BoundByTanh, ConcatObsAndAction
from pfrl.policies import DeterministicHead
import os
from pfrl.initializers import init_chainer_default
from torch import distributions, nn
from pfrl.nn.lmbda import Lambda


class MultiActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Tuple((
            # (dx, dy) movemment vector
            gym.spaces.Box(low=-1, high=1, shape=(2,)),
            # 0=noop  1=feed  2=split
            gym.spaces.Discrete(3),
        ))

    def action(self, action):
        return action  # no-op on the second action

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.observation_space.shape[3], self.observation_space.shape[1], self.observation_space.shape[2]), dtype=np.uint8)

    def observation(self, observation):
        return observation[0].transpose(2, 0, 1).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[0].transpose(2, 0, 1).astype(np.uint8), info
    

class NormalizeReward(gym.RewardWrapper):
    #MIN-MAX Normalization
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.r_min = -1.0
        self.r_max = 1.0
        self.epsilon = 1e-8  # Small value to prevent division by zero
    
    def reward(self, reward):
        """Normalize reward to [-1, 1] range."""
        if self.r_max - self.r_min < self.epsilon:
            return 0.0  # Avoid division by zero, return neutral reward
        # print("REWARD: ", reward)
        r = (reward - self.r_min) / (self.r_max - self.r_min + self.epsilon)
        # r = 2 * (reward - self.r_min) / (self.r_max - self.r_min + self.epsilon) - 1
        return r
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="agario-screen-v0",
        help="AgarIO",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2 * 10**6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=1,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
    "--reward", 
    type=str, 
    default = "min_max", #min-max, reward_gym     
    help="REWARD TYPE"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Learning rate."
    )
    parser.add_argument(
        "--target_update_interval", 
        type=int,
        default=4,
        help="Target network update interval"
    )
    parser.add_argument(
        "--update_interval",
        type=int, 
        default=4,
        help="network update interval"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int,
        default=1,
        help="Number of epochs"
    )
    
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=10**5,
        help="Replay Buffer Size"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=5e-3,
        help="Soft update tau"
    )
    
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    def make_env(test):
        import json
        env_config = json.load(open('env_config.json', 'r'))
        env = gym.make(args.env, **env_config)
        gamma  = 0.99
        env.seed(args.seed)
        env = MultiActionWrapper(env)
        env = ObservationWrapper(env)
        if(args.reward == "reward_gym"):
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        else: 
            print("Using Min-Max Normalization")
            env = NormalizeReward(env, gamma=gamma)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    # obs_size = obs_space.low.size
    # action_size = action_space.low.size
    # obs_shape = (obs_space.shape[0], obs_space.shape[1], obs_space.shape[2])
    
    obs_size = 4
    c_action_size = 2
    d_action_size = 3

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
    
    class SoftQNetwork(nn.Module):
        def __init__(self, image_channels,device, c_action_dim = 2, d_action_dim = 3, feature_dim=64):
            super(SoftQNetwork, self).__init__()
            self.d_action_dim = d_action_dim
            self.device = device
            # Convolutional layers for image encoding
            self.conv = nn.Sequential(
                CustomCNN(n_input_channels=image_channels, n_output_channels=256),
                nn.ReLU(),
            )
            conv_output_size = 256
            
            # Fully connected layers that combine the flattened conv features and action
            self.fc1 = init_chainer_default(nn.Linear(conv_output_size + c_action_dim, feature_dim))
            self.fc2 = init_chainer_default(nn.Linear(feature_dim, d_action_dim))  # Output a single Q-value

        def forward(self, state_action):
            """
            state: Tensor of shape [batch, channels, height, width]
            action: Tensor of shape [batch, action_dim]
            """

            assert len(state_action) == 2
            state,action = state_action
            c_action, d_action = action 
            conv_out = self.conv(state)
            conv_out = conv_out.view(conv_out.size(0), -1)
            x = torch.cat([conv_out, c_action], dim=-1)  # [batch, conv_features + action_dim]
            x = self.fc1(x)
            x = nn.ReLU()(x)
            q_values = self.fc2(x)  # [batch, 3]
            q_value = q_values.gather(1, d_action.long().view(-1, 1).to(self.device)).squeeze().view(-1)
            return q_value
        
    q_func = nn.Sequential(
            SoftQNetwork(image_channels=obs_size, device=args.gpu, c_action_dim=c_action_size, d_action_dim=d_action_size)
        )
    
    policy = nn.Sequential(
        CustomCNN(n_input_channels=obs_size, n_output_channels=256),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                init_chainer_default(nn.Linear(256, c_action_size)),
                BoundByTanh(low=action_space[0].low, high=action_space[0].high),
                DeterministicHead()
            ),
            nn.Sequential(
                init_chainer_default(nn.Linear(256, d_action_size)),
                pfrl.policies.SoftmaxCategoricalHead(),
            )
        )
    )

    opt_a = torch.optim.Adam(policy.parameters() , lr = args.lr)
    opt_c = torch.optim.Adam(q_func.parameters(), lr = args.lr)

    rbuf = replay_buffers.ReplayBuffer(args.replay_buffer_size)
    explorer = explorers.AdditiveGaussian(
        scale=0.1, low=action_space[0].low, high=action_space[0].high
    )

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return (
            np.random.uniform(action_space[0].low, action_space[0].high).astype(np.float32),
            np.random.randint(action_space[1].n)
        )

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255
    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = HybridDDPG(
        policy,
        q_func,
        opt_a,
        opt_c,
        rbuf,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_method="soft",
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        soft_update_tau=args.tau,
        n_times_update=args.epochs,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
        phi = phi
    )

    if len(args.load) > 0 or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not len(args.load) > 0 or not args.load_pretrained
        if len(args.load) > 0:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model("DDPG", args.env, model_type=args.pretrained_type)[
                    0
                ]
            )

    eval_env = make_env(test=True)
    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
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
        import json
        import os

        with open(os.path.join(args.outdir, "demo_scores.json"), "w") as f:
            json.dump(eval_stats, f)
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_env=eval_env,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            train_max_episode_len=timestep_limit,
        )


if __name__ == "__main__":
    main()