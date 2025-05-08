"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.
"""
import argparse
import functools
import logging
import sys
from distutils.version import LooseVersion

import gymnasium as gym
import numpy as np
import torch
from torch import distributions, nn

import pfrl
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda
from pfrl.initializers import init_chainer_default

import gym_agario
import os
import json
import os

import wandb

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
    assert torch.cuda.is_available(), "torch.cuda must be available. Aborting."
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/mamm/ayman/thesis/AgarLE-benchmark/SAC_mode_3_cont",
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
    parser.add_argument(
    "--reward", 
    type=str, 
    default = "reward_gym", #min-max, reward_gym     
    help="REWARD TYPE"
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
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
        default= 5 * 10**6,
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
        default=50000,
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
        "--log-interval",
        type=int,
        default=500,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--policy-output-scale",
        type=float,
        default=1.0,
        help="Weight initialization scale of policy output.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate."
    )
    parser.add_argument(
        "--update-interval", type=int , default=4, help = "Updating the neural network in every time steps."
    )
    parser.add_argument(
        "--replay-buffer", type=int, default=int(1e5), help="Replay Buffer."
    )
    parser.add_argument(
        "--soft-update-tau",
        type=float,
        default=0.01,
        help="Coefficient for soft update of the target network.",
    )
    parser.add_argument(
        "--max-grad-norm", 
        type=float, 
        default=0.5, 
        help="Norm of max_grad",
    )
    parser.add_argument(
        "--temperature-lr",
        type=float,
        default=1e-4,
        help="Learning rate of temperature optimizer.",
    )
    
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument(
        "--cont", action="store_true", help="Continue training from checkpoint"
    )
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--load-replay-buffer", type=str, default="")
    
    parser.add_argument("--load-env", type=str, default="")
    
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    if (args.load != ""): 
        exp_id = args.load.split("/")[-2]
        args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id, argv=sys.argv)
        #Here update both --load-env and --load-replay-buffer
        checkpoint_number = args.load.split("/")[-1].split("_")[0]
        load_env_checkpoint_name = f"checkpoint_{checkpoint_number}.json"
        args.load_env = os.path.join(args.outdir, load_env_checkpoint_name)
        args.load_replay_buffer = os.path.join(args.outdir, f"{checkpoint_number}_replay.pkl")
    else: 
        args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    if args.wandb:
        wandb.init(project="agarle", name="SAC", config=vars(args))
        wandb.config.update(args)

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)
    env_config = json.load(open('env_config.json', 'r'))
    def make_env(process_idx, test):

        
        env = gym.make(args.env, **env_config)
        gamma  = 0.99
        # Use different random seeds for train and test envs
        # env_seed = (2**32 - 1 - process_seed if test else process_seed) % (2**32)
        env.seed(args.seed)
        if args.load_env != "": 
            env.load_env_state(args.load_env)
            
        env = MultiActionWrapper(env)
        env = ObservationWrapper(env)
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.flatten_observation.FlattenObservation(env)
        # Cast observations to float32 because our model uses float32
        # env = pfrl.wrappers.CastObservationToFloat32(env)
        #Scaling Rewards
        if(args.reward == "reward_gym"):
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        else: 
            print("Using Min-Max Normalization")
            env = NormalizeReward(env, gamma=gamma)
        return env

    def make_batch_env(test):
        return make_env(0, test)

    sample_env = make_env(process_idx=0, test=False)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = 4
    c_action_size = 2
    d_action_size = 3
    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == c_action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )
    
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

    policy = nn.Sequential(
        CustomCNN(n_input_channels=obs_size, n_output_channels=256),
        nn.ReLU(),
        pfrl.nn.Branched(
            #continuous actions
            nn.Sequential(
                init_chainer_default(nn.Linear(256, c_action_size * 2)),
                Lambda(squashed_diagonal_gaussian_head),
            ),
            #Discrete actions
            nn.Sequential(
                init_chainer_default(nn.Linear(256, d_action_size)),
                pfrl.policies.SoftmaxCategoricalHead(),
            )
        )
    )

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

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

    def make_q_func_with_optimizer():
        q_func = SoftQNetwork(image_channels=obs_size, device=device, c_action_dim=c_action_size, d_action_dim=d_action_size)
        
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=args.lr)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(args.replay_buffer)
    
    if(args.load_replay_buffer != ""):
        rbuf.load(args.load_replay_buffer)
        print("Replay buffer loaded from: ", args.load_replay_buffer)

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
    agent = pfrl.agents.HybridSoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        phi=phi,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
        c_entropy_target=-c_action_size,
        d_entropy_target=-d_action_size,
        temperature_optimizer_lr=args.temperature_lr,
        update_interval=args.update_interval,
        soft_update_tau=args.soft_update_tau,
        max_grad_norm=args.max_grad_norm,
        act_deterministically=False,
    )

    if len(args.load) > 0 or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not len(args.load) > 0 or not args.load_pretrained
        if len(args.load) > 0:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model("SAC", args.env, model_type=args.pretrained_type)[
                    0
                ]
            )

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
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
        #     env=make_batch_env(test=False),
        #     eval_env=make_batch_env(test=True),
        #     outdir=args.outdir,
        #     steps=args.steps,
        #     eval_n_steps=None,
        #     eval_n_episodes=args.eval_n_runs,
        #     eval_interval=args.eval_interval,
        #     log_interval=args.log_interval,
        #     max_episode_len=timestep_limit,
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
            save_best_so_far_agent=False,
            checkpoint_freq = 1000000,
            # log_interval=args.log_interval,
             train_max_episode_len=timestep_limit,
             eval_max_episode_len=timestep_limit,
            case="continuing" if args.cont else "episodic",
            step_offset=args.step_offset,
            # env_checkpointable=True,

        )


if __name__ == "__main__":
    main()