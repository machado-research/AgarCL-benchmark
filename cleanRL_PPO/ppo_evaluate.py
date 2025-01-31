# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
# from torch.distributions.normal import Normal
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
import gym_agario 
import json
from custom_cleanrl_utils.evals.ppo_eval import evaluate
from custom_cleanrl_utils.huggingface import push_to_hub

import cv2

class MultiActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # self.action_space = gym.spaces.Tuple((
        #     gym.spaces.Box(low=-1, high=1, shape=(2,)),  # (dx, dy) movement vector
        #     gym.spaces.Discrete(3),                      # 0=noop, 1=split, 2=feed
        # ))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,)),  # (dx, dy) movement vector

    def action(self, action):
        return (action, 0)  # no-op on the second action

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Modify observation space if needed
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.observation_space.shape[3], self.observation_space.shape[1], self.observation_space.shape[2]), dtype=np.uint8)

    def observation(self, observation):
        # Convert the observation to grayscale
        # gray_observation = cv2.cvtColor(observation[0], cv2.COLOR_RGB2GRAY)

        # # Create the folder if it does not exist
        # os.makedirs("images", exist_ok=True)

        # # Save the grayscale image
        # cv2.imwrite(f"images/observation_{int(time.time())}.png", gray_observation)
        return observation.transpose(0, 3, 1, 2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.transpose(0, 3, 1, 2), info

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "agario-screen-v0"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 100
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    env_config = json.load(open('/home/mamm/ayman/thesis/AgarLE-benchmark/env_config.json', 'r'))
    env = gym.make(env_id, **env_config)
    # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = MultiActionWrapper(env)
    env = ObservationWrapper(env)
    # env = gym.wrappers.ClipAction(env)
    
    # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=envs.observation_space.shape[0], out_channels=64, kernel_size=16, stride=1)),
            nn.LayerNorm([64, 113, 113]),  # Add LayerNorm after the first Conv2d layer
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1)),
            nn.LayerNorm([64, 106, 106]),  # Add LayerNorm after the second Conv2d layer
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 106* 106, 256)),
            nn.LayerNorm(256),  # Add LayerNorm after the first Linear layer
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_base = nn.Sequential(
            nn.Conv2d(in_channels=envs.observation_space.shape[0], out_channels=64, kernel_size=16, stride=1),
            nn.LayerNorm([64, 113, 113]),  # Add LayerNorm after the first Conv2d layer
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1),  # Additional Conv2d layer
            nn.LayerNorm([64, 106, 106]),  # Add LayerNorm after the additional Conv2d layer
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 106* 106, 256)),
            nn.LayerNorm(256),  # Add LayerNorm after the first Linear layer
            nn.ReLU(),
            # nn.Linear(256, 5),  # Output layer 5 actions [continous_1, continous_2, do_nothing, split, feed]
            # LambdaLayer(lambda x: torch.cat((x[:, :2], torch.softmax(x[:, 2:], dim=-1)), dim=-1))
            )
        
        # self.actor_logstd = nn.Parameter(torch.zeros(1, 3))
         # Separate outputs for continuous and discrete actions
        self.actor_mean = layer_init(nn.Linear(256, 2))  # 2 continuous actions
        self.actor_logstd = nn.Parameter(torch.zeros(2))  # Log std for 2 continuous actions
        # self.actor_discrete = layer_init(nn.Linear(256, 3))  # 3 discrete actions (logits)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        features = self.actor_base(x)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        
        return torch.tanh(action), probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)
    
    
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    model_path = '/home/mamm/ayman/thesis/AgarLE-benchmark/cleanRL_PPO/runs/agario-screen-v0__ppo_custom__1__1738218023/ppo_custom.cleanrl_model'
    envs = [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    env = envs[0]
    episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
            env=env,
        )
    with open("evaluation_results.txt", "a") as file:
        for idx, episodic_return in enumerate(episodic_returns):
                file.write(f"Episode {idx}: {episodic_return}\n")