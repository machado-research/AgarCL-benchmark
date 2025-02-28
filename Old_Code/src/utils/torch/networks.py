import gymnasium as gym
import torch.nn as nn
import torch
from torch.distributions import Normal, Categorical

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class CNNPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # Assuming obs_shape is (C, H, W) for channel, height, width
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=16, stride=1),
            # Add LayerNorm after the first Conv2d layer
            nn.LayerNorm([64, 113, 113]),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=8, stride=1),
            # Add LayerNorm after the second Conv2d layer
            nn.LayerNorm([32, 106, 106]),
            nn.ReLU(),
            nn.Flatten(),
            # Adjust the input size according to the output of Conv2d
            nn.Linear(32 * 106 * 106, 128),
            nn.LayerNorm(128),  # Add LayerNorm after the first Linear layer
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        return x
    


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer 

#The CNN architecture for the Actor-Critic network: Len_Screen is 128x128x4
class PPOAgent(nn.Module):
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
            )
        
         # Separate outputs for continuous and discrete actions
        self.actor_mean = layer_init(nn.Linear(256, 2))  # 2 continuous actions
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))  # Log std for 2 continuous actions
        self.actor_discrete = layer_init(nn.Linear(256, 3))  # 3 discrete actions (logits)

    def get_value(self, x):
        return self.critic(x)
    
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
