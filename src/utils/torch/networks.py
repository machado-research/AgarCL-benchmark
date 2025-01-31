import gymnasium as gym
import torch.nn as nn
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
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer