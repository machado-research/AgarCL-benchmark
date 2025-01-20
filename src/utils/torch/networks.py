import gymnasium as gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # Assuming obs_shape is (C, H, W) for channel, height, width
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=16, stride=1),
            # Add LayerNorm after the first Conv2d layer
            nn.LayerNorm([64, 69, 69]),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=8, stride=1),
            # Add LayerNorm after the second Conv2d layer
            nn.LayerNorm([32, 62, 62]),
            nn.ReLU(),
            nn.Flatten(),
            # Adjust the input size according to the output of Conv2d
            nn.Linear(32 * 62 * 62, 128),
            nn.LayerNorm(128),  # Add LayerNorm after the first Linear layer
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        return x