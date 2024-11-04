import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from src.utils.torch.mis import preprocess_image_observation

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNAgent(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=64):
        super().__init__()
        
        # Assuming obs_shape is (C, H, W) for channel, height, width
        self.conv_layers = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            sample_input = torch.zeros(*obs_shape)
            sample_input = preprocess_image_observation(sample_input)
            conv_out = self.conv_layers(sample_input)
            conv_out_size = conv_out.size(1)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(conv_out_size, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(conv_out_size, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, np.prod(action_shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))
        self.action_dim = action_shape[0]
        
        print(f'PPO has {self.get_parameter_count()} parameters!')
        
    def forward(self, x, action=None):
        x = preprocess_image_observation(x)
        features = self.conv_layers(x)
        return self.actor_mean(features), self.critic(features)

    def get_value(self, x):
        x = preprocess_image_observation(x)
        features = self.conv_layers(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None, action_limits=None):
        x = preprocess_image_observation(x)
        features = self.conv_layers(x)
        action_mean = self.actor_mean(features).reshape(-1, self.action_dim)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        if action_limits:
            action = torch.clamp(action, action_limits[0], action_limits[1])

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, np.prod(action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))
        self.action_dim = action_shape[0]

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_limits=None):
        action_mean = self.actor_mean(x).reshape(-1, self.action_dim)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if action_limits:
            action = np.clip(action, action_limits[0], action_limits[1])

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class HybridAgent(nn.Module):
    def __init__(self, obs_shape, cont_action_shape, dis_action_shape, hidden_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, np.prod(cont_action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(cont_action_shape)))
        self.action_dim = cont_action_shape

        self.dis_actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, dis_action_shape), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_limits=None):
        action_mean = self.actor_mean(x).reshape(-1, self.action_dim)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cont_probs = Normal(action_mean, action_std)

        disc_action_mean = self.dis_actor_mean(x)
        disc_probs = Categorical(logits=disc_action_mean)

        if action is None:
            cont_action = cont_probs.sample()
            dis_action = disc_probs.sample((1, 1))
        else:
            cont_action, dis_action = action[:, :-1], action[:, -1]

        if action_limits:
            cont_action = np.clip(
                cont_action, action_limits[0], action_limits[1])

        cont_action_log_prob = cont_probs.log_prob(
            cont_action).sum(-1, keepdim=True)
        cont_action_entropy = cont_probs.entropy().sum(-1, keepdim=True)

        dis_action_log_prob = disc_probs.log_prob(
            dis_action).sum(-1, keepdim=True)
        dis_action_entropy = disc_probs.entropy().sum(-1, keepdim=True)

        if action is None:
            action = torch.cat([cont_action, dis_action], 1)

        log_prob = cont_action_log_prob + dis_action_log_prob
        entropy = cont_action_entropy + dis_action_entropy

        return action, log_prob, entropy, self.critic(x)

