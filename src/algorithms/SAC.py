import gym
# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_shape).prod(
        ) + np.prod(action_shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, action_shape, observation_shape, action_low, action_high):
        print(action_high, action_low)
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(observation_shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(action_shape))
        self.fc_logstd = nn.Linear(256, np.prod(action_shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor(
                (action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(
                (action_high + action_low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)

        # From SpinUp / Denis Yarats
        log_std = LOG_STD_MIN + 0.5 * \
            (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class HybridActor(nn.Module):
    def __init__(self, cont_action_shape, dis_action_shape, observation_shape, action_low, action_high):
        print(cont_action_shape, dis_action_shape, observation_shape, action_low, action_high)
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(observation_shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_cont_mean = nn.Linear(256, cont_action_shape)
        self.fc_cont_logstd = nn.Linear(256, cont_action_shape)
        
        self.fc_disc_mean = nn.Linear(256, dis_action_shape)
        
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor(
                (action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(
                (action_high + action_low) / 2.0, dtype=torch.float32)
        )
        
        self.action_dim = cont_action_shape

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        cont_mean = self.fc_cont_mean(x)
        log_std = self.fc_cont_logstd(x)
        log_std = torch.tanh(log_std)

        # From SpinUp / Denis Yarats
        log_std = LOG_STD_MIN + 0.5 * \
            (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        
        # Discrete action forward
        disc_mean = self.fc_disc_mean(x)

        return cont_mean, log_std, disc_mean

    def get_action(self, x):
        batch_size = 1 if x.ndim == 1 else x.shape[0]
        cont_mean, log_std, disc_mean = self(x)
        
        cont_mean = cont_mean.reshape(-1, self.action_dim)
        log_std = log_std.expand_as(cont_mean)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(cont_mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)

        cont_action = y_t * self.action_scale + self.action_bias
        cont_log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        cont_log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        cont_log_prob = cont_log_prob.sum(-1, keepdim=True)
        cont_mean_action = torch.tanh(cont_mean) * self.action_scale + self.action_bias
        
        # Discrete action sampling
        cat = torch.distributions.Categorical(logits=disc_mean)
        disc_action = cat.sample().reshape(batch_size, 1)
        disc_mean_action = torch.argmax(torch.softmax(disc_mean, 0), -1).reshape(batch_size, 1)
        
        dis_action_log_prob = cat.log_prob(disc_action).sum(-1, keepdim=True)
        log_prob = cont_log_prob + dis_action_log_prob
        
        action = torch.cat([cont_action, disc_action], 1)
        mean_action = torch.cat([cont_mean_action, disc_mean_action], 1)
        
        return action, log_prob, mean_action
