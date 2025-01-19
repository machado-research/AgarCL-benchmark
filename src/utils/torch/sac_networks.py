import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.torch.mis import preprocess_image_observation

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_shape).prod(
        ) + np.prod(action_shape), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CriticNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=256):
        super().__init__()

        # Assuming observation_shape is (C, H, W) for channels, height, width
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            sample_input = torch.zeros(*obs_shape)
            sample_input = preprocess_image_observation(sample_input)
            conv_out = self.conv(sample_input)
            conv_out_size = conv_out.size(1)

        self.fc1 = nn.Linear(conv_out_size + np.prod(action_shape), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = preprocess_image_observation(x)

        x = self.conv(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Actor(nn.Module):
    def __init__(self, action_shape, observation_shape, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(observation_shape).prod(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, np.prod(action_shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(action_shape))
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


class ActorNetwork(nn.Module):
    def __init__(self, action_shape, obs_shape, action_low, action_high, hidden_dim=256):
        super().__init__()

        # Assuming observation_shape is (C, H, W) for channels, height, width
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            sample_input = torch.zeros(*obs_shape)
            sample_input = preprocess_image_observation(sample_input)
            conv_out = self.conv(sample_input)
            conv_out_size = conv_out.size(1)

        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, np.prod(action_shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(action_shape))

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
        x = preprocess_image_observation(x)

        x = self.conv(x)
        x = F.relu(self.fc1(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)

        # From SpinUp / Denis Yarats
        log_std = LOG_STD_MIN + 0.5 * \
            (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class HybridActor(nn.Module):
    def __init__(self, cont_action_shape, dis_action_shape, observation_shape, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(observation_shape).prod(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_cont_mean = nn.Linear(hidden_dim, cont_action_shape)
        self.fc_cont_logstd = nn.Linear(hidden_dim, cont_action_shape)

        self.fc_disc_mean = nn.Linear(hidden_dim, dis_action_shape)

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
        cont_mean_action = torch.tanh(
            cont_mean) * self.action_scale + self.action_bias

        # Discrete action sampling
        cat = torch.distributions.Categorical(logits=disc_mean)
        disc_action = cat.sample().reshape(batch_size, 1)
        disc_mean_action = torch.argmax(torch.softmax(
            disc_mean, 0), -1).reshape(batch_size, 1)

        dis_action_log_prob = cat.log_prob(disc_action).sum(-1, keepdim=True)
        log_prob = cont_log_prob + dis_action_log_prob

        action = torch.cat([cont_action, disc_action], 1)
        mean_action = torch.cat([cont_mean_action, disc_mean_action], 1)

        return action, log_prob, mean_action
