import gymnasium as gym
import gym_agario

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

from src.wrappers.gym import FlattenObservation
from src.utils import modify_action, modify_hybrid_action
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity

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


class SAC:
    def __init__(self,
                 env: gym.Env,
                 seed: int,
                 device: str,
                 hypers: dict,
                 collector_config: dict,
                 total_timesteps: int = 1e6,
                 eval_timesteps: int = 3000,
                 render: bool = False,
                 ) -> None:

        self.env = env
        self.total_timesteps = total_timesteps
        self.eval_timesteps = eval_timesteps
        self.render = render
        self.device = device
        self.autotune = True
        
        self.collector = self.collector_init(collector_config)
        self.collector.setIdx(seed)

        self.hybrid = hypers['hybrid']
        self.q_lr = hypers['q_lr']
        self.policy_lr = hypers['policy_lr']
        # self.alpha = alpha
        self.buffer_size = int(hypers['buffer_size'])
        self.learning_starts = int(hypers['learning_starts'])
        self.batch_size = hypers['batch_size']
        self.policy_frequency = hypers['policy_frequency']
        self.target_network_frequency = hypers['target_network_frequency']
        self.gamma = hypers['gamma']
        self.tau = hypers['tau']

        self.env.action_space = (gym.spaces.Box(-1, 1, self.env.action_space[0].shape, dtype=np.float32),
                                 gym.spaces.Discrete(3))

        # Define networks
        self.max_action = float(self.env.action_space[0].high[0])
        self.min_action = float(self.env.action_space[0].low[0])
        self.action_shape = (self.env.action_space[0].shape[0] + 1,)
        self.obs_shape = (np.prod(self.env.observation_space.shape),)
        self.env.observation_space.dtype = np.float32

        if self.hybrid:
            cont_action_shape = self.env.action_space[0].shape[0]
            dis_action_shape = self.env.action_space[1].n
            self.actor = HybridActor(cont_action_shape, dis_action_shape,
                                     self.obs_shape, self.min_action, self.max_action).to(self.device)
        else:
            self.actor = Actor(self.action_shape, self.obs_shape,
                               self.min_action, self.max_action).to(self.device)

        self.qf1 = SoftQNetwork(
            self.obs_shape, self.action_shape).to(self.device)
        self.qf2 = SoftQNetwork(
            self.obs_shape, self.action_shape).to(self.device)
        self.qf1_target = SoftQNetwork(
            self.obs_shape, self.action_shape).to(self.device)
        self.qf2_target = SoftQNetwork(
            self.obs_shape, self.action_shape).to(self.device)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) +
                                      list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.policy_lr)

        # Automatic entropy tuning
        self.target_entropy = - \
            torch.prod(torch.Tensor(
                self.env.action_space[0].shape).to(self.device)).item()
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)

        self.rb = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            gym.spaces.Box(-1, 1, self.action_shape, dtype=np.float32),
            self.device,
            handle_timeout_termination=False,
        )

    def train(self):
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs = self.env.reset()
        for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.learning_starts:
                action = gym.spaces.Box(-1, 1, self.action_shape,
                                        dtype=np.float32).sample()
            else:
                action, _, _ = self.actor.get_action(
                    torch.Tensor(obs).to(self.device))
                action = action.detach().cpu().numpy().squeeze()

            if self.hybrid:
                step_action = modify_hybrid_action(action)
            else:
                step_action = modify_action(
                    action, self.min_action, self.max_action)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, termination, truncation, info = self.env.step(
                step_action)

            self.collector.collect("reward", reward)
            self.collector.collect("moving_avg", reward)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            self.rb.add(obs, next_obs, action, reward, termination, info)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                data = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                        data.next_observations)
                    qf1_next_target = self.qf1_target(
                        data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(
                        data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(
                        qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * \
                        self.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(
                    data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(
                    data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        self.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(
                            data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if self.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(
                                    data.observations)
                            alpha_loss = (-self.log_alpha.exp() *
                                          (log_pi + self.target_entropy)).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_step % self.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data)

                if global_step % 100 == 0:
                    self.collector.collect(
                        "qf1_values", qf1_a_values.mean().item())
                    self.collector.collect(
                        "qf2_values", qf2_a_values.mean().item())
                    self.collector.collect("qf1_loss", qf1_loss.item())
                    self.collector.collect("qf2_loss", qf2_loss.item())
                    self.collector.collect("qf_loss", qf_loss.item() / 2.0)
                    self.collector.collect("actor_loss", actor_loss.item())
                    self.collector.collect("alpha", alpha)
                    self.collector.collect("alpha_loss", alpha_loss.item())

                    print("SPS:", int(global_step / (time.time() - start_time)))
                    self.collector.collect("SPS", int(global_step /
                                                      (time.time() - start_time)))

        return torch.Tensor(next_obs).to(self.device)

    def eval(self, obs):
        eval_avg_reward = 0
        eval_mov_average = 0

        for _ in range(self.eval_timesteps):
            with torch.no_grad():
                action, _, _ = self.actor.get_action(obs)
                action = action.detach().cpu().numpy().squeeze()

                if self.hybrid:
                    step_action = modify_hybrid_action(action)
                else:
                    step_action = modify_action(
                        action, self.min_action, self.max_action)

                next_obs, reward, termination, truncation, info = self.env.step(
                    step_action)

                eval_avg_reward += reward
                eval_mov_average = 0.99 * eval_mov_average + 0.01 * reward

                if self.render:
                    self.env.render()

                obs = torch.Tensor(next_obs).to(self.device)

                self.collector.collect("eval_reward", reward)
                self.collector.collect("eval_moving_avg", reward)

        return eval_avg_reward, eval_mov_average

    @staticmethod
    def collector_init(config):
        config['qf1_values'] = Identity()
        config['qf2_values'] = Identity()
        config['qf1_loss'] = Identity()
        config['qf2_loss'] = Identity()
        config['actor_loss'] = Identity()
        config['alpha'] = Identity()
        config['alpha_loss'] = Identity()

        return Collector(config)
    
    def save_collector(self, exp, save_path):
        from PyExpUtils.results.sqlite import saveCollector
        
        self.collector.reset()
        saveCollector(exp, self.collector, base=save_path)