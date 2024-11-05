import gymnasium as gym

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity

from src.utils.torch.actions import modify_action, modify_hybrid_action
from src.utils.torch.sac_networks import *


class SAC:
    def __init__(self,
                 env: gym.Env,
                 seed: int,
                 device: str,
                 hypers: dict,
                 collector_config: dict,
                 total_timesteps: int = 1e6,
                 render: bool = False,
                 ) -> None:

        self.env = env
        self.total_timesteps = total_timesteps
        self.render = render
        self.device = device
        self.autotune = True

        self.collector = self.collector_init(collector_config)
        self.collector.setIdx(seed)

        self.q_lr = hypers['q_lr']
        self.policy_lr = hypers['policy_lr']
        self.buffer_size = int(hypers['buffer_size'])
        self.hidden_dim = hypers['hidden_size']
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
        self.obs_shape = self.env.observation_space.shape
        self.env.observation_space.dtype = np.float32

        self.actor = CNNActor(self.action_shape, self.obs_shape,
                              self.min_action, self.max_action, self.hidden_dim).to(self.device)

        if len(self.obs_shape) > 1:
            self.qf1 = CNNSoftQNetwork(
                self.obs_shape, self.action_shape).to(self.device)
            self.qf2 = CNNSoftQNetwork(
                self.obs_shape, self.action_shape).to(self.device)
            self.qf1_target = CNNSoftQNetwork(
                self.obs_shape, self.action_shape).to(self.device)
            self.qf2_target = CNNSoftQNetwork(
                self.obs_shape, self.action_shape).to(self.device)
        else:
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
        trial_rewards = 0
        steps = 0
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

            step_action = modify_action(
                action, self.min_action, self.max_action)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, termination, truncation, info = self.env.step(
                step_action)

            trial_rewards += reward
            steps += 1
            
            

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

                if global_step % 10 == 0:
                    self.collector.next_frame()
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

                    print("SPS:", int(global_step / (time.time() - start_time)), " | Reward: ", reward, " | Score: ", trial_rewards/steps)
                    self.collector.collect("SPS", int(global_step /
                                                      (time.time() - start_time)))

        return trial_rewards/steps

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
