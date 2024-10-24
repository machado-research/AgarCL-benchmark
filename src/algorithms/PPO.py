import gymnasium as gym

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity

from src.wrappers.gym import *
from src.utils.actions import modify_action, modify_hybrid_action
from src.utils.ppo_networks import *
from PIL import Image
import os

from tqdm import tqdm

class PPO:
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
        self.render = render
        self.total_timesteps = total_timesteps
        self.eval_timesteps = eval_timesteps
        self.num_envs = 1

        self.collector = self.collector_init(collector_config)
        self.collector.setIdx(seed)

        # Hyperparameters
        self.hybrid = hypers['hybrid']
        self.num_steps = hypers['num_steps']
        self.anneal_lr = hypers['anneal_lr']
        self.learning_rate = hypers['learning_rate']
        self.gamma = hypers['gamma']
        self.gae_lambda = hypers['gae_lambda']
        self.num_minibatches = hypers['num_minibatches']
        self.update_epochs = hypers['update_epochs']
        self.norm_adv = hypers['norm_adv']
        self.clip_coef = hypers['clip_coef']
        self.clip_vloss = hypers['clip_vloss']
        self.ent_coef = hypers['ent_coef']
        self.vf_coef = hypers['vf_coef']
        self.max_grad_norm = hypers['max_grad_norm']
        self.hidden_dim = hypers['hidden_size']
        self.target_kl = hypers.get('target_kl', None)

        # to be filled in runtime
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size

        self.device = device
        self.env.action_space = (gym.spaces.Box(-1, 1, self.env.action_space[0].shape, dtype=np.float32),
                                 gym.spaces.Discrete(3))

        # Define networks
        self.max_action = float(self.env.action_space[0].high[0])
        self.min_action = float(self.env.action_space[0].low[0])
        self.action_shape = (self.env.action_space[0].shape[0] + 1,)
        self.obs_shape = self.env.observation_space.shape
        self.env.observation_space.dtype = np.float32

        if self.hybrid:
            cont_action_shape = self.env.action_space[0].shape[0]
            dis_action_shape = self.env.action_space[1].n
            self.agent = HybridAgent(
                self.obs_shape, cont_action_shape, dis_action_shape).to(self.device)
        else:
            if len(self.obs_shape) > 1:
                self.agent = CNNAgent(
                    self.obs_shape, self.action_shape, self.hidden_dim).to(self.device)
            else:
                self.agent = Agent(
                    self.obs_shape, self.action_shape).to(self.device)

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.obs_shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) +
                                   self.action_shape).to(self.device)
        self.logprobs = torch.zeros(
            (self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros(
            (self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros(
            (self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros(
            (self.num_steps, self.num_envs)).to(self.device)

    def train(self):
        global_step = 0

        start_time = time.time()
        next_obs = self.env.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        for iteration in tqdm(range(1, self.num_iterations + 1), desc="Training Progress"):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs, action_limits=(self.min_action, self.max_action))
                    self.values[step] = value.flatten()
                    
                #     obs_image = next_obs.cpu().numpy().squeeze(0)  # Remove the batch dimension
                #     obs_image = (obs_image).astype(np.uint8)  # Convert to uint8
                #     img = Image.fromarray(obs_image)
                #     if not os.path.exists("/home/mamm/ayman/thesis/AgarLE-benchmark/observations"):
                #         os.makedirs("/home/mamm/ayman/thesis/AgarLE-benchmark/observations")
                #     img.save(f"/home/mamm/ayman/thesis/AgarLE-benchmark/observations/obs_{global_step}.png")
                # next_obs = (next_obs - next_obs.mean()) / (next_obs.std() + 1e-8)
                
                
                self.actions[step] = action
                self.logprobs[step] = logprob
                action = action.detach().cpu().numpy().squeeze()

                if self.hybrid:
                    step_action = modify_hybrid_action(action)
                else:
                    step_action = modify_action(
                        action, self.min_action, self.max_action)
                # print(f"Step: {step}, Action: {step_action}, Logprob: {logprob}, Value: {value}")
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, termination, truncation, info = self.env.step(
                    step_action)
                next_done = np.ones((1,)) * termination
                self.rewards[step] = torch.tensor(
                    reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device), torch.Tensor(next_done).to(self.device)

                self.collector.collect("reward", reward)
                self.collector.collect("moving_avg", reward)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * \
                        nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * \
                        self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.obs_shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.action_shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds], action_limits=(self.min_action, self.max_action))
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() >
                                       self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()
                                         ) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * \
                        torch.clamp(ratio, 1 - self.clip_coef,
                                    1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(
                            v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * \
                            ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - \
                np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.collector.next_frame()
            self.collector.collect("lr", self.optimizer.param_groups[0]["lr"])
            self.collector.collect("value_loss", v_loss.item())
            self.collector.collect("policy_loss", pg_loss.item())
            self.collector.collect("entropy_loss", entropy_loss.item())
            self.collector.collect("old_approx_kl", old_approx_kl.item())
            self.collector.collect("approx_kl", approx_kl.item())
            self.collector.collect("clipfrac", np.mean(clipfracs))
            self.collector.collect("explained_variance", explained_var)

            # print("SPS:", int(global_step / (time.time() - start_time)), global_step)
            self.collector.collect("SPS", int(
                global_step / (time.time() - start_time)))

        return torch.Tensor(next_obs).to(self.device)

    def eval(self):
        obs = self.env.reset()
        obs = torch.Tensor(obs).to(self.device)
        eval_avg_reward = 0
        eval_mov_average = 0
        import imageio
        import os
        video_dir = "/home/mamm/ayman/thesis/AgarLE-benchmark"
        video_writer = imageio.get_writer(os.path.join(video_dir, 'grid_env.mp4'), fps=60)
        for x in range(self.eval_timesteps):
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(
                    obs, action_limits=(self.min_action, self.max_action))
                action = action.detach().cpu().numpy().squeeze()
                # Save the observation as an image
                # obs_image = obs.cpu().numpy().squeeze(0)  # Remove the batch dimension
                # obs_image = (obs_image).astype(np.uint8)  # Convert to uint8
                # img = Image.fromarray(obs_image)
                # import os
                # if not os.path.exists("/home/mamm/ayman/thesis/AgarLE-benchmark/observations"):
                #     os.makedirs("/home/mamm/ayman/thesis/AgarLE-benchmark/observations")
                # img.save(f"/home/mamm/ayman/thesis/AgarLE-benchmark/observations/obs_test_{x}.png")

                if self.hybrid:
                    step_action = modify_hybrid_action(action)
                else:
                    step_action = modify_action(
                        action, self.min_action, self.max_action)

                next_obs, reward, termination, truncation, info = self.env.step(
                    step_action)

                eval_avg_reward += reward
                eval_mov_average = 0.999 * eval_mov_average + 0.001 * reward

                if self.render:
                    rendered = self.env.render()
                    for j in range(self.env.num_frames):
                        video_writer.append_data(rendered[j])
                    

                obs = torch.Tensor(next_obs).to(self.device)

                self.collector.collect("eval_reward", reward)
                self.collector.collect("eval_moving_avg", reward)
        
        return eval_avg_reward, eval_mov_average

    @staticmethod
    def collector_init(config):
        config['lr'] = Identity()
        config['value_loss'] = Identity()
        config['policy_loss'] = Identity()
        config['entropy_loss'] = Identity()
        config['old_approx_kl'] = Identity()
        config['approx_kl'] = Identity()
        config['clipfrac'] = Identity()
        config['explained_variance'] = Identity()

        return Collector(config)

    def save_collector(self, exp, save_path):
        from PyExpUtils.results.sqlite import saveCollector
        self.collector.reset()
        saveCollector(exp, self.collector, base=save_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_checkpoint(self, path):
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    