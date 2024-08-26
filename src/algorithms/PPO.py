import gymnasium as gym
import time
# import gym
import gym_agario
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from src.wrappers.gym import *
from src.utils import modify_action, modify_hybrid_action


def make_env(env_name, config, gamma):
    env = gym.make(env_name, **config)
    # deal with dm_control's Dict observation space
    env = FlattenObservation(env)
    env = NormalizeObservation(env)
    env = TransformObservation(
        env, lambda obs: np.clip(obs, -10, 10))
    env = NormalizeReward(env, gamma=gamma)
    env = TransformReward(
        env, lambda reward: np.clip(reward, -10, 10))
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
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
    def __init__(self, obs_shape, cont_action_shape, dis_action_shape):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(cont_action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(cont_action_shape)))
        self.action_dim = cont_action_shape

        self.dis_actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, dis_action_shape), std=0.01),
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


class PPO:
    def __init__(self,
                 run_name: str,
                 env_config: dict,
                 total_timesteps: int = 1e6,
                 eval_timesteps: int = 3000,
                 env_name: str = "agario-grid-v0",
                 cuda: bool = False,
                 hybrid: bool = False,
                 render: bool = False,
                 anneal_lr: bool = True,
                 num_envs: int = 1,
                 num_steps: int = 2048,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_minibatches: int = 32,
                 update_epochs: int = 10,
                 norm_adv: bool = True,
                 clip_coef: float = 0.2,
                 clip_vloss: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = None
                 ) -> None:

        self.env = make_env(env_name, env_config,  gamma)

        self.writer = SummaryWriter(f"runs/{run_name}")
        self.cuda = cuda
        self.hybrid = hybrid
        self.render = render
        self.total_timesteps = total_timesteps
        self.eval_timesteps = eval_timesteps

        self.num_bots = env_config['num_bots']
        self.run_name = run_name

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # to be filled in runtime
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   and self.cuda else "cpu")

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
            self.agent = HybridAgent(
                self.obs_shape, cont_action_shape, dis_action_shape).to(self.device)
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
        moving_avg = 0
        cum_reward = 0
        bot_moving_avg = np.zeros(self.num_bots)
        bot_cum_reward = np.zeros(self.num_bots)

        start_time = time.time()
        next_obs = self.env.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        for iteration in range(1, self.num_iterations + 1):
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

                self.actions[step] = action
                self.logprobs[step] = logprob
                action = action.detach().cpu().numpy().squeeze()

                if self.hybrid:
                    step_action = modify_hybrid_action(action)
                else:
                    step_action = modify_action(
                        action, self.min_action, self.max_action)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, termination, truncation, info = self.env.step(
                    step_action)
                
                reward = rewards[0]
                next_done = np.ones((1,)) * termination
                self.rewards[step] = torch.tensor(
                    reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device), torch.Tensor(next_done).to(self.device)

                cum_reward += reward
                moving_avg = 0.99 * moving_avg + 0.01 * reward

                # self.writer.add_scalar("cum_reward", cum_reward, global_step)
                # self.writer.add_scalar("avg_reward", cum_reward / global_step, global_step)
                # self.writer.add_scalar("moving_avg", moving_avg, global_step)
                for idx, rew in enumerate(rewards[1:]):
                    bot_cum_reward[idx] += rew
                    bot_moving_avg[idx] = 0.99 * bot_moving_avg[idx] + 0.01 * rew

                if global_step % 1000 == 0:
                    with open(f"runs/{self.run_name}/cum_reward.csv", "a") as f:
                        f.write(f"{global_step},{cum_reward}\n")
                    with open(f"runs/{self.run_name}/avg_reward.csv", "a") as f:
                        f.write(f"{global_step},{cum_reward / global_step}\n")
                    with open(f"runs/{self.run_name}/moving_avg.csv", "a") as f:
                        f.write(f"{global_step},{moving_avg}\n")

                    for idx, rew in enumerate(rewards[1:]):
                    #     bot_cum_reward[idx] += rew
                    #     bot_moving_avg[idx] = 0.99 * bot_moving_avg[idx] + 0.01 * rew
                    #     self.writer.add_scalar(
                    #         f"cum_reward_{idx}", bot_cum_reward[idx], global_step)
                    #     self.writer.add_scalar(
                    #         f"avg_reward_{idx}", bot_cum_reward[idx] / global_step, global_step)
                    #     self.writer.add_scalar(
                    #         f"moving_avg_{idx}", bot_moving_avg[idx], global_step)
                        
                        with open(f"runs/{self.run_name}/cum_reward_{idx}.csv", "a") as f:
                            f.write(f"{global_step},{bot_cum_reward[idx]}\n")
                        with open(f"runs/{self.run_name}/avg_reward_{idx}.csv", "a") as f:
                            f.write(f"{global_step},{bot_cum_reward[idx] / global_step}\n")
                        with open(f"runs/{self.run_name}/moving_avg_{idx}.csv", "a") as f:
                                f.write(f"{global_step},{bot_moving_avg[idx]}\n")

                

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
            self.writer.add_scalar("charts/learning_rate",
                                   self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss",
                                   v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss",
                                   pg_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl",
                                   old_approx_kl.item(), global_step)
            self.writer.add_scalar(
                "losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar(
                "losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar(
                "losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step /
                                                     (time.time() - start_time)), global_step)

        return torch.Tensor(next_obs).to(self.device)

    def eval(self, obs):
        eval_avg_reward = 0
        eval_mov_average = 0
        for _ in range(self.eval_timesteps):
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(
                    obs, action_limits=(self.min_action, self.max_action))
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

        print(f'Average reward from evaluation: {eval_avg_reward}')
        print(f'Exp. moving reward from evaluation: {eval_mov_average}')

        self.env.close()
        self.writer.close()
