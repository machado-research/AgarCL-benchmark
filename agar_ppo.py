import os
import random
import time
from dataclasses import dataclass

# import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import gym
import gym_agario

from src.algorithms.PPO import *
from src.wrappers.gym import *


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "Agar-PPO"
    wandb_entity: str = None

    # Algorithm specific arguments
    total_timesteps: int = 10000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


default_config = {
    'ticks_per_step':  4,
    'num_frames':      1,
    'arena_size':      500,
    'num_pellets':     1000,
    'num_viruses':     25,
    'num_bots':        10,
    'pellet_regen':    True,
    'grid_size':       25,
    'observe_cells':   False,
    'observe_others':  True,
    'observe_viruses': True,
    'observe_pellets': True,
    'obs_type': "grid",  # Two options: screen, grid
    'allow_respawn': True,  # If False, the game will end when the player is eaten
    # Two options: "mass:reward=mass", "diff = reward=mass(t)-mass(t-1)"
    'reward_type': 1,
    'c_death': -100,  # reward = [diff or mass] - c_death if player is eaten
}


args = tyro.cli(Args)
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size
run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

env = gym.make('agario-grid-v0', **default_config)
env = NormalizeObservation(env)
env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
env = NormalizeReward(env, gamma=args.gamma)
env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))

if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

device = torch.device("cuda" if torch.cuda.is_available()
                      and args.cuda else "cpu")

obs_shape = (np.prod(env.observation_space.shape),)
env.observation_space = gym.spaces.Box(-1,
                                       np.iinfo(np.int32).max, obs_shape, dtype=np.float32)

env.action_space = (gym.spaces.Box(-10, 10, env.action_space[0].shape, dtype=np.float32),
                    gym.spaces.Discrete(3))
action_shape = env.action_space[0].shape
max_action = float(env.action_space[0].high[0])
min_action = float(env.action_space[0].low[0])

agent = Agent(obs_shape, action_shape).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# ALGO Logic: Storage setup
obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) +
                      action_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
avg_reward = 0
moving_avg = 0

start_time = time.time()
next_obs = env.reset()
next_obs = torch.Tensor(next_obs.flatten()).to(device)
next_done = torch.zeros(args.num_envs).to(device)

for iteration in range(1, args.num_iterations + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(
                next_obs, action_limits=(min_action, max_action))
            values[step] = value.flatten()

        actions[step] = action
        logprobs[step] = logprob
        action = action.detach().cpu().numpy().squeeze()

        step_action = ([(action), 0])
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, info = env.step(
            step_action)
        # env.render()
        next_done = np.ones((1,)) * termination
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs.flatten()).to(
            device), torch.Tensor(next_done).to(device)

        avg_reward += reward
        moving_avg = 0.99 * moving_avg + 0.01 * reward

        writer.add_scalar("avg_reward", avg_reward, global_step)
        writer.add_scalar("moving_avg", moving_avg, global_step)

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * \
                nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * \
                args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + obs_shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds], action_limits=(min_action, max_action))
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() >
                               args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()
                                 ) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * \
                torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - \
        np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate",
                      optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl",
                      old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step /
                      (time.time() - start_time)), global_step)


env.close()
writer.close()
