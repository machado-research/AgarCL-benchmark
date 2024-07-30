import random
import time
import os
from dataclasses import dataclass

import gym.spaces
import gym.spaces
import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import gym
import gym_agario

from src.algorithms.SAC import *
from src.wrappers.gym import FlattenObservation
from src.utils import modify_hybrid_action

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = False
    track: bool = False
    wandb_project_name: str = "Agar-SAC"
    wandb_entity: str = None
    render: bool = True

    # Algorithm specific arguments
    total_timesteps: int = 5000
    eval_timesteps: int = 1000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 1e3
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    # Denis Yarats' implementation delays this by 2.
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True


default_config = {
    'ticks_per_step':  4,
    'num_frames':      1,
    'arena_size':      500,
    'num_pellets':     500,
    'num_viruses':     20,
    'num_bots':        10,
    'pellet_regen':    True,
    'grid_size':       32,
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
run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

env = gym.make('agario-grid-v0', **default_config)
env = FlattenObservation(env)

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

env.action_space = (gym.spaces.Box(-1, 1, env.action_space[0].shape, dtype=np.float32),
                    gym.spaces.Discrete(3))

cont_action_shape = env.action_space[0].shape[0]
dis_action_shape = env.action_space[1].n

max_action = float(env.action_space[0].high[0])
min_action = float(env.action_space[0].low[0])

action_shape = (cont_action_shape + 1,)
obs_shape = (np.prod(env.observation_space.shape),)
env.observation_space.dtype = np.float32

actor = HybridActor(cont_action_shape, dis_action_shape, obs_shape, min_action, max_action).to(device)
qf1 = SoftQNetwork(obs_shape, action_shape).to(device)
qf2 = SoftQNetwork(obs_shape, action_shape).to(device)
qf1_target = SoftQNetwork(obs_shape, action_shape).to(device)
qf2_target = SoftQNetwork(obs_shape, action_shape).to(device)

qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()) +
                         list(qf2.parameters()), lr=args.q_lr)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

# Automatic entropy tuning
if args.autotune:
    target_entropy = - \
        torch.prod(torch.Tensor(env.action_space[0].shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha

rb = ReplayBuffer(
    args.buffer_size,
    env.observation_space,
    gym.spaces.Box(-1, 1, action_shape, dtype=np.float32),
    device,
    handle_timeout_termination=False,
)
start_time = time.time()
avg_reward = 0
moving_avg = 0

# TRY NOT TO MODIFY: start the game
obs = env.reset()
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    # if global_step < args.learning_starts:
    #     action = gym.spaces.Box(-1, 1, action_shape, dtype=np.float32).sample()
    # else:
    action, _, _ = actor.get_action(torch.Tensor(obs).to(device))
    action = action.detach().cpu().numpy()
    
    step_action = modify_hybrid_action(action)

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, termination, info = env.step(step_action)

    avg_reward += reward
    moving_avg = 0.99 * moving_avg + 0.01 * reward

    writer.add_scalar("avg_reward", avg_reward, global_step)
    writer.add_scalar("moving_avg", moving_avg, global_step)

    # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
    rb.add(obs, next_obs, action, reward, termination, info)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(
                data.next_observations)
            qf1_next_target = qf1_target(
                data.next_observations, next_state_actions)
            qf2_next_target = qf2_target(
                data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(
                qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * \
                args.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = qf1(data.observations, data.actions).view(-1)
        qf2_a_values = qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = actor.get_action(data.observations)
                qf1_pi = qf1(data.observations, pi)
                qf2_pi = qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (-log_alpha.exp() *
                                  (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # update the target networks
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)

        if global_step % 100 == 0:
            writer.add_scalar("losses/qf1_values",
                              qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values",
                              qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss",
                              qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss",
                              actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step /
                              (time.time() - start_time)), global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss",
                                  alpha_loss.item(), global_step)
                
for eval_step in range(args.eval_timesteps):
    with torch.no_grad():
        action, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        action = action.detach().cpu().numpy()
        
        step_action = modify_hybrid_action(action)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, info = env.step(
            step_action)
        if args.render:
            env.render()
        
        next_obs = torch.Tensor(next_obs).to(device)

env.close()
writer.close()
