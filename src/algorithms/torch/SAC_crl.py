import os
import random
import cv2
import cv2
import time
# import modules.agar.gym_agario 
from dataclasses import dataclass

import gymnasium as gym
from gym.spaces import flatten_space, flatten, unflatten
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer

from src.wrappers.gym import SB3Wrapper, ModifyActionWrapperCRL, FlattenObservationWrapper, FlattenObservation
from src.wrappers.gym import SB3Wrapper, ModifyActionWrapperCRL, FlattenObservationWrapper, FlattenObservation
from torch.utils.tensorboard import SummaryWriter
#from src.wrappers.gym import make_env

from cleanrl.cleanrl.sac_continuous_action import Args


class MultiActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,)),  # (dx, dy) movement vector
    def action(self, action):
        return (action, 0)  # no-op on the second action

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.observation_space.shape[3], self.observation_space.shape[1], self.observation_space.shape[2]), dtype=np.uint8)

    def observation(self, observation):
        normalized = (observation - np.mean(observation) )/ np.std(observation)
        return normalized.transpose(0, 3, 1, 2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.transpose(0, 3, 1, 2), info
    
def eval_agent(env, actor, save_path="agent_playing.mp4", eval_steps=500):
    obs, _ = env.reset()
    cumulative_reward = 0


    height, width = (84,84)

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

   
    for _ in range(eval_steps):
        action, _, _ = actor.get_action(  torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, -1) )
        action = action.cpu().detach().numpy()  # Convert to NumPy for env step

        # Convert grayscale to RGB if needed
        image = obs.astype(np.uint8)
        if len(image.shape) == 2:  
            # image = image.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W) for RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # image = image.permute(1, 2, 0).cpu().numpy()  # Convert (3, H, W) → (H, W, 3)

        video.write( np.ascontiguousarray(image) )  # Save frame

        print( action.shape )

        obs, reward, done, trunc, _ = env.step(action.squeeze())
        cumulative_reward += reward

        if done or trunc:
            obs,_ = env.reset()

    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved at {save_path}")
    return cumulative_reward
class MultiActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))  # (dx, dy) movement vector
        self.single_action_space = self.action_space
    def action(self, action):
        return (action,0)  # no-op on the second action
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Modify observation space if needed
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.observation_space.shape[3], self.observation_space.shape[1], self.observation_space.shape[2]), dtype=np.uint8)
        self.single_observation_space = self.observation_space
    def observation(self, observation):
        return observation.transpose(0, 3, 1, 2)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.transpose(0, 3, 1, 2), info

def eval_agent(env, actor, save_path="agent_playing.mp4", eval_steps=500):
    obs, _ = env.reset()
    cumulative_reward = 0


    height, width = (84,84)

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

   
    for _ in range(eval_steps):
        action, _, _ = actor.get_action(  torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, -1) )
        action = action.cpu().detach().numpy()  # Convert to NumPy for env step

        # Convert grayscale to RGB if needed
        image = obs.astype(np.uint8)
        if len(image.shape) == 2:  
            # image = image.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W) for RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # image = image.permute(1, 2, 0).cpu().numpy()  # Convert (3, H, W) → (H, W, 3)

        video.write( np.ascontiguousarray(image) )  # Save frame

        print( action.shape )

        obs, reward, done, trunc, _ = env.step(action.squeeze())
        cumulative_reward += reward

        if done or trunc:
            obs,_ = env.reset()

    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved at {save_path}")
    return cumulative_reward

def make_env(env_id, seed, idx, capture_video, run_name, **kwargs):
    #return gym.make(env_id, **kwargs)
    def thunk():
        if capture_video and idx == 0:
            if ( env_id == "agario-screen-v0" ):
                env = gym.make(env_id, **kwargs)
                env = SB3Wrapper(env)
                env = ModifyActionWrapperCRL(env)
                env = FlattenObservation(env)
                env = gym.wrappers.RecordEpisodeStatistics(env)
            else:
                env = gym.make(env_id, render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if ( env_id == "agario-screen-v0" ):
                env = gym.make(env_id, **kwargs)
                # env = SB3Wrapper(env)
                # env = ModifyActionWrapperCRL(env)
                # env = FlattenObservation(env)
                # env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                # env = gym.wrappers.NormalizeObservation(env)
                # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
                env = MultiActionWrapper(env)
                # env = FlattenActionWrapper(env)
                env = ObservationWrapper(env)
            else:
                env = gym.make(env_id, render_mode='rgb_array')

        # env.action_space.seed(seed)
  
        obs = env.reset()
        # print(f"Reset output: {obs}")  # Debugging
        return env

    return thunk()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=envs.observation_space.shape[0], out_channels=64, kernel_size=16, stride=1)),
            nn.LayerNorm([64, 113, 113]),  # Add LayerNorm after the first Conv2d layer
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1)),
            nn.LayerNorm([64, 106, 106]),  # Add LayerNorm after the second Conv2d layer
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 106* 106, 256)),
            nn.LayerNorm(256),  # Add LayerNorm after the first Linear layer
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *env.single_observation_space.shape)
            conv_out = self.conv(dummy_input)

        conv_out_size = conv_out.shape[1]
        
        box = env.action_space
        box_dim = int(np.prod(box.shape))

        # For the Discrete space:
        # discrete = env.action_space.spaces[1]
        # discrete_dim = discrete.n 

        total_action_dim = box_dim  #+ discrete_dim
        
        # Fully connected part: takes concatenated conv features and action vector.
        self.fc = nn.Sequential(
            layer_init(nn.Linear(conv_out_size + total_action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def forward(self, x, a):
        #Assumes discrete aciton is converted to one-hot encoding
        x = self.conv(x)
        # Concatenate the flattened features with the action vector
        x = torch.cat([x, a], dim=1)
        q_value = self.fc(x)
        return q_value

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
   
        self.actor_mean = nn.Sequential(
            nn.Conv2d(in_channels=envs.observation_space.shape[0], out_channels=64, kernel_size=16, stride=1),
            nn.LayerNorm([64, 113, 113]),  # Add LayerNorm after the first Conv2d layer
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1),  # Additional Conv2d layer
            nn.LayerNorm([64, 106, 106]),  # Add LayerNorm after the additional Conv2d layer
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 106* 106, 256)),
            nn.LayerNorm(256),  # Add LayerNorm after the first Linear layer
            nn.ReLU(),
            layer_init(nn.Linear(256, 2)) # 2 continuous actions
        )
        
        self.fc_logstd = nn.Parameter(torch.zeros(1,2))

        # action rescaling
        # self.register_buffer(
        #     "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        # )
        # self.register_buffer(
        #     "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        # )

    def forward(self, x):
        mean = self.actor_mean(x)
        log_std = self.fc_logstd.expand_as(mean)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # action[:, 2] = 0.0
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)

    args.env_id =  "agario-screen-v0" #"MountainCarContinuous-v0"
    args.env_id =  "agario-screen-v0" #"MountainCarContinuous-v0"

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
   # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    import json
    env_config = json.load(open('env_config.json', 'r'))

    envs = make_env(args.env_id, args.seed, 0, args.capture_video, run_name, **env_config)

    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # eval_env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name, **env_config)()

    # # Evaluate and save the agent's gameplay video
    # eval_agent(eval_env, actor, save_path="agario_agent.mp4")

    # eval_env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name, **env_config)()

    # # Evaluate and save the agent's gameplay video
    # eval_agent(eval_env, actor, save_path="agario_agent.mp4")

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

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
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()