# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule
from pfrl.initializers import init_chainer_default

# from stable_baselines3.common.buffers import ReplayBuffer
from torchrl.data import LazyTensorStorage, ReplayBuffer

import gym_agario
import json 
import csv

class MultiActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))  # (dx, dy) movement vector
        self.single_action_space = self.action_space
    def action(self, action):
        return (action, 0)  # no-op on the second action

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.observation_space.shape[3], self.observation_space.shape[1], self.observation_space.shape[2]), dtype=np.uint8)
        self.single_observation_space = self.observation_space
    def observation(self, observation):
        return observation.transpose(0, 3, 1, 2).astype(np.float32)/255.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.transpose(0, 3, 1, 2).astype(np.float32)/255.0, info


class CustomCNN(nn.Module):
        def __init__(self, n_input_channels, n_output_channels, activation=nn.ReLU(), bias=0.1, device=None):
            super().__init__()
            self.n_input_channels = n_input_channels
            self.activation = activation
            self.n_output_channels = n_output_channels
            self.layers = nn.ModuleList(
                [
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, device=device),
                    nn.LayerNorm([32, 31, 31], device=device),
                    nn.Conv2d(32, 64, 4, stride=2, device=device),
                    nn.LayerNorm([64, 14, 14], device=device),
                    nn.Conv2d(64, 32, 3, stride=1, device=device),
                    nn.LayerNorm([32, 12, 12], device=device),
                    nn.Flatten(),
                ]
            )
            self.output = nn.Linear(32 * 12 * 12, n_output_channels, device=device)  # Adjusted for 3x84x84 input

            # self.apply(torch.nn.init.xavier_uniform_)
            self.apply(init_chainer_default)
            self.apply(self.constant_bias_initializer(bias=bias))

        def constant_bias_initializer(self, bias=0.1):
            def init(m):
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, bias)
            return init

        def forward(self, state):
            h = state
            for layer in self.layers:
                h = self.activation(layer(h))
            h_flat = h.view(h.size(0), -1)
            return self.activation(self.output(h_flat))

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "agario-screen-v0"
    """the environment id of the task"""
    total_timesteps: int = 2 * int(1e6)
    """total timesteps of the experiments"""
    buffer_size: int = int(2e4)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 5e-3
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 1e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""


def make_env(env_id, seed, idx, capture_video, run_name):
    env_config = json.load(open('env_config.json', 'r'))
    env = gym.make(env_id, **env_config)
    env = MultiActionWrapper(env)
    env = ObservationWrapper(env)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, n_act, n_obs, device=None, feature_dim=64):
        super().__init__()
        # self.fc1 = nn.Linear(n_act + n_obs, 256, device=device)
        # self.fc2 = nn.Linear(256, 256, device=device)
        # self.fc3 = nn.Linear(256, 1, device=device)
        self.conv = nn.Sequential(
                CustomCNN(n_input_channels=n_obs, n_output_channels=256, device=device),
                nn.ReLU(),
            )
        conv_output_size = 256
            
            # Fully connected layers that combine the flattened conv features and action
        self.fc1 = init_chainer_default(nn.Linear(conv_output_size + n_act, feature_dim, device=device))
        self.fc2 = init_chainer_default(nn.Linear(feature_dim, 1, device=device))  # Output a single Q-valu

    def forward(self, state, action):
        conv_out = self.conv(state)
        conv_out = conv_out.view(conv_out.size(0), -1)
        x = torch.cat([conv_out, action], dim=-1)  # [batch, conv_features + action_dim]
        x = self.fc1(x)
        x = nn.ReLU()(x)
        q_value = self.fc2(x)  # [batch, 1]
        return q_value


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, n_obs, n_act, device=None):
        super().__init__()
        self.fc1 = CustomCNN(n_input_channels=n_obs, n_output_channels=256, device=device)
        self.fc_mean = init_chainer_default(nn.Linear(256, n_act, device=device))
        self.fc_logstd = init_chainer_default(nn.Linear(256, n_act, device=device))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32, device=device),
        )

    def forward(self, x):
        x = self.fc1(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
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
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # import pdb; pdb.set_trace()
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    # wandb.init(
    #     project="sac_continuous_action",
    #     name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
    #     config=vars(args),
    #     save_code=True,
    # )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    envs = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    # n_act = math.prod(envs.single_action_space.shape)
    # n_obs = math.prod(envs.single_observation_space.shape)
    n_act = 2
    n_obs = 4
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    actor = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    actor_detach = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(actor_detach.get_action, in_keys=["observation"], out_keys=["action"])

    def get_q_params():
        qf1 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qf2 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()

        # discard params of net
        qnet = SoftQNetwork(envs, device="meta", n_act=n_act, n_obs=n_obs)
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()

    q_optimizer = optim.Adam(qnet.parameters(), lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, capturable=args.cudagraphs and not args.compile)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    def update_main(data):
        # optimize the model
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, data["next_observations"], next_state_actions
            )
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (
                ~data["dones"].flatten()
            ).float() * args.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, data["observations"], data["actions"], next_q_value
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_pol(data):
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, data["observations"], pi)
        min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_loss.backward()
        actor_optimizer.step()

        if args.autotune:
            a_optimizer.zero_grad()
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(data["observations"])
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            alpha_loss.backward()
            a_optimizer.step()
        return TensorDict(alpha=alpha.detach(), actor_loss=actor_loss.detach(), alpha_loss=alpha_loss.detach())

    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)

    is_extend_compiled = False
    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])
        # policy = CudaGraphModule(policy)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    episodic_return = 0
    episode = 0 # episode counter
    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = envs.single_action_space.sample()
        else:
            actions = policy(obs)
            actions = actions.cpu().numpy()[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        # import pdb; pdb.set_trace()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        episodic_return += infos['untransformed_rewards']
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         r = float(info["episode"]["r"])
        #         max_ep_ret = max(max_ep_ret, r)
        #         avg_returns.append(r)
        #     desc = (
        #         f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
        #     )
        

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        # real_next_obs = next_obs.clone()
        # for idx, trunc in enumerate([truncations]):
        #     if trunc:
        #         real_next_obs[idx] = torch.as_tensor(infos["final_observation"][idx], device=device, dtype=torch.float)
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
        transition = TensorDict(
            observations=obs,
            next_observations=next_obs,
            actions=torch.as_tensor(actions, dtype=torch.float32).reshape(1,-1),
            rewards=torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1),
            terminations=torch.as_tensor(terminations, dtype=torch.bool).reshape(-1, 1),
            dones=torch.as_tensor(terminations, dtype=torch.bool).reshape(-1, 1),
            batch_size=obs.shape[0],
            device = device,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        
        
        obs = next_obs
        data = extend_and_sample(transition)
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            out_main = update_main(data)
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    out_main.update(update_pol(data))

                    alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target.lerp_(qnet_params.data, args.tau)

            if global_step % 500 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "episode_return": episodic_return,
                        "actor_loss": out_main["actor_loss"].mean(),
                        "alpha_loss": out_main.get("alpha_loss", 0),
                        "qf_loss": out_main["qf_loss"].mean(),
                    }
                print("speed: ", speed, "logs: ", logs)
                # Save episodic_return in a CSV file
                with open(f'episode_returns_{args.seed}.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if global_step == args.learning_starts:
                        writer.writerow(['episode', 'reward'])  # Write header only once
                    writer.writerow([episode, episodic_return])
                episode += 1
        if terminations:
            obs, _ = envs.reset(seed=args.seed)
            obs = torch.as_tensor(obs, device=device, dtype=torch.float)
            episodic_return = 0
            
    envs.close()