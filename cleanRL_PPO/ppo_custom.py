# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
# from torch.distributions.normal import Normal
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
import gym_agario 
import json
from custom_cleanrl_utils.evals.ppo_eval import evaluate
from custom_cleanrl_utils.huggingface import push_to_hub

import cv2

class MultiActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # self.action_space = gym.spaces.Tuple((
        #     gym.spaces.Box(low=-1, high=1, shape=(2,)),  # (dx, dy) movement vector
        #     gym.spaces.Discrete(3),                      # 0=noop, 1=split, 2=feed
        # ))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,)),  # (dx, dy) movement vector
    def action(self, action):
        return (action, 0)  # no-op on the second action

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Modify observation space if needed
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.observation_space.shape[3], self.observation_space.shape[1], self.observation_space.shape[2]), dtype=np.uint8)

    def observation(self, observation):
        return observation.transpose(0, 3, 1, 2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.transpose(0, 3, 1, 2), info

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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "agario-screen-v0"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 15
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.3
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 100
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    env_config = json.load(open('/home/mamm/ayman/thesis/AgarLE-benchmark/env_config.json', 'r'))
    env = gym.make(env_id, **env_config)
    # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = MultiActionWrapper(env)
    env = ObservationWrapper(env)
    # env = gym.wrappers.ClipAction(env)
    
    # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
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
        
        # self.actor_logstd = nn.Parameter(torch.zeros(1, 3))
         # Separate outputs for continuous and discrete actions
        # self.actor_mean = layer_init(nn.Linear(256, 2))  # 2 continuous actions
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        # self.actor_discrete = layer_init(nn.Linear(256, 3))  # 3 discrete actions (logits)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            
        return torch.tanh(action), probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    # def get_action_and_value(self, x, action=None):
    #     features = self.actor_base(x)
        
    #     # Continuous actions
    #     action_mean = self.actor_mean(features)
    #     action_std = torch.exp(self.actor_logstd)  # Ensure std is positive
    #     continuous_dist = Normal(action_mean, action_std)
        
    #     # Discrete actions
    #     action_logits = self.actor_discrete(features)
    #     discrete_dist = Categorical(logits=action_logits)

    #     if action is None:
    #         continuous_action = continuous_dist.sample()
    #         discrete_action = discrete_dist.sample()
    #     else:
    #         action = action.reshape(-1,action.shape[-1])
    #         continuous_action, discrete_action = action[:, :2], action[:, 2]
        
    #     #Clip continuous action to the valid range [-1,1]
    #     continuous_action = torch.tanh(continuous_action)
    #     # Compute log probabilities
    #     continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(-1)
    #     discrete_log_prob = discrete_dist.log_prob(discrete_action)
    #     log_prob = continuous_log_prob + discrete_log_prob  # Sum log probabilities
        
    #     # Compute entropy for PPO updates
    #     entropy = continuous_dist.entropy().sum(-1) + discrete_dist.entropy()

    #     return (continuous_action, discrete_action), log_prob, entropy, self.critic(x)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    # )
    envs = [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    envs = envs[0]
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_shape = tuple(envs.observation_space.shape)
    # transposed_obs_shape = (obs_shape[0], obs_shape[3]) + obs_shape[1:3]
    obs = torch.zeros((args.num_steps,) + obs_shape).to(device)
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    actions   = torch.zeros((args.num_steps, args.num_envs) + (2,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    best_reward = -np.inf
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

            if(next_done == True):
                next_obs, _ = envs.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action[0].cpu().numpy())
            next_done = np.logical_or(terminations, truncations).astype(np.float32)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.tensor(next_obs, dtype=torch.float32).to(device), torch.tensor(next_done, dtype=torch.float32).to(device)
            # if "final_info" in infos:
                # for info in infos["final_info"]:
                    # if info and "episode" in info:

                        # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
   
                
        # import pdb; pdb.set_trace()

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
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1,actions.shape[-1])
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
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # Print the metrics
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Entropy: {entropy_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clipfracs)}")
        print(f"Explained Variance: {explained_var}")
        print("SPS:", int(global_step / (time.time() - start_time)))

        # Write the metrics to a CSV file
        with open(f"runs/{run_name}/metrics.csv", "a") as f:
            f.write(f"{global_step},{optimizer.param_groups[0]['lr']},{v_loss.item()},{pg_loss.item()},{entropy_loss.item()},{old_approx_kl.item()},{approx_kl.item()},{np.mean(clipfracs)},{explained_var},{int(global_step / (time.time() - start_time))},{rewards.sum().item()}\n")
        # Save the model if there is no model or if the rewards are better
        
        if rewards.sum().item() > best_reward:
            best_reward = rewards.sum().item()
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Log the metrics to TensorBoard
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     torch.save(agent.state_dict(), model_path)
    #     print(f"model saved to {model_path}")
        

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
            env=envs,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()