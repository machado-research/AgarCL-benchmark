import random
import time
import os
from dataclasses import dataclass

import numpy as np
import torch
import tyro

from src.algorithms.SAC import SAC
from src.algorithms.PPO import PPO


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = False
    track: bool = True
    wandb_project_name: str = "Agar-PPO"
    wandb_entity: str = None
    render: bool = True
    hybrid: bool = False
    total_timesteps: int = 1000
    eval_timesteps: int = 3000


default_config = {
    'ticks_per_step':  4,
    'num_frames':      1,
    'arena_size':      500,
    'num_pellets':     500,
    'num_viruses':     20,
    'num_bots':        10,
    'pellet_regen':    True,
    'grid_size':       32,
    'screen_len': 512,
    'observe_cells':   False,
    'observe_others':  True,
    'observe_viruses': True,
    'observe_pellets': True,
    'obs_type': "grid",  # Two options: screen, grid
    'render_mode': "rgb_array", # Two options: human, rgb_array
    'allow_respawn': True,  # If False, the game will end when the player is eaten
    # Two options: "mass:reward=mass", "diff = reward=mass(t)-mass(t-1)"
    'reward_type': 1,
    'c_death': -100,  # reward = [diff or mass] - c_death if player is eaten
    'video_path': "screen_video.mp4" # for render_mode: rgb_array
}

args = tyro.cli(Args)
run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

# seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

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


rl_alg = PPO(
    total_timesteps=args.total_timesteps,
    eval_timesteps=args.eval_timesteps,
    env_name="agario-screen-v0",
    env_config=default_config,
    run_name=run_name,
    cuda=args.cuda,
    hybrid=args.hybrid,
    render=args.render,
    anneal_lr=True,
    num_envs=1,
    num_steps=2048,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=10,
    norm_adv=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None
)

last_obs = rl_alg.train()
rl_alg.eval(last_obs)
