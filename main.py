import random
import time
import os

import numpy as np
import torch
import argparse
import logging
import json
import socket

# saving the results of the experiment
from PyExpUtils.collection.Sampler import MovingAverage, Subsample, Identity
from PyExpUtils.collection.utils import Pipe
from src.experiment import ExperimentModel
from src.experiment.RLAgent import RLAgent


# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
parser.add_argument('--save_path', type=str, default=f'{os.getcwd()}/')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
parser.add_argument('--silent', action='store_true', default=False)
parser.add_argument('--render', action='store_true', default=True)
parser.add_argument('--track', action='store_true', default=False)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

env_config = json.load(open('env_config.json', 'r'))
device = torch.device("cuda" if torch.cuda.is_available()
                      and args.gpu else "cpu")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('exp')
prod = 'cdr' in socket.gethostname() or args.silent
if not prod:
    logger.setLevel(logging.DEBUG)

# ----------------------
# -- Experiment Def'n --
# ----------------------
exp = ExperimentModel.load(args.exp)
indices = args.idxs

for idx in indices:
    collector_config = {
        'SPS': Identity(),
        'reward': Identity(),
        'moving_avg': Pipe(
            MovingAverage(0.999),
            Subsample(500),
        ),
        'eval_reward': Identity(),
        'eval_moving_avg': Pipe(
            MovingAverage(0.999),
            Subsample(500),
        ),
    }

    run = exp.getRun(idx)

    # set random seeds accordingly
    random.seed(idx)
    np.random.seed(idx)
    torch.manual_seed(idx)
    torch.backends.cudnn.deterministic = True

    if args.track:
        import wandb
        wandb.init(
            project=exp.name,
            sync_tensorboard=True,
            config=vars(args),
            name=f'{exp.name}-{idx}',
            monitor_gym=True,
            save_code=True,
        )

    # Run the experiment
    start_time = time.time()

    rl_agent = RLAgent(exp, idx, env_config=env_config,  device=device,
                       collector_config=collector_config, render=args.render)
    rl_agent.load_checkpoint('/home/mamm/ayman/thesis/AgarLE-benchmark/checkpoints/run_1.pt')

    # last_obs = rl_agent.train()
    eval_avg_reward, eval_mov_average  = rl_agent.eval()

    print(f'Run {idx} took {time.time() - start_time:.2f}s')
    print(f'Eval Avg Reward: {eval_avg_reward:.2f}')
    print(f'Eval Moving Avg Reward: {eval_mov_average:.2f}')
    
    # rl_agent.save_collector(exp, args.save_path)
    # full_path = os.path.join(args.checkpoint_path, f'run_{idx}.pt')
    # os.makedirs(os.path.dirname(full_path), exist_ok=True)
    # rl_agent.save_checkpoint(full_path)