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
from PyExpUtils.collection.Sampler import Subsample, Identity
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.results.sqlite import saveCollector

from src.experiment import ExperimentModel
from src.experiment.RLAgent import RLAgent

import warnings
warnings.filterwarnings("ignore")

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True,
                    help="The Json File of DQN, PPO or SAC")
parser.add_argument('-i', '--idxs', nargs='+', type=int,
                    required=True, help="The indices of the runs to train: Seeds")
parser.add_argument('--save_path', type=str, default=f'{os.getcwd()}')
parser.add_argument('--checkpoint_path', type=str,
                    default=f'{os.getcwd()}/checkpoints')
parser.add_argument('--silent', action='store_true', default=False)
parser.add_argument('--track', action='store_true', default=False)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--checkpoint', action='store_true', default=False)

args = parser.parse_args()

env_config = json.load(open('env_config.json', 'r'))
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('exp')
logging.getLogger('jax').setLevel(logging.ERROR)
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
        'reward': Identity(),
        'steps': Identity(),
        'eval_reward': Identity(),
        'episode_reward': Identity()
    }

    collector = Collector(collector_config)
    collector.setIdx(idx)

    run = exp.getRun(idx)

    # set random seeds accordingly
    random.seed(idx)
    np.random.seed(idx)
    torch.manual_seed(idx)
    torch.set_num_threads(8) # set the number of threads for torch
    torch.backends.cudnn.deterministic = True

    exp_name = '/'.join(exp.path.replace('.json', '').split('/')[1:])
    checkpoint_path = f'{args.checkpoint_path}/{exp_name}'
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = f'{checkpoint_path}/{idx}.zip'

    render_path = f'{args.save_path}/videos/{exp_name}'
    os.makedirs(render_path, exist_ok=True)
    import pdb; pdb.set_trace()


    # Run the experiment
    start_time = time.time()
    rl_agent = RLAgent(exp, idx, env_config=env_config,  device=device,
                       collector=collector, render=args.render)

    if os.path.exists(checkpoint_path) and args.checkpoint:
        print(f'Loading checkpoint from {checkpoint_path}')
        rl_agent.load_checkpoint(checkpoint_path)

    score, last_obs = rl_agent.train()
    logger.debug(f'Train: {time.time() - start_time:.2f}s and scored {score}')

    eval_time = time.time()
    score = rl_agent.eval(save_path=f'{render_path}/{idx}.mp4')

    logger.debug(
        f'Eval: {time.time() - eval_time:.2f}s and scored {score}, saved at {render_path}/{idx}.mp4')

    # save the results
    collector.reset()
    saveCollector(exp, collector, base=args.save_path)

    # save checkpoint
    rl_agent.save_checkpoint(checkpoint_path)
