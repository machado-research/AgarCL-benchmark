import random
import time
import os

import numpy as np
import torch
import argparse
import logging
import json
import socket
# import jax

# saving the results of the experiment
from PyExpUtils.collection.Sampler import MovingAverage, Subsample, Identity
from PyExpUtils.collection.utils import Pipe
from src.experiment import ExperimentModel
from src.experiment.RLAgent import RLAgent

import warnings
warnings.filterwarnings("ignore")

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True, help = "The Json File of PPO or SAC")
parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True, help = "The indices of the runs to train: Seeds")
parser.add_argument('--save_path', type=str, default=f'{os.getcwd()}/')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
parser.add_argument('--silent', action='store_true', default=False)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--gpu', action='store_true', default=False)

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
        'SPS': Identity(),
        'reward': Subsample(500),
        'moving_avg': Pipe(
            MovingAverage(0.999),
            Subsample(500),
        ),
        'l2_norm': Identity(),
        'activation_norm': Identity(),
        'spectral_norm': Identity(),
        'spectral_norm_grad': Identity(),
        'hidden_stable_rank': Identity(),
        'stable_weight_rank': Identity(),
        'dormant_units': Identity(),
    }

    run = exp.getRun(idx)

    # set random seeds accordingly
    random.seed(idx)
    np.random.seed(idx)
    torch.manual_seed(idx)
    torch.backends.cudnn.deterministic = True
    

    # Run the experiment
    start_time = time.time()

    rl_agent = RLAgent(exp, idx, env_config=env_config,  device=device,
                       collector_config=collector_config, render=args.render)
    
    score, last_obs = rl_agent.train()
    
    save_path = f'{args.save_path}/results'
    rl_agent.eval(last_obs, save_path=save_path)

    logger.debug(f'Run {idx} took {time.time() - start_time:.2f}s and scored {score}')
    rl_agent.save_collector(exp, args.save_path)
    
