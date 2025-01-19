import gym
import numpy as np
import torch
import logging
import json
from PyExpUtils.collection.Sampler import MovingAverage, Subsample, Identity
from PyExpUtils.collection.utils import Pipe
from src.experiment.RLAgent import RLAgent, get_agent
from src.experiment.ExperimentModel import ExperimentModel, load
import os
print(os.path.exists('experiments/mountain-car/sac.json'))

exp = load('experiments/mountain-car/sac.json')


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SAC Test')

# Environment and Hyperparameters
env_name = "MountainCarContinuous-v0"
total_timesteps = 10000
seed = 42

def main():
    # Set up the environment
    env = gym.make(env_name)
    env.action_space.seed(seed)  
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_config = json.load(open('env_config.json', 'r'))
    
    # Keep only specific keys
    env_config = {
        key: env_config[key] for key in ['screen_len', 'render_mode', 'reward_type'] if key in env_config
    }


    # Define hyperparameters
    hypers = {
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 64,
        'policy_frequency': 2,
        'gamma': 0.99,
        'tau': 0.005,
        'hidden_size': 256,
        'target_entropy': -1
    }

    #Copied from main.py
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

    # Initialize RLAgent
    logger.info("Initializing RLAgent with SAC...")
    rl_agent = None

    rl_agent = RLAgent(
        exp=exp,
        seed=seed,
        env_config={},
        device="cpu",
        collector_config=collector_config,
        render=False
    )

    logger.info(f"Training on {env_name} for {total_timesteps} timesteps...")

    score, last_obs = rl_agent.train()

    # Output final score
    logger.info(f"Training completed. Final Score: {score}")

    # Evaluate the agent
    logger.info("Evaluating the trained agent...")
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = rl_agent.agent.sample_action(rl_agent.agent.train_state.policy_params, obs, np.random.RandomState(), training=False)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    env.close()
    logger.info(f"Evaluation completed. Total Reward: {total_reward}")

if __name__ == "__main__":
    main()

