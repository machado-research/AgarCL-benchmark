import gymnasium as gym
import numpy as np
import cv2

from stable_baselines3 import DQN as sb3_DQN
from stable_baselines3.dqn.policies import DQNPolicy
from PyExpUtils.collection.Collector import Collector

from src.wrappers.sb3 import TrainingCallback
from src.wrappers.gym import SB3Wrapper, ModifyDiscreteActionWrapper
from src.utils.torch.networks import CNNPolicy


class CustomActorCriticPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        *args,
        **kwargs,
    ):
        # Pass custom feature extractor to the constructor
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[64],
            features_extractor_class=CNNPolicy,
            features_extractor_kwargs={"features_dim": 128},
            *args,
            **kwargs,
        )


class DQN:
    def __init__(self,
                 env: gym.Env,
                 seed: int,
                 device: str,
                 hypers: dict,
                 collector: Collector = None,
                 total_timesteps: int = 1e6,
                 eval_steps: int = 3000,
                 ) -> None:

        self.collector = collector
        self.total_timesteps = total_timesteps
        self.eval_steps = eval_steps

        self.lr = hypers['lr']
        self.buffer_size = int(hypers['buffer_size'])
        self.learning_starts = int(hypers['learning_starts'])
        self.batch_size = hypers['batch_size']
        self.hidden_size = hypers['hidden_size']
        self.gamma = hypers['gamma']
        self.tau = hypers['tau']

        self.env = SB3Wrapper(env)
        self.env = ModifyDiscreteActionWrapper(self.env)

        self.net = sb3_DQN(CustomActorCriticPolicy,
                           self.env,
                           verbose=0,
                           learning_rate=self.lr,
                           buffer_size=self.buffer_size,
                           learning_starts=self.learning_starts,
                           batch_size=self.batch_size,
                           gamma=self.gamma,
                           tau=self.tau,
                           device=device,
                           seed=seed,
                           )

    def train(self, time_steps: int = None):
        callback = TrainingCallback(self.collector)

        total_timesteps = time_steps if time_steps is not None else self.total_timesteps
        self.net.learn(total_timesteps=total_timesteps,
                       callback=callback, reset_num_timesteps=False)
        return callback.get_avg_reward(), callback.get_last_obs()

    def eval(self, obs: np.ndarray, save_path: str):
        cumulative_reward = 0
        if obs is None:
            obs, _ = self.env.reset()

        width, height = obs.shape[1], obs.shape[2]
        # or use 'avc1' for H.264 encoding
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(
            save_path,
            fourcc,
            30,  # framerate - increased to 30fps for smoother video
            (width, height)
        )

        for _ in range(self.eval_steps):
            action, _ = self.net.predict(obs)
            image = obs[0]  # Convert to uint8
            image = image.astype('uint8')  # Convert to uint8
            video.write(image)

            obs, reward, done, trunc, _ = self.env.step(action)
            cumulative_reward += reward
            
            # collect reward
            self.collector.next_frame()
            self.collector.collect('eval_reward', reward)

        cv2.destroyAllWindows()
        video.release()

        return cumulative_reward  # need to collect episodic reward, not reward per step

    def save_checkpoint(self, path: str):
        self.net.save(path)

    def load_checkpoint(self, path: str):
        self.net.load(path)
