import gymnasium as gym
import numpy as np
import cv2

from stable_baselines3 import PPO as sb3_PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from PyExpUtils.collection.Collector import Collector

from src.wrappers.sb3 import TrainingCallback
from src.wrappers.gym import SB3Wrapper, ModifyContinuousActionWrapper
from src.utils.torch.networks import CNNPolicy


class CustomActorCriticPolicy(ActorCriticPolicy):
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


class PPO:
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

        # Hyperparameters
        self.num_steps = hypers['num_steps']
        self.anneal_lr = hypers['anneal_lr']
        self.learning_rate = hypers['learning_rate']
        self.gamma = hypers['gamma']
        self.gae_lambda = hypers['gae_lambda']
        self.num_minibatches = hypers['num_minibatches']
        self.update_epochs = hypers['update_epochs']
        self.norm_adv = hypers['norm_adv']
        self.clip_coef = hypers['clip_coef']
        self.clip_vloss = hypers['clip_vloss']
        self.ent_coef = hypers['ent_coef']
        self.vf_coef = hypers['vf_coef']
        self.max_grad_norm = hypers['max_grad_norm']
        self.target_kl = hypers.get('target_kl', None)

        self.env = SB3Wrapper(env)
        self.env = ModifyContinuousActionWrapper(self.env)

        self.net = sb3_PPO(CustomActorCriticPolicy,
                           self.env,
                           verbose=1,
                           seed=seed,
                           device=device,
                           learning_rate=self.learning_rate,
                           n_steps=self.num_steps,
                           batch_size=self.num_minibatches,
                           n_epochs=self.update_epochs,
                           gamma=self.gamma,
                           gae_lambda=self.gae_lambda,
                           clip_range=self.clip_coef,
                           ent_coef=self.ent_coef,
                           vf_coef=self.vf_coef,
                           max_grad_norm=self.max_grad_norm,
                           #    target_kl=self.target_kl,
                           )
        
    def train(self, time_steps: int = None):
        callback = TrainingCallback(self.collector)

        total_timesteps = time_steps if time_steps is not None else self.total_timesteps
        self.net.learn(total_timesteps=total_timesteps,
                       callback=callback)
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
