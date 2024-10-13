import gymnasium as gym
import gym_agario

import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd
import imageio

from typing import Callable, Any


def make_env(env_name, config, gamma, normalize_observation=False, normalize_reward=False):
    env = gym.make(env_name, **config)
    # deal with dm_control's Dict observation space
    if normalize_observation:
        env = NormalizeObservation(env)

    if normalize_reward:
        env = NormalizeReward(env, gamma=gamma)

    if config['render_mode'] == "rgb_array":
        env = VideoRecorderWrapper(env, config['video_path'])
    return env


class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(
                shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncated, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncated, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs)
        else:
            return self.normalize(np.array([obs]))[0]

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncated, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        self.returns[terminateds] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncated, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)


class TransformObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, f: Callable[[Any], Any]):
        super().__init__(env)
        assert callable(f)
        self.f = f

    def observation(self, observation):
        return self.f(observation)

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        return self.observation(observation), reward, terminated, truncated, info


class TransformReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env, f: Callable[[float], float]):
        super().__init__(env)
        assert callable(f)
        self.f = f

    def reward(self, reward):
        return self.f(reward)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        return observation, self.reward(reward), terminated, truncated, info


class ClipAction:

    def __init__(self, action_limits, env):
        self.low, self.high = action_limits
        self.env = env

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return np.clip(action, self.low, self.high)

    def step(self, action):
        """Runs the environment :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        return self.env.step(self.action(action))


class FlattenObservation(gym.ObservationWrapper):
    """Observation wrapper that flattens the observation.

    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = gym.spaces.flatten_space(
            env.observation_space)

    def observation(self, observation):
        """Flattens an observation.

        Args:
            observation: The observation to flatten

        Returns:
            The flattened observation
        """
        return gym.spaces.flatten(self.env.observation_space, observation)

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        return self.observation(observation), reward, terminated, truncated, info


class VideoRecorderWrapper(gym.Wrapper):
    """Wrapper that records a video of an episode, if render_mode is rgb_array"""

    def __init__(self, env, video_path):
        super().__init__(env)
        self.video_path = video_path
        self.frames = []
        self.video_writer = None

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if self.env.render_mode == "rgb_array":
            self.start_video_writer()
        return observation

    def start_video_writer(self):
        if self.video_writer is None:
            self.video_writer = imageio.get_writer(self.video_path, fps=50)

    def record_frame(self):
        frame = self.env.render()
        for i in range(self.env.unwrapped.num_frames):
            if isinstance(frame, np.ndarray):
                self.frames.append(frame[i])

    def step(self, action):
        observation, reward, termination, truncation, info = self.env.step(
            action)
        if self.env.render_mode == "rgb_array":
            self.record_frame()
        return observation, reward, termination, truncation, info

    def close_video_writer(self):
        if self.video_writer is not None:
            for frame in self.frames:
                self.video_writer.append_data(frame)
            self.video_writer.close()
            self.video_writer = None
            self.frames = []

    def close(self):
        self.close_video_writer()
        self.env.close()
