import gymnasium as gym
import gym_agario

import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd
import imageio

from typing import Callable, Any



def make_env(env_name, config, gamma, normalize_observation=False, normalize_reward=False, hybrid_action=False):

    env = gym.make(env_name, **config)
    # deal with dm_control's Dict observation space
    # env = FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if hybrid_action:
        env = HybridActionWrapper(env)
    else:
        env = ContinuousActionWrapper(env)
    # env = ObservationWrapper(env)

    if normalize_observation:
        # env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

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
        obs, _ = self.env.reset(**kwargs)
        return self.observation(obs), {}

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
        obs, _ = self.env.reset(**kwargs)
        return self.observation(obs), {}

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
        observation, _ = self.env.reset(**kwargs)
        if self.env.render_mode == "rgb_array":
            self.start_video_writer()
        return observation, {}

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


class SB3Wrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        return self.env.step(action)


class ModifyContinuousActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Redefining the action space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.low, self.high = self.action_space.low, self.action_space.high

    def action(self, action):
        # cont_action, dis_action = action[:-1], action[-1]
        # range_array = np.linspace(-1, 1, 4)
        # insert_index = np.searchsorted(range_array, dis_action)
        # dis_action = insert_index - 1
        # dis_action = np.maximum(dis_action, 0)
        return ([((action), 0)])

    def step(self, action):
        return self.env.step(self.action(action))


class ModifyDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, action):
        # Map the discrete action to (dx, dy)
        action_mappings = [
            np.array([0, 1], dtype=np.float32),   # Up
            np.array([1, 1], dtype=np.float32),   # Up-Right
            np.array([1, 0], dtype=np.float32),   # Right
            np.array([1, -1], dtype=np.float32),  # Down-Right
            np.array([0, -1], dtype=np.float32),  # Down
            np.array([-1, -1], dtype=np.float32),  # Down-Left
            np.array([-1, 0], dtype=np.float32),  # Left
            np.array([-1, 1], dtype=np.float32),  # Up-Left
        ]
        # Ensure action is within valid range
        assert 0 <= action < len(action_mappings)
        dx_dy = action_mappings[action]
        # Discrete action is always 0 (noop)
        return (dx_dy, 0)

    def step(self, action):
        return self.env.step(self.action(action))


class ModifyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Modify observation space if needed
        self.observation_space = env.observation_space

    def observation(self, observation):
        # Modify the observation here
        # Normalize the observation
        modified_observation = observation[0].transpose(2, 0, 1)
        # modified_observation = modified_observation/255.0        
        # Normalize the observation
        return modified_observation.astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs[0].transpose(2,0,1).astype(np.uint8)
        # obs = obs/255.0
        return obs, info


#PPO - CleanRL
class ContinuousActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def action(self, action):
        return (action,0)

class HybridActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Tuple((
            gym.spaces.Box(low=-1, high=1, shape=(2,)),  # (dx, dy) movement vector
            gym.spaces.Discrete(3),                      # 0=noop, 1=split, 2=feed
        ))

    def action(self, action):
        return action