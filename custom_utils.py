import gymnasium as gym
import numpy as np



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


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(24)
        self.action_mappings = [
            (np.array([0, 1], dtype=np.float32), i) for i in range(3)   # Up
        ] + [
            (np.array([1, 1], dtype=np.float32), i) for i in range(3)   # Up-Right
        ] + [
            (np.array([1, 0], dtype=np.float32), i) for i in range(3)   # Right
        ] + [
            (np.array([1, -1], dtype=np.float32), i) for i in range(3)  # Down-Right
        ] + [
            (np.array([0, -1], dtype=np.float32), i) for i in range(3)  # Down
        ] + [
            (np.array([-1, -1], dtype=np.float32), i) for i in range(3) # Down-Left
        ] + [
            (np.array([-1, 0], dtype=np.float32), i) for i in range(3)  # Left
        ] + [
            (np.array([-1, 1], dtype=np.float32), i) for i in range(3)  # Up-Left
        ]
        self.action_space = gym.spaces.Discrete(len(self.action_mappings))

    def action(self, action):
        # Map the discrete action to (dx, dy)
        
        # Ensure action is within valid range
        assert 0 <= action < len(self.action_mappings)
        mapped_action = self.action_mappings[action]
        # Discrete action is always 0 (noop)
        return mapped_action