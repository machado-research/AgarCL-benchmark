
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
class GoBiggerActionWrapper(gym.ActionWrapper):
    """
    Flatten the 2-part action tuple into the form the env expects:
      - model sees Box(low=[-1,-1,0], high=[1,1,2]) 
      - wrapper splits it back into (dx,dy), discrete
    """
    def __init__(self, env):
        super().__init__(env)
        # continuous dx,dy plus an integer code in [0,2]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, +1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        # action: 1D array [dx, dy, code]
        dx, dy, code = action
        # round the discrete part
        discrete = int(np.clip(np.round(code), 0, 2))
        return (np.array([dx, dy], dtype=np.float32), discrete)


class GoBiggerObsFlatten(gym.ObservationWrapper):
    """
    Flatten GoBigger obs into a fixed‐size 1D float32 vector:
      [ (1,x,y,r,score)*max_food,
                   (2,x,y,r,score)*max_virus,
                   (3,x,y,r,score)*max_spore,
                   (4,x,y,r,score)*max_clone ]
    """
    def __init__(self, env, max_food, max_virus, max_spore, max_clone):
        super().__init__(env)
        self.max_food   = max_food
        self.max_virus  = max_virus
        self.max_spore  = max_spore
        self.max_clone  = max_clone

        # do one reset to infer final flat size
        obs, info = env.reset()
        flat0 = self.observation(obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=flat0.shape, dtype=np.float32
        )

    def _pack_list(self, infos, type_id, max_count):
        """Turn a list of Info objects into a (max_count × 5) block."""
        out = []
        # actual items
        for info in infos[:max_count]:
            x = info.get_position_x()
            y = info.get_position_y()

            out += [float(type_id), x, y, info.radius, info.score]
       
        # pad if fewer
        for _ in range(len(infos), max_count):
            out += [0.0, 0.0, 0.0, 0.0, 0.0]
        return out

    def observation(self, obs):
        gs = obs["global_state"]
        ps = obs["player_states"].get_all_player_states()

        print( ps )
        # if single-agent, grab the only player
        state = list(ps.values())[0]

        flat = []
    
        # 2) food  (type 1)
        flat += self._pack_list(state.get_food_infos(),  1, self.max_food)
        # 3) virus (type 2)
        flat += self._pack_list(state.get_virus_infos(), 2, self.max_virus)
        # 4) spore (type 3)
        flat += self._pack_list(state.get_spore_infos(), 3, self.max_spore)
        # 5) clone (type 4)
        flat += self._pack_list(state.get_clone_infos(), 4, self.max_clone)

        return np.array(flat, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    


class CustomMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, emb_dim, activation=nn.ReLU()):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), activation]
            in_dim = h
        layers.append(nn.Linear(in_dim, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, obs_dim)
        return self.net(x)