
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import collections
import heapq


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
    

class ModularGoBiggerMLP(nn.Module):
    def __init__(self, feature_dims, n_actions):
        """
        feature_dims: dict mapping each key in
          ['food','spore','thorn','clone','clone_mask','clone_history']
        to its flattened length.
        n_actions: number of discrete actions.
        """
        super().__init__()
        self.feature_dims = feature_dims

        # 1) Objects branch: (food+spore+thorn+clone) → 1024→1024
        total_objs = (
            feature_dims['food']
          + feature_dims['spore']
          + feature_dims['thorn']
          + feature_dims['clone']
        )
        
        self.objects_fc = nn.Sequential(
            nn.Linear(total_objs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        # 2) Clone history branch: clone_history → 512→512
        self.history_fc = nn.Sequential(
            nn.Linear(feature_dims['clone_history'], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # 3) Mask branch: clone_mask → 256→256
        self.mask_fc = nn.Sequential(
            nn.Linear(feature_dims['clone_mask'], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # final head: (1024 + 512 + 256) → 1024 → 512 → n_actions
        head_in = 1024 + 512 + 256
        self.head = nn.Sequential(
            nn.Linear(head_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, flat):
        """
        flat: Tensor of shape (batch, total_dim), where total_dim
        = sum(feature_dims.values()), and ordered exactly as the
        φ flattened them.
        """
        idx = 0

        def take(key):
            nonlocal idx
            L = self.feature_dims[key]
            out = flat[:, idx:idx + L]
            idx += L
            return out

        # Objects branch
        objs = torch.cat([
            take('food'),
            take('spore'),
            take('thorn'),
            take('clone'),
        ], dim=1)
        obj_emb = self.objects_fc(objs)

        # History branch
        hist = take('clone_history')
        hist_emb = self.history_fc(hist)

        # Mask branch
        mask = take('clone_mask')
        mask_emb = self.mask_fc(mask)

        # Concat & head
        x = torch.cat([obj_emb, hist_emb, mask_emb], dim=1)
        return self.head(x)

class TranslatorResetWrapper(gym.Wrapper):
    """
    Calls translator.reset() every time the env is reset, so
    your Translator.clone_history deque starts fresh each episode.
    """
    def __init__(self, env, translator):
        super().__init__(env)
        self.translator = translator

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.translator.reset()
        return obs, info

class Translator:
    def __init__(self, max_food, max_virus, max_spore, max_clone, history_length):
        self.max_food       = max_food
        self.max_virus      = max_virus
        self.max_spore      = max_spore
        self.max_clone      = max_clone
        self.history_length = history_length
        self.clone_history  = collections.deque(maxlen=history_length)

    def reset(self):
        self.clone_history.clear()

    def _nearest_and_pack(self, infos, type_id, max_count, cx, cy):
        # sorted_infos = sorted(
        #     infos,
        #     key=lambda f: (f.get_position_x() - cx)**2 + (f.get_position_y() - cy)**2
        # )
        nearest = heapq.nsmallest(
            max_count, infos,
            key=lambda f: (f.get_position_x() - cx)**2
                    + (f.get_position_y() - cy)**2
        )
        # sorted_infos = infos
        # nearest = sorted_infos[:max_count]
        out = []
        for f in nearest:
            x = float(f.get_position_x())
            y = float(f.get_position_y())
            out += [float(type_id), x, y, float(f.radius), float(f.score)]
        # pad if fewer
        pad = max_count - len(nearest)
        out += [0.0] * (pad * 5)
        return out

    def handle_obs(self, obs):
        ps    = obs["player_states"].get_all_player_states()
        state = list(ps.values())[0]

        # reference point: first clone or origin
        clones = list(state.get_clone_infos())
        # if clones:
        #     cx = float(clones[0].get_position_x())
        #     cy = float(clones[0].get_position_y())
        # else:
        cx = cy = 0.0

        feats = {}
        # nearest-N for each entity
        feats["food"]  = np.array(
            self._nearest_and_pack(state.get_food_infos(),  1, self.max_food,  cx, cy),
            dtype=np.float32,
        )
        feats["thorn"] = np.array(
            self._nearest_and_pack(state.get_virus_infos(), 2, self.max_virus, cx, cy),
            dtype=np.float32,
        )
        feats["spore"] = np.array(
            self._nearest_and_pack(state.get_spore_infos(),  3, self.max_spore,  cx, cy),
            dtype=np.float32,
        )
        feats["clone"] = np.array(
            self._nearest_and_pack(state.get_clone_infos(), 4, self.max_clone, cx, cy),
            dtype=np.float32,
        )

        # presence mask
        n = len(clones)
        mask = [1.0]*min(n, self.max_clone) + [0.0]*max(0, self.max_clone - n)
        feats["clone_mask"] = np.array(mask, dtype=np.float32)

        # --- update & pad/truncate history ---
        snapshot = []
        for f in clones[: self.max_clone]:
            snapshot += [
                float(f.get_position_x()),
                float(f.get_position_y()),
                float(f.radius),
            ]
        snapshot += [0.0]*((self.max_clone - len(clones)) * 3)
        self.clone_history.append(snapshot)

        hist_list = list(self.clone_history)
        # clamp to history_length
        if len(hist_list) > self.history_length:
            hist_list = hist_list[-self.history_length:]
        # pad front if too short
        pad_frames = self.history_length - len(hist_list)
        if pad_frames > 0:
            zero_frame = [0.0] * (self.max_clone * 3)
            hist_list = [zero_frame] * pad_frames + hist_list

        feats["clone_history"] = np.array(hist_list, dtype=np.float32).flatten()

        return feats

#class Translator:
    #"""
    #Minimal feature extractor for GoBigger:
      #- nearest lists for food, spore, thorn, clone
      #- clone-centric history of positions and radii
      #- clone presence mask
    #"""
    #def __init__(self, max_food, max_spore, max_thorn, max_clone, history_length):
        #self.max_food = max_food
        #self.max_spore = max_spore
        #self.max_thorn = max_thorn
        #self.max_clone = max_clone
        #self.history_length = history_length
        ## store last history_length snapshots of clone (x,y,r)
        #self.clone_history = collections.deque(maxlen=history_length)

    #def reset(self):
        ## Clear clone history at episode start
        #self.clone_history.clear()

    #def _pack_list(self, infos, type_id, max_count):
        #out = []
        ## pack up to max_count items
        #for info in infos[:max_count]:
            #x = float(info.get_position_x())
            #y = float(info.get_position_y())
            #out += [float(type_id), x, y, float(info.radius), float(info.score)]
            ## pad if fewer
            #for _ in range(max_count - len(infos)):
                #out += [0.0, 0.0, 0.0, 0.0, 0.0]
        #return out


    #def handle_obs(self, obs):
        #ps = obs['player_states'].get_all_player_states()
        #state = list(ps.values())[0]

        ## 1) find your reference clone position
        #clones = state.get_clone_infos()
        ## if clones:
        ##     cx = float(clones[0].get_position_x())
        ##     cy = float(clones[0].get_position_y())
        ## else:
        ## All the positions are relative to the agent
        #cx = cy = 0.0

        #feats = {}

        ## 2) sort the food list by distance to (cx,cy)
        #food_infos = list(state.get_food_infos())
        #food_infos.sort(key=lambda info: 
            #(info.get_position_x() - cx)**2 + (info.get_position_y() - cy)**2
        #)
        ## 3) now pack only the nearest self.max_food
        #feats['food'] = np.array(
            #self._pack_list(food_infos, type_id=1, max_count=self.max_food),
            #dtype=np.float32
        #)

        #spore_infos = list(state.get_spore_infos())
        #spore_infos.sort(key=lambda info: 
            #(info.get_position_x() - cx)**2 + (info.get_position_y() - cy)**2
        #)
        #feats['spore'] = np.array(
            #self._pack_list(spore_infos, type_id=3, max_count=self.max_spore),
            #dtype=np.float32
        #)

        #thorn_infos = list(state.get_virus_infos())
        #thorn_infos.sort(key=lambda info: 
            #(info.get_position_x() - cx)**2 + (info.get_position_y() - cy)**2
        #)
        #feats['thorn'] = np.array(
            #self._pack_list(thorn_infos, type_id=2, max_count=self.max_thorn),
            #dtype=np.float32
        #)

        #clone_infos = list(state.get_clone_infos())
        #clone_infos.sort(key=lambda info: 
            #(info.get_position_x() - cx)**2 + (info.get_position_y() - cy)**2
        #)
        #feats['clone'] = np.array(
            #self._pack_list(clone_infos, type_id=4, max_count=self.max_clone),
            #dtype=np.float32
        #)

        ## clone presence mask
        #n = len(state.get_clone_infos())
        #mask = [1.0]*min(n, self.max_clone) + [0.0]*max(0, self.max_clone - n)
        #feats['clone_mask'] = np.array(mask, dtype=np.float32)


        #snapshot = []
        #for info in state.get_clone_infos()[: self.max_clone]:
            #snapshot += [
                #float(info.get_position_x()),
                #float(info.get_position_y()),
                #float(info.radius),
            #]
        ## pad this snapshot up to max_clone
        #snapshot += [0.0] * ( (self.max_clone - len(state.get_clone_infos())) * 3 )
        #self.clone_history.append(snapshot)

        ## --- now build the fixed-length history feature ---
        ## copy into a list
        #hist_list = list(self.clone_history)  
        ## if not enough frames yet, pad at front
        #pad_frames = self.history_length - len(hist_list)
        #if pad_frames > 0:
            #zero_frame = [0.0] * (self.max_clone * 3)
            #hist_list = [zero_frame] * pad_frames + hist_list
        ## hist_list is now exactly history_length frames long
        #feats['clone_history'] = np.array(hist_list, dtype=np.float32).flatten()

        ## # update clone history snapshot
        ## snapshot = []
        ## for info in state.get_clone_infos()[:self.max_clone]:
        ##     x = float(info.get_position_x())
        ##     y = float(info.get_position_y())
        ##     r = float(info.radius)
        ##     snapshot += [x, y, r]
    
        ## # pad snapshot
        ## for _ in range(self.max_clone - len(state.get_clone_infos())):
        ##     snapshot += [0.0, 0.0, 0.0]

        ## self.clone_history.append(snapshot)

        ## # assemble clone_history feature (history_length × max_clone × 3)
        ## history = list(self.clone_history)
        ## # pad front if too short
        ## while len(history) < self.history_length:
        ##     history.insert(0, [0.0]*(self.max_clone*3))
        ## feats['clone_history'] = np.array(history, dtype=np.float32).flatten()

        #return feats
