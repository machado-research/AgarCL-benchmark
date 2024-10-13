import os
import torch
import pickle

def save_env_state(env, env_state_path):
    with open(env_state_path, 'wb') as f:
        pickle.dump(env, f)
    print("Environment state saved")

def load_env_state(env_state_path):
    with open(env_state_path, 'rb') as f:
        env = pickle.load(f)
    print("Environment state loaded")
    return env

def save_checkpoint_env(state_dict, checkpoint_path, env=None):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state_dict, checkpoint_path)
    
    # save_env_state(env, checkpoint_path.replace('.pt', '_env.pkl'))
    print(f"Checkpoint saved!")
    
def load_checkpoint_env(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    # env = load_env_state(checkpoint_path.replace('.pt', '_env.pkl'))
    
    print(f"Checkpoint loaded!")
    return state_dict #, env

def preprocess_image_observation(x):
    # Remove any extra dimensions of size 1
    x = x.squeeze()
    
    # Ensure we have 4 dimensions
    if x.dim() == 3:
        x = x.unsqueeze(0)  # Add batch dimension if it's missing
        
    elif x.dim() != 4:
        raise ValueError(f"Expected 3 or 4 dimensions after squeezing, but got shape {x.shape}")
    
    # Rearrange dimensions to [batch, channels, height, width]
    if x.shape[-1] == 3:  # If the last dimension is 3, assume it's RGB channels
        x = x.permute(0, 3, 1, 2)
    
    return x