from typing import Callable

import gymnasium as gym
import torch
import cv2
import os
import time

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
    env: gym.Env = None,
):
    envs = env 
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = [0]
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, reward, done, truncations, step_num = envs.step(actions[0].cpu().numpy())
        gray_scale = next_obs.mean(axis=1)[0] # Compute the mean across the color channels
        gray_scale = (gray_scale * 255).astype('uint8') # Convert to uint8
        
        os.makedirs("images", exist_ok=True)
        cv2.imwrite(f"images/observation_{step_num}.png", gray_scale)
        # print(f"Reward: {reward}")
        if done or truncations:
            episodic_returns.append(reward)
            obs, _ = envs.reset()
            break
        else:
            episodic_returns[-1] += reward
        
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.ppo_continuous_action import Agent, make_env

    model_path = hf_hub_download(
        repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "Hopper-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
    )