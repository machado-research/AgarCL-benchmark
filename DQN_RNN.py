import argparse
import json
import os

import numpy as np
import torch.nn as nn
import gymnasium as gym
import pfrl
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
import gym_agario

import torch
import wandb

from RNN_arch import MultiInputEmbeddingModule, GRURecurrent, MLPActorHead

from custom_utils import ModifyObservationWrapper, DiscreteActions

def main():
    print("Python Version: ", torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="agario-screen-v0",
        help="OpenAI Atari domain to perform algorithm on.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="PATH_TO_OUTPUT_DIR",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default= 10 * 10**6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=5,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--eval-n-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--n-best-episodes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--target_update_interval", type=int, default=10**4)
    parser.add_argument("--batch_accumulator", type=str, default="sum") #sum or mean
    parser.add_argument("--minibatch_size", type=int, default=1) 
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument('--ezGreedy', action='store_true', help='Use EZGreedy explorer')
    parser.add_argument('--cont', action='store_true', help='Use continuing training')
    parser.add_argument("--lr_decay", type=bool, default=False)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--load-replay-buffer", type=str, default="")
    parser.add_argument("--load-env", type=str, default="")
    parser.add_argument("--total-reward", type=int, default=0)
    parser.add_argument(
        "--episodic-update-len",
        type=int,
        default=10**3,
        help="Maximum length of sequences for updating recurrent models",
    )
    
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="DQN", config=args)
        wandb.config.update(args)

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2**31 - 1 - args.seed

    if (args.load is not None): 
        exp_id = args.load.split("/")[-2]
        args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id)
        #Here update both --load-env and --load-replay-buffer
        checkpoint_number = args.load.split("/")[-1].split("_")[0]
        load_env_checkpoint_name = f"checkpoint_{checkpoint_number}.json"
        args.load_env = os.path.join(args.load, load_env_checkpoint_name)
        args.load_replay_buffer = os.path.join(args.load, f"{checkpoint_number}_checkpoint.replay.pkl")
        print("Replay buffer loaded from: ", args.load_replay_buffer)
        print("Env state loaded from: ", args.load_env)
        args.step_offset = int(checkpoint_number)
        episodic_rewards_path = os.path.join(args.outdir, "episodic_rewards.csv")
        if os.path.exists(episodic_rewards_path):
            with open(episodic_rewards_path, "r") as f:
                last_line = f.readlines()[-1].strip()
                args.total_reward = float(last_line.split(",")[2])
        else:
            args.total_reward = 0.0
        print("Total reward so far: ", args.total_reward)
        print("Step offset: ", args.step_offset)
    else: 
        args.outdir = experiments.prepare_output_dir(args, args.outdir)

    # args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env_(test):
        # Use different random seeds for train and test envs
        
        env_seed = test_seed if test else train_seed
        env_name = "agario-screen-v0"

        env_config = json.load(open('env_config.json', 'r'))

        # env = make_env(env_name, env_config, gamma, norm_obs, norm_reward)
        env = gym.make(env_name, **env_config)
        env.seed(int(env_seed))
        if args.load_env != "": 
            env.load_env_state(args.load_env)
        env = DiscreteActions(env)
        env = ModifyObservationWrapper(env)
        
            
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, 0.5)
        
       
        return env

    env = make_env_(test=False)
    eval_env = make_env_(test=True)

    n_actions = env.action_space.n

    # q_func =  pfrl.nn.RecurrentSequential(
    #     DQNNetwork(
    #     obs_channels=4,  # Assuming RGB input
    #     act_dim=n_actions,
    #     obs_hidden_dim=128, 
    #     )
    # )
    obs_channels = 4
    q_func = pfrl.nn.RecurrentSequential(
    MultiInputEmbeddingModule(
        obs_channels=obs_channels,
        act_dim=n_actions,
        obs_hidden_dim_a=64,
        obs_hidden_dim_b=64,
        act_hidden_dim=16,
        reward_hidden_dim=8
    ),
    GRURecurrent(
        input_size=64 + 64 + 16 + 8,
        hidden_size=128
    ),
    MLPActorHead(
        hidden_size=128,
        act_dim=n_actions
    )
)


    opt = torch.optim.Adam(q_func.parameters(), args.lr , eps=1.5 * 10**-4)

    rbuf = replay_buffers.EpisodicReplayBuffer(100000)#1e5
    
    # if args.load_replay_buffer != "":
    #     rbuf.load(args.load_replay_buffer)
    #     print("Replay buffer loaded from: ", args.load_replay_buffer)
        

    if args.ezGreedy:
        explorer = explorers.EZGreedy(
            start_epsilon=1.0,
            end_epsilon=0.1,
            decay_steps=10**6,
            random_action_func=lambda: np.random.randint(n_actions),
        )
    else: 
        explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,
        end_epsilon=0.1,
        decay_steps=10**6,
        random_action_func=lambda: np.random.randint(n_actions),
        )
        
    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    # agent = Agent(
    #     q_func,
    #     opt,
    #     rbuf,
    #     gpu=args.gpu,
    #     gamma=0.99,
    #     explorer=explorer,
    #     replay_start_size=args.replay_start_size,
    #     target_update_interval=args.target_update_interval,
    #     clip_delta=True,
    #     update_interval=4,
    #     batch_accumulator=args.batch_accumulator,
    #     minibatch_size = args.minibatch_size,
    #     soft_update_tau = args.tau,
    #     n_times_update  = args.epochs,
    #     phi=phi,
    #     recurrent = True,
    #     episodic_update_len=args.episodic_update_len,
    # )
    agent = pfrl.agents.DQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=4,
        batch_accumulator="mean",
        phi=phi,
        minibatch_size=args.minibatch_size,
        episodic_update_len=args.episodic_update_len,
        recurrent=True,
    )
    step_hooks = []
    if args.lr_decay == True:
        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value
        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
        ) 

    step_hooks = []
    # Linearly decay the learning rate to zero
    def lr_setter(env, agent, value):
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = value
    step_hooks.append(
        experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
    ) 

    if args.load or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model("DQN", args.env, model_type=args.pretrained_type)[
                    0
                ]
            )

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=args.eval_n_steps, n_episodes=None
        )
        print(
            "n_episodes: {} mean: {} median: {} stdev {}".format(
                eval_stats["episodes"],
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_with_evaluation_RNN(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            eval_env=eval_env,
            checkpoint_freq = 5000000,
            step_hooks=step_hooks,
            case="continuing" if args.cont else "episodic",
            step_offset=args.step_offset,
            env_checkpointable=False,
            buffer_checkpointable=False,
            total_reward_so_far=args.total_reward,
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        # run 30 evaluation episodes, each capped at 5 mins of play
        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=4500,
            logger=None,
        )
        with open(os.path.join(args.outdir, "bestscores.json"), "w") as f:
            json.dump(stats, f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))


if __name__ == "__main__":
    main()