import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pfrl
from pfrl import agents, experiments, explorers, replay_buffers, utils
from pfrl.q_functions import DiscreteActionValueHead
from goBigUtils import GoBiggerObsFlatten, CustomMLP, Translator, ModularGoBiggerMLP, TranslatorResetWrapper
import gym_agario
import wandb

class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 8 directions × 3 codes = 24 discrete actions
        dirs = [
            np.array([0, 1], dtype=np.float32),   # up
            np.array([1, 1], dtype=np.float32),   # up-right
            np.array([1, 0], dtype=np.float32),   # right
            np.array([1, -1], dtype=np.float32),  # down-right
            np.array([0, -1], dtype=np.float32),  # down
            np.array([-1, -1], dtype=np.float32), # down-left
            np.array([-1, 0], dtype=np.float32),  # left
            np.array([-1, 1], dtype=np.float32),  # up-left
        ]
        self.action_mappings = [(d, code) for d in dirs for code in range(3)]
        self.action_space = gym.spaces.Discrete(len(self.action_mappings))

    def action(self, action):
        assert 0 <= action < len(self.action_mappings)
        return self.action_mappings[action]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="agario-screen-v0")
    parser.add_argument("--outdir", type=str, default="./Results/DQN_gobigger")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--steps", type=int, default=5 * 10**6)
    parser.add_argument("--replay-start-size", type=int, default=10**4)
    parser.add_argument("--eval-n-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=50000)
    parser.add_argument("--n-best-episodes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--target-update-interval", type=int, default=10**4)
    parser.add_argument("--batch-accumulator", type=str, default="sum")
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_decay", type=bool, default=False)
    parser.add_argument("--no-hidden-layers", type=int, default=2)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )

    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="DQN_gobigger2", config=args)

    utils.set_random_seed(args.seed)
    train_seed = args.seed
    test_seed = 2**31 - 1 - args.seed
    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    # load config for GoBigger
    env_config = json.load(open('env_config.json', 'r'))

    translator = Translator(
        max_food=128,
        max_virus=128,
        max_spore=64,
        max_clone=64,
        history_length=20,
    )

    def make_env_(test):
        # Use different random seeds for train and test envs
        
        env_seed = test_seed if test else train_seed
        env_name = "agario-screen-v0"

        # env = make_env(env_name, env_config, gamma, norm_obs, norm_reward)
        env = gym.make(env_name, **env_config)
        ######################################################
        ###################################################################
        env = TranslatorResetWrapper(env, translator)
        env = DiscreteActions(env)


        # env = GoBiggerObsFlatten(env,
        #                     max_food=env_config['num_pellets'],
        #                     max_virus=env_config['num_viruses'],
        #                     max_spore=env_config.get('num_spores', 0),
        #                     max_clone=env_config.get('num_bots', 0))
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, 0.5)
       
        return env

    # instantiate envs and get dims
    env = make_env_(test=False)
    eval_env = make_env_(test=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    raw, _ = env.reset()
    # env.translator.reset()

    feats = translator.handle_obs(raw)
    keep = ['food','spore','thorn','clone','clone_mask','clone_history']
    feature_dims = {k: feats[k].shape[0] for k in keep}

    # build Q-function with CustomMLP
    # if args.no_hidden_layers == 2:
    #     hidden_sizes = [256, 128]
    # elif args.no_hidden_layers > 2:
    #     hidden_sizes = [256] + [128] * (args.no_hidden_layers - 1)
    # else:
    #     raise ValueError("Number of hidden layers must be at least 2")

    # emb_dim = 64
    # q_body = CustomMLP(obs_dim, hidden_sizes, emb_dim)
 
    q_body = ModularGoBiggerMLP(
        feature_dims=feature_dims,
        n_actions=n_actions, )
    

    q_func = nn.Sequential(
        q_body,
        # nn.ReLU(),
        # nn.Linear(emb_dim, n_actions),
        DiscreteActionValueHead(),
    )

    opt = torch.optim.Adam(q_func.parameters(), lr=args.lr, eps=1.5e-4)
    rbuf = replay_buffers.ReplayBuffer(100000)
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,
        end_epsilon=0.1,
        decay_steps=10**6,
        random_action_func=lambda: np.random.randint(n_actions),
    )

    # identity phi (already float32)
    EXPECTED_DIM = sum(feature_dims[k] for k in keep)

    def phi(obs):
        feats = translator.handle_obs(obs)                 # dict of np.ndarray
        flat  = np.concatenate([feats[k] for k in keep], axis=-1)
        # for k in keep:
        #     print(k, feats[k].shape[0])
        assert flat.shape[0] == EXPECTED_DIM, (
        f"phi() produced length {flat.shape[0]}, but expected {EXPECTED_DIM}"
        )
        return flat.astype(np.float32)     

    agent = agents.DQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        clip_delta=True,
        update_interval=4,
        batch_accumulator=args.batch_accumulator,
        minibatch_size=args.minibatch_size,
        soft_update_tau=args.tau,
        n_times_update=args.epochs,
        phi=phi,
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
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            eval_env=eval_env,
            checkpoint_freq = 1000000,
            step_hooks=step_hooks,
            wandb_logging=True if args.wandb else False,
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
