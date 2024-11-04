import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

import time
import gymnasium as gym
from typing import NamedTuple, Any
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity
from PyExpUtils.results.sqlite import saveCollector

from src.utils.jax.ppo_nets import PPONetwork
from src.utils.jax.mis import modify_action
from src.measurements.jax_norms import get_statistics


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@struct.dataclass
class LogEnvState:
    returned_returns: float
    timestep: int


class EnvState(struct.PyTreeNode):
    env: Any = struct.field(pytree_node=False)
    env_state: Any = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, env, env_state, **kwargs):
        """Creates a new instance"""
        return cls(
            env=env,
            env_state=env_state,
            **kwargs,
        )


class CustomTrainState(TrainState):
    grads: Any = struct.field(pytree_node=True)


class PPO:
    def __init__(self,
                 env: gym.Env,
                 seed: int,
                 hypers: dict,
                 collector_config: dict,
                 total_timesteps: int = 1e6,
                 render: bool = False,
                 device: str = 'cpu',
                 ) -> None:

        self.env = env
        self.seed = seed
        self.render = render
        self.total_timesteps = total_timesteps

        self.collector = Collector(collector_config)
        self.collector.setIdx(seed)

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
        self.hidden_size = hypers['hidden_size']
        self.target_kl = hypers.get('target_kl', None)

        self.minibatch_size = int(self.num_steps // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.num_steps

        self.env.action_space = (gym.spaces.Box(-1, 1, self.env.action_space[0].shape, dtype=jnp.float32),
                                 gym.spaces.Discrete(3))

        self.action_shape = (self.env.action_space[0].shape[0] + 1,)
        self.obs_shape = self.env.observation_space.shape
        self.max_action = float(self.env.action_space[0].high[0])
        self.min_action = float(self.env.action_space[0].low[0])

        self.train_state = None
        self.rng = None

    def train(self):
        self.trial_return = 0.0
        self.episodes = 0
        self.steps = 0

        # ENV INIT
        log_env_state = LogEnvState(returned_returns=0, timestep=0)
        env_state = EnvState.create(
            env=self.env, env_state=log_env_state)
        init_obs = self.env.reset()

        # TRAIN STATE INIT AND NETWORK AND OPTIMIZER
        rng = jax.random.PRNGKey(self.seed)
        rng, _rng = jax.random.split(rng)

        network = PPONetwork(self.action_shape[0], self.hidden_size)
        params = network.init(_rng, jnp.zeros(self.obs_shape))
        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.learning_rate),
        )

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
            grads=params,
        )

        def update(env_step_state, num_updates, rollout_steps, gae_lambda,
                   gamma, clip_eps, vf_coef, ent_coef, batch_size, epochs):
            @jax.jit
            def calculate_gae(traj_batch, last_val, gamma, gae_lambda):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + gamma * gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            @jax.jit
            def agent_step(train_state, params, obs, rng):
                pi, value = train_state.apply_fn(params, obs)
                action = pi.sample(seed=rng).squeeze()
                log_prob = pi.log_prob(action)
                return action, log_prob, value

            def env_step(runner_state):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION (using agent rng)
                rng, _rng = jax.random.split(rng)
                action, log_prob, value = agent_step(
                    train_state, train_state.params, last_obs, _rng)
                step_action = modify_action(
                    action, self.min_action, self.max_action)

                # STEP ENV (using env rng)
                obs, reward, done, _, info = self.env.step(step_action)
                step = env_state.env_state.timestep + 1
                log_state = LogEnvState(returned_returns=reward,
                                        timestep=step)

                info["reward"] = reward
                info["timestep"] = env_state.env_state.timestep
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs.squeeze(), info)

                # Update runner state
                env_state = EnvState.create(
                    env=env_state.env, env_state=log_state)
                runner_state = (train_state, env_state, obs, rng)
                return runner_state, transition

            @jax.jit
            def update_minbatch(train_state, batch_info):
                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = train_state.apply_fn(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)
                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-clip_eps, clip_eps)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(
                        value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses,
                                          value_losses_clipped).mean()
                    )
                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - clip_eps,
                            1.0 + clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()
                    total_loss = (
                        loss_actor
                        + vf_coef * value_loss
                        - ent_coef * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                traj_batch, advantages, targets = batch_info
                grad_fn = jax.value_and_grad(
                    _loss_fn, has_aux=True, allow_int=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grads=grads)
                return train_state, total_loss

            @jax.jit
            def create_minibaches(batch, rng):
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, rollout_steps)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch)

                minibatch_size = rollout_steps // batch_size
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size, minibatch_size) + x.shape[1:]), shuffled_batch)

                return batch, rng

            @jax.jit
            def update_epoch(update_state, unused):
                train_state, traj_batch, advantages, targets, rng = update_state
                batch = (traj_batch, advantages, targets)
                # Prepare minibatches
                # minibatches: (num_minibatches, seq_len,minibatch_size, _)
                minibatches_info, rng = create_minibaches(batch, rng)
                # Loop through minibatches
                train_state, _ = jax.lax.scan(
                    update_minbatch, train_state, minibatches_info)
                update_state = (train_state, traj_batch,
                                advantages, targets, rng)
                return update_state, {}

            @jax.jit
            def _update_step(update_state):
                # Learning
                update_state, _ = jax.lax.scan(
                    update_epoch, update_state, None, epochs)

                # Update runner state
                train_state = update_state[0]
                rng = update_state[-1]

                return train_state, rng

            start_time = time.time()

            for _ in range(num_updates):
                traj_batch = []
                # Collect trajectory
                for _ in range(rollout_steps):
                    env_step_state, transition = env_step(env_step_state)
                    traj_batch.append(transition)

                traj_batch = jax.tree_util.tree_map(
                    lambda *v: jnp.stack(v), *traj_batch)  # stacking pytrees
                train_state, env_state, last_obs, rng = env_step_state

                _, last_value = train_state.apply_fn(
                    train_state.params, jnp.expand_dims(last_obs, 0))
                last_val = last_value.squeeze()

                # Calculate GAE
                advantages, targets = calculate_gae(
                    traj_batch, last_val, gamma, gae_lambda)

                # Update step
                rng, update_rng = jax.random.split(rng)
                update_state = (train_state, traj_batch,
                                advantages, targets, update_rng)
                train_state, rng = _update_step(update_state)

                # Update env_step_state
                env_step_state = (train_state, env_state, last_obs, rng)

                # log metrics
                info = traj_batch.info
                rewards = jnp.mean(info["reward"])
                timesteps = len(info["reward"])

                self.trial_return += rewards
                self.episodes += 1
                self.steps += timesteps
                self.train_state = train_state
                self.rng = rng

                # stats = get_statistics(
                #     network, train_state.params, traj_batch.obs, train_state.grads)
                print(
                    f"Step: {self.steps}, Reward: {rewards}, Time: {time.time() - start_time}, \
                        FPS: {timesteps / (time.time() - start_time)}")
                start_time = time.time()

                self.collector.next_frame()
                self.collector.collect('reward', rewards)
                self.collector.collect('steps', timesteps)
                # self.collect_stats(stats)

            score = self.trial_return / self.episodes
            return score

        runner_state = (train_state, env_state, init_obs, rng)
        score = update(runner_state, self.num_iterations, self.num_steps,
                       self.gae_lambda, self.gamma, self.clip_coef, self.vf_coef,
                       self.ent_coef, self.minibatch_size, self.update_epochs)

        return score

    def collect_stats(self, stats):
        self.collector.collect('spectral_norm', stats['spectral_norm'])
        self.collector.collect('spectral_norm_grad',
                               stats['spectral_norm_grad'])
        self.collector.collect('l2_norm', stats['l2_norm'])
        self.collector.collect('activation_norm', stats['activation_norm'])
        self.collector.collect('hidden_stable_rank',
                               stats['hidden_stable_rank'])
        self.collector.collect('stable_weight_rank',
                               stats['stable_weight_rank'])
        self.collector.collect('dormant_units', stats['dormant_units'])

    def save_collector(self, exp, save_path):
        self.collector.reset()
        saveCollector(exp, self.collector, base=save_path)

    def load_checkpoint(self, path):
        checkpoint = jnp.load(path)
        self.train_state = checkpoint['train_state']
        self.steps = checkpoint['steps']
        self.rng = checkpoint['rng']

    def save_checkpoint(self, path):
        checkpoint = {
            'train_state': self.train_state,
            'steps': self.steps,
            'rng': self.rng,
        }
        jnp.save(path, checkpoint)
