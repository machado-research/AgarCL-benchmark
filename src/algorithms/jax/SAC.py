from typing import Dict, Tuple
import jax
import chex
import optax
import jax.numpy as jnp
import gymnasium as gym
from functools import partial
import flashbax as fb

from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Identity
from PyExpUtils.results.sqlite import saveCollector

from src.utils.jax.sac_nets import SACCriticNetwork, SACActorNetwork
from src.utils.jax.mis import modify_action
from src.measurements.jax_norms import get_statistics


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array


@chex.dataclass
class AgentState:
    critic_params: chex.Array
    target_params: chex.Array
    policy_params: chex.Array
    log_alpha: chex.Array
    critic_optim: optax.OptState
    policy_optim: optax.OptState
    alpha_optim: optax.OptState
    policy_grads: chex.Array
    critic_grads: chex.Array
    timesteps: int = 0


class SAC:
    def __init__(self, env: gym.Env, seed: int, hypers: dict, collector_config: dict,
                 total_timesteps: int = 1e6, render: bool = False, device: str = 'cpu') -> None:
        self.env = env
        self.seed = seed
        self.render = render
        self.total_timesteps = total_timesteps

        self.collector = Collector(collector_config)
        self.collector.setIdx(seed)

        self.q_lr = hypers['q_lr']
        self.policy_lr = hypers['policy_lr']
        self.buffer_size = int(hypers['buffer_size'])
        self.hidden_size = hypers['hidden_size']
        self.learning_starts = int(hypers['learning_starts'])
        self.batch_size = hypers['batch_size']
        self.update_frequency = hypers['policy_frequency']
        self.gamma = hypers['gamma']
        self.tau = hypers['tau']

        # Define networks
        self.env.action_space = (gym.spaces.Box(-1, 1, self.env.action_space[0].shape, dtype=jnp.float32),
                                 gym.spaces.Discrete(3))

        self.max_action = float(self.env.action_space[0].high[0])
        self.min_action = float(self.env.action_space[0].low[0])
        self.action_shape = (self.env.action_space[0].shape[0] + 1,)
        self.obs_shape = self.env.observation_space.shape

        self.target_entropy = hypers.get(
            'target_entropy', -self.action_shape[0])

        self.train_state = None
        self.buffer_state = None
        self.rng = jax.random.PRNGKey(seed)

        # Initialize critics and actor

        self.critic = SACCriticNetwork(self.hidden_size)
        self.actor = SACActorNetwork(self.action_shape[0], self.hidden_size)

        # Initialize optimizers
        self.actor_optimizer = optax.adam(self.policy_lr)
        self.critic_optimizer = optax.adam(self.q_lr)
        self.alpha_optimizer = optax.adam(self.policy_lr)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_critic_loss(self, critic_params: chex.Array, batch: TimeStep, target_q: chex.Array) -> Tuple[chex.Array, Dict]:
        current_q1, current_q2 = self.critic.get_action_values(
            critic_params, batch.experience.obs, batch.experience.action)
        critic_loss = jnp.mean((current_q1 - target_q) **
                               2 + (current_q2 - target_q)**2)
        return critic_loss, {'critic_loss': critic_loss}

    @partial(jax.jit, static_argnums=(0,))
    def _compute_actor_loss(self, policy_params: chex.Array, critic_params: chex.Array,
                            alpha: chex.Array, batch: TimeStep, rng: chex.PRNGKey) -> Tuple[chex.Array, Dict]:
        actions, log_probs = self.actor.sample(
            policy_params, batch.experience.obs, rng)
        q1, q2 = self.critic.get_action_values(
            critic_params, batch.experience.obs, actions)
        min_q = jnp.minimum(q1, q2)
        actor_loss = jnp.mean(alpha * log_probs - min_q)
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -jnp.mean(log_probs)}

    @partial(jax.jit, static_argnums=(0,))
    def _compute_alpha_loss(self, log_alpha: chex.Array, entropy: chex.Array) -> Tuple[chex.Array, Dict]:
        alpha = jnp.exp(log_alpha)
        alpha_loss = jnp.mean(alpha * (-entropy - self.target_entropy))
        return alpha_loss, {'alpha_loss': alpha_loss, 'alpha': alpha}

    @partial(jax.jit, static_argnums=(0,))
    def _update_step(self, train_state: AgentState, batch: TimeStep, rng: chex.PRNGKey):
        # Critic update
        next_actions, next_log_probs = self.actor.sample(
            train_state.policy_params, batch.experience.next_obs, rng)
        next_q1, next_q2 = self.critic.get_action_values(
            train_state.target_params, batch.experience.next_obs, next_actions)

        # Compute targets for Q-function
        next_q = jnp.minimum(next_q1, next_q2)
        alpha = jnp.exp(train_state.log_alpha)
        next_q = next_q - alpha * next_log_probs
        target_q = batch.experience.reward + \
            (1 - batch.experience.done) * self.gamma * next_q

        # Update critics
        critic_grads = jax.grad(lambda p: self._compute_critic_loss(
            p, batch, target_q)[0])(train_state.critic_params)
        critic_updates, critic_opt_state = self.critic_optimizer.update(
            critic_grads, train_state.critic_optim)

        # Update actor
        actor_loss, actor_info = self._compute_actor_loss(
            train_state.policy_params, train_state.critic_params, alpha, batch, rng)
        actor_grads = jax.grad(lambda p: self._compute_actor_loss(
            p, train_state.critic_params, alpha, batch, rng)[0])(train_state.policy_params)
        actor_updates, actor_opt_state = self.actor_optimizer.update(
            actor_grads, train_state.policy_optim)

        # Update alpha
        alpha_grads = jax.grad(lambda a: self._compute_alpha_loss(
            a, actor_info['entropy'])[0])(train_state.log_alpha)
        alpha_updates, alpha_opt_state = self.alpha_optimizer.update(
            alpha_grads, train_state.alpha_optim)

        # Apply updates
        new_critic_params = optax.apply_updates(
            train_state.critic_params, critic_updates)
        new_policy_params = optax.apply_updates(
            train_state.policy_params, actor_updates)
        new_log_alpha = optax.apply_updates(
            train_state.log_alpha, alpha_updates)

        # Update target network
        new_target_params = optax.incremental_update(
            new_critic_params, train_state.target_params, self.tau)

        return AgentState(
            critic_params=new_critic_params,
            target_params=new_target_params,
            policy_params=new_policy_params,
            log_alpha=new_log_alpha,
            critic_optim=critic_opt_state,
            policy_optim=actor_opt_state,
            alpha_optim=alpha_opt_state,
            policy_grads=actor_grads,
            critic_grads=critic_grads,
            timesteps=train_state.timesteps
        )

    # @partial(jax.jit, static_argnums=(0,))
    def sample_action(self, policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey, training: bool) -> chex.Array:
        if training:
            return self.actor.sample(policy_params, obs, rng)[0].squeeze()
        else:
            return jax.random.uniform(rng, self.action_shape, minval=-1.0, maxval=1.0)

    def train(self):
        """
        Main training loop with optimizations for memory and speed.
        Returns the average score across all episodes.
        """
        # Initialize tracking variables
        self.trial_return = 0.0
        self.episodes = 0
        self.steps = 0

        # Initialize networks if not already done
        if self.train_state is None:
            self.rng, init_rng = jax.random.split(self.rng)
            d_state = jnp.zeros(self.obs_shape)
            d_action = jnp.zeros(self.action_shape)

            # Initialize parameters
            critic_params = self.critic.init(init_rng, d_state, d_action)
            actor_params = self.actor.init(init_rng, d_state)
            log_alpha = jnp.zeros(1)

            self.train_state = AgentState(
                critic_params=critic_params,
                target_params=critic_params,
                policy_params=actor_params,
                log_alpha=log_alpha,
                critic_optim=self.critic_optimizer.init(critic_params),
                policy_optim=self.actor_optimizer.init(actor_params),
                alpha_optim=self.alpha_optimizer.init(log_alpha),
                policy_grads=actor_params,
                critic_grads=critic_params,
                timesteps=0
            )

        # Initialize replay buffer if not already done
        if self.buffer_state is None:
            self.buffer = fb.make_item_buffer(
                max_length=self.buffer_size,
                min_length=self.learning_starts,
                sample_batch_size=self.batch_size,
            )

            dummy_items = TimeStep(
                obs=jnp.zeros(self.obs_shape, dtype=jnp.uint8),
                action=jnp.zeros(self.action_shape),
                reward=jnp.zeros(1),
                done=jnp.zeros(1),
                next_obs=jnp.zeros(self.obs_shape, dtype=jnp.uint8)
            )
            self.buffer_state = self.buffer.init(dummy_items)

        # Get initial observation
        obs = self.env.reset()

        # Main training loop
        for step in range(self.total_timesteps):
            # Sample action
            self.rng, action_rng = jax.random.split(self.rng)
            training = self.buffer.can_sample(self.buffer_state)

            # Get action from policy or random
            action = self.sample_action(
                self.train_state.policy_params,
                obs,
                action_rng,
                training
            )

            # Convert action for environment
            step_action = modify_action(
                action, self.min_action, self.max_action)

            # Step environment
            next_obs, reward, done, _, info = self.env.step(step_action)

            # Store transition in buffer
            timestep = TimeStep(
                obs=obs,
                action=action,
                reward=jnp.array(reward),
                done=jnp.array(done, dtype=jnp.float32),
                next_obs=next_obs
            )
            self.buffer_state = self.buffer.add(self.buffer_state, timestep)

            # Training phase
            if training and self.train_state.timesteps % self.update_frequency == 0:
                self.rng, train_rng = jax.random.split(self.rng)

                # Sample batch from buffer
                batch = self.buffer.sample(self.buffer_state, train_rng)

                # Update networks
                self.train_state = self._update_step(
                    self.train_state, batch, train_rng)

            # Update tracking variables
            self.train_state = self.train_state.replace(
                timesteps=self.train_state.timesteps + 1
            )

            # actor_stats = get_statistics(self.actor, self.train_state.policy_params, obs, self.train_state.policy_grads)
            # critic_stats = get_statistics(self.critic, self.train_state.critic_params, obs, self.train_state.critic_grads)

            # Update metrics
            self.collector.collect('reward', reward)
            self.collector.collect('moving_avg', reward)

            self.trial_return += reward
            self.steps += 1

            # Handle episode termination
            if done:
                obs = self.env.reset()
                self.episodes += 1
            else:
                obs = next_obs

        # Calculate and return final score
        final_score = self.trial_return / self.steps
        return final_score

    def save_collector(self, exp, save_path):
        saveCollector(exp, self.collector, save_path)

    def load_checkpoint(self, checkpoint_path):
        pass

    def save_checkpoint(self, checkpoint_path):
        pass
