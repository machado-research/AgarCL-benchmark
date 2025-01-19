from src.utils.jax.mis import CNN, MLP, preprocess_observation
from typing import Optional, Any
from flax import linen as nn
import jax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from functools import partial

tfd = tfp.distributions


class Network(nn.Module):
    """Memory-efficient base network with gradient checkpointing"""
    hidden_units: int 
    output_dim: int

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray = None) -> jnp.ndarray:
        # Use gradient checkpointing for the main computation
        # @jax.checkpoint
        def network_forward(state, action=None):
            # Preprocess and encode state
            state = preprocess_observation(state)
            embedding = CNN()(state)
            x = embedding.reshape((embedding.shape[0], -1))

            # Concatenate action if provided
            if action is not None:
                if action.ndim == 1:
                    action = action[None, :]
                x = jnp.concatenate([x, action], axis=-1)

            # Pass through MLP
            x = MLP(self.output_dim, self.hidden_units)(x)
            return x

        return network_forward(state, action)


class SACCriticNetwork(nn.Module):
    """Memory-efficient critic network"""
    hidden_units: int

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Use gradient checkpointing for critic computations
        def critic_forward(state, action):
            q1 = Network(self.hidden_units, 1)(state, action).squeeze()
            q2 = Network(self.hidden_units, 1)(state, action).squeeze()
            return q1, q2

        return critic_forward(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_values(self, params, state: jnp.ndarray, action: jnp.ndarray):
        return self.apply(params, state, action)


class SACActorNetwork(nn.Module):
    """Memory-efficient actor network"""
    action_dim: int
    hidden_units: int
    log_min_std: float = -5
    log_max_std: float = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> tfd.Distribution:
        # Use gradient checkpointing for actor computations
        def actor_forward(state):
            x = Network(self.hidden_units, self.action_dim)(state)

            # Compute mean and log_std
            mean = nn.Dense(
                features=self.action_dim,
                kernel_init=nn.initializers.orthogonal(),
                bias_init=nn.initializers.zeros
            )(x)

            log_std = nn.Dense(
                features=self.action_dim,
                kernel_init=nn.initializers.orthogonal(),
                bias_init=nn.initializers.zeros
            )(x)

            log_std = jnp.clip(log_std, self.log_min_std, self.log_max_std)

            return mean, log_std

        mean, log_std = actor_forward(state)

        # Create distribution
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(
                loc=mean,
                scale_diag=jnp.exp(log_std)
            )
        )
        return dist

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, params, state: jnp.ndarray, key: jnp.ndarray):
        dist = self.apply(params, state)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        return actions, log_probs


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution,
            bijector=tfp.bijectors.Tanh(),
            validate_args=validate_args
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
