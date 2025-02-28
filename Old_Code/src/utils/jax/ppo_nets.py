import distrax
from jax import numpy as jnp
from flax import linen as nn

from src.utils.jax.mis import CNN, MLP


class PPONetwork(nn.Module):
    action_dim: int
    hidden_units: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, state):
        if self.activation == "tanh":
            activation = nn.tanh
        elif self.activation == "relu":
            activation = nn.relu

        embedding = CNN()(state)
        embedding = embedding.reshape((embedding.shape[0], -1))

        # Actor
        actor_mean = MLP(self.action_dim, activation=activation, hidden_units=self.hidden_units)(embedding)
        actor_logtstd = self.param(
            "log_std", nn.initializers.zeros, (self.action_dim,))
        actor_logtstd = jnp.clip(actor_logtstd, -20.0, 2.0)
        pi = distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(actor_logtstd))

        # Critic
        critic = MLP(1, activation=activation, hidden_units=self.hidden_units)(embedding).squeeze()
        return pi, critic
