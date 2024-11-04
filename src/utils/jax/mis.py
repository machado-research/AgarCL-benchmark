import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, List, Callable
from functools import partial


def modify_action(logits, start, end):
    logits = jnp.clip(logits, start, end)
    cont_action, dis_action = logits[:-1], logits[-1]
    range_array = jnp.linspace(start, end, 4)
    insert_index = jnp.searchsorted(range_array, dis_action)
    dis_action = jnp.maximum(insert_index - 1, 0).item()
    return ([((cont_action), dis_action)])


class CNN(nn.Module):
    """Convolutional neural network"""

    @nn.compact
    def __call__(self, state):
        value = nn.Sequential([
            nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2)),
            nn.relu,
            nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1)),
            nn.relu,
            nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1)),
            nn.relu,
            nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1)),
            nn.relu,
        ])(state)
        return value


class MLP(nn.Module):
    """Multi-layer perceptron"""
    output_dim: int  # Changed from output_units to match the original code
    hidden_units: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_units)(x)
        x = self.activation(x)
        x = nn.Dense(features=self.hidden_units)(x)
        x = self.activation(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


def preprocess_observation(observation: jnp.ndarray) -> jnp.ndarray:
    return observation.astype(jnp.float32) / 255.0
