import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.scope import FrozenVariableDict

from typing import Tuple, Dict, Any, Callable, List
import numpy as np
import optax


def power_iteration(weight: jnp.ndarray, n_iterations: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Power iteration for computing the dominant singular vector of a matrix using JAX.
    """
    # Reshape if conv weights (4D) to 2D matrix
    if len(weight.shape) == 4:  # Conv weights
        weight_matrix = weight.reshape(weight.shape[0], -1)
    else:  # Linear weights
        weight_matrix = weight

    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (weight_matrix.shape[0], 1))
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, (weight_matrix.shape[1], 1))

    def iteration_step(_, vectors):
        u, v = vectors
        v = jnp.matmul(u.T, weight_matrix).T
        v = v / jnp.linalg.norm(v)
        u = jnp.matmul(weight_matrix, v)
        u = u / jnp.linalg.norm(u)
        return (u, v)

    u, v = jax.lax.fori_loop(0, n_iterations, iteration_step, (u, v))
    return u, v


def get_spectral_norms(params: Dict) -> Dict[str, float]:
    """
    Compute spectral norms for all Linear and Conv layers in the model.
    """
    spectral_norms = {}

    def process_layer(path: str, param: Any):
        path_str = '.'.join(str(key).strip("[]'") for key in path)
        if 'kernel' in path_str:  # Convention in Flax for weight matrices
            weight = param
            u, v = power_iteration(weight)

            if len(weight.shape) == 4:  # Conv weights
                weight_matrix = weight.reshape(weight.shape[0], -1)
                spectral_norm = jnp.matmul(u.T, jnp.matmul(weight_matrix, v))
            else:  # Dense weights
                spectral_norm = jnp.matmul(u.T, jnp.matmul(weight, v))
            spectral_norms[path_str] = spectral_norm.item()

    jax.tree_util.tree_map_with_path(process_layer, params)
    return spectral_norms


def get_spectral_norm_gradients(grads: Dict) -> List[Tuple[str, float]]:
    """
    Compute spectral norms of gradients for all Linear and Conv layers in the model.

    Args:
        model: Flax model
        params: Model parameters
        inputs: Input batch
        targets: Target batch
    Returns:
        List of tuples containing (layer_name, spectral_norm)
    """
    spectral_norms = {}

    def process_layer(path, grad):
        path_str = '.'.join(str(key).strip("[]'") for key in path)
        if 'kernel' in path_str:
            u, v = power_iteration(grad)

            if len(grad.shape) == 4:  # Conv weights
                weight_matrix = grad.reshape(grad.shape[0], -1)
                spectral_norm = jnp.abs(jnp.matmul(
                    u.T, jnp.matmul(weight_matrix, v)))
            else:  # Dense weights
                spectral_norm = jnp.abs(jnp.matmul(u.T, jnp.matmul(grad, v)))

            spectral_norms[path_str] = spectral_norm.item()

    # Process all gradients
    jax.tree_util.tree_map_with_path(process_layer, grads)
    return spectral_norms


def get_layer_l2_norms(params: Dict) -> Dict[str, float]:
    """
    Compute L2 norms of all layers in the model.
    """
    layer_norms = {}

    def process_layer(path: str, param: Any):
        path_str = '.'.join(str(key).strip("[]'") for key in path)
        if 'kernel' in path_str:
            norm = jnp.linalg.norm(param).item()
            layer_norms[path_str] = norm

    jax.tree_util.tree_map_with_path(process_layer, params)
    return layer_norms


def process_activation(activation: Any) -> jnp.ndarray:
    """
    Recursively process activation to get the array value.
    """
    if isinstance(activation, tuple):
        return process_activation(activation[0])
    elif isinstance(activation, dict):
        # If it's a dictionary, process the first value
        first_key = list(activation.keys())[0]
        return process_activation(activation[first_key])
    elif isinstance(activation, jnp.ndarray):
        return activation
    else:
        raise ValueError(f"Unexpected activation type: {type(activation)}")


def get_activation_norms(model: nn.Module, params: Dict, inputs: jnp.ndarray) -> Dict[str, float]:
    """
    Compute activation norms for the model.
    Handles various types of intermediate activations (arrays, tuples, dicts).
    """
    # Forward pass collecting activations
    outputs, intermediates = model.apply(
        params, inputs, mutable=['intermediates'], capture_intermediates=True)

    # Process intermediate activations
    norms = {}
    if 'intermediates' in intermediates:
        for path, activation in intermediates['intermediates'].items():
            try:
                # Convert path tuple to string
                # path_str = '/'.join(str(p) for p in path)

                # Process the activation to get the array
                activation_array = process_activation(activation)

                # Calculate norm and convert to Python float
                norm = float(jnp.linalg.norm(activation_array))
                norms[path] = norm
            except Exception as e:
                print(
                    f"Warning: Could not process activation at {path}: {str(e)}")
                continue

    return norms


def stable_rank(weight: jnp.ndarray) -> float:
    """
    Calculate the stable rank of a weight matrix/tensor using JAX.
    """
    if len(weight.shape) == 4:  # Conv weights
        weight_matrix = weight.reshape(weight.shape[0], -1)
    else:  # Linear weights
        weight_matrix = weight

    # Calculate singular values
    singular_values = jnp.linalg.svd(weight_matrix, compute_uv=False)

    # Stable rank = ||W||_F^2 / ||W||_2^2 = sum(σᵢ²) / σ₁²
    return (jnp.sum(singular_values**2) / (singular_values[0]**2)).item()


def get_stable_ranks(params: Dict) -> Dict[str, float]:
    """
    Compute stable ranks for all layers in the model.
    """
    ranks = {}

    def process_layer(path: str, param: Any):
        path_str = '.'.join(str(key).strip("[]'") for key in path)
        if 'kernel' in path_str:
            ranks[path_str] = stable_rank(param)

    jax.tree_util.tree_map_with_path(process_layer, params)
    return ranks


def get_hidden_stable_ranks(
    model: nn.Module,
    params: Dict,
    inputs: jnp.ndarray,
    activation_layers: List[str] = None
) -> List[Tuple[str, float]]:
    """
    Compute stable ranks for hidden layer activations.

    Args:
        model: Flax model
        params: Model parameters
        inputs: Input tensor
        activation_layers: Optional list of layer names to compute stable ranks for.
                         If None, computes for all activation functions.
    Returns:
        List of tuples containing (layer_name, stable_rank)
    """
    def _forward(params: FrozenVariableDict, inputs: jnp.ndarray):
        return model.apply(
            params,
            inputs,
            capture_intermediates=True,
            mutable=['intermediates']
        )

    # Forward pass collecting activations
    outputs, intermediates = _forward(params, inputs)

    stable_ranks = {}

    if 'intermediates' in intermediates:
        for path, activation in intermediates['intermediates'].items():
            try:
                # Skip if we're only looking for specific layers and this isn't one
                if activation_layers and not any(layer in path for layer in activation_layers):
                    continue

                # Only process activation function outputs (relu, sigmoid, tanh)
                # if any(act_name in path.lower() for act_name in ['relu', 'sigmoid', 'tanh']):
                # Process the activation to get the array
                activation_array = process_activation(activation)

                # Reshape to 2D
                if len(activation_array.shape) > 2:
                    activation_2d = activation_array.reshape(
                        activation_array.shape[0], -1)
                else:
                    activation_2d = activation_array

                # Calculate stable rank
                rank = stable_rank(activation_2d)
                stable_ranks[path] = rank

            except Exception as e:
                print(
                    f"Warning: Could not process activation at {path}: {str(e)}")
                continue

    return stable_ranks


def count_dormant_neurons(
    model: nn.Module,
    params: Dict,
    inputs: jnp.ndarray,
    activation_fn: Callable = jax.nn.relu,
    threshold: float = 1e-6
) -> Dict[str, Tuple[int, float]]:
    """
    Count dormant neurons (neurons that never activate) in each layer.

    Args:
        model: Flax model
        params: Model parameters
        inputs: Input batch
        activation_fn: Activation function to check (default: ReLU)
        threshold: Threshold below which a neuron is considered dormant

    Returns:
        Dictionary mapping layer names to tuples of (dormant_count, dormant_percentage)
    """
    def _forward(params: FrozenVariableDict, inputs: jnp.ndarray):
        return model.apply(
            params,
            inputs,
            capture_intermediates=True,
            mutable=['intermediates']
        )

    # Forward pass collecting activations
    outputs, intermediates = _forward(params, inputs)

    dormant_stats = {}

    if 'intermediates' in intermediates:
        for path, activation in intermediates['intermediates'].items():
            try:
                # Only process activation function outputs
                # if any(act_name in path.lower() for act_name in ['relu', 'sigmoid', 'tanh']):
                # Process the activation to get the array
                activation_array = process_activation(activation)

                # Check activation shape and process accordingly
                if len(activation_array.shape) == 4:  # Conv layer
                    # Reshape (batch, channels, height, width) to (batch, channels, height*width)
                    reshaped = activation_array.reshape(
                        activation_array.shape[0],
                        activation_array.shape[1],
                        -1
                    )
                    # Count neurons that never activate above threshold
                    dormant = jnp.sum(jnp.abs(reshaped) <=
                                      threshold, axis=(0, 2))
                    dormant_count = jnp.sum(
                        dormant == (
                            activation_array.shape[0] * reshaped.shape[2])
                    ).item()
                    total_units = activation_array.shape[1]

                elif len(activation_array.shape) >= 2:  # Dense layer or other
                    # Reshape to (batch_size, features)
                    reshaped = activation_array.reshape(
                        activation_array.shape[0], -1)
                    # Count neurons that never activate above threshold
                    dormant = jnp.sum(jnp.abs(reshaped) <= threshold, axis=0)
                    dormant_count = jnp.sum(
                        dormant == activation_array.shape[0]).item()
                    total_units = reshaped.shape[1]

                else:
                    continue  # Skip 1D activations

                dormant_percentage = (dormant_count / total_units) * 100
                dormant_stats[path] = (dormant_count, dormant_percentage)

            except Exception as e:
                print(
                    f"Warning: Could not process activation at {path}: {str(e)}")
                continue

    return dormant_stats


def get_statistics(model: nn.Module, params: Dict, inputs: jnp.ndarray, grads: Dict) -> Dict:
    """
    Get all statistics for a model.
    """
    return {
        'spectral_norm': get_spectral_norms(params),
        'spectral_norm_grad': get_spectral_norm_gradients(grads),
        'l2_norm': get_layer_l2_norms(params),
        'activation_norm': get_activation_norms(model, params, inputs),
        'stable_weight_rank': get_stable_ranks(params),
        'hidden_stable_rank': get_hidden_stable_ranks(model, params, inputs),
        'dormant_units': count_dormant_neurons(model, params, inputs)
    }


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.utils.jax.ppo_nets import PPONetwork

    # Initialize model and parameters
    key = jax.random.PRNGKey(0)
    actor = PPONetwork(4, 64)

    dummy_input = jnp.ones((1, 64, 64, 3))
    params = actor.init(key, dummy_input)
    
    # calculate grads
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(params)
    def loss_fn(params):
        pi, q = actor.apply(params, dummy_input)
        action = pi.sample(seed=0)
        log_probs = pi.log_prob(action)
        actor_loss = jnp.mean(0.01 * log_probs - q)
        return actor_loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)

    stats = get_statistics(actor, params, dummy_input, grads)
    print("Model statistics:", stats)
