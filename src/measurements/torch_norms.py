import torch.nn as nn
import torch
import sys
sys.path.append('.')

def power_iteration(weight, n_iterations=10):
    """
    Power iteration for computing the dominant singular vector of a matrix.
    Works for both Linear and Conv2d layers.
    """
    # Get the shape of the weight matrix
    if len(weight.shape) == 4:  # Conv2d
        # Reshape (out_channels, in_channels, kernel_h, kernel_w) to (out_channels, in_channels * kernel_h * kernel_w)
        weight_matrix = weight.reshape(weight.shape[0], -1)
    else:  # Linear
        weight_matrix = weight

    # Initialize random vectors
    u = torch.randn(weight_matrix.size(0), 1, device=weight.device)
    v = torch.randn(1, weight_matrix.size(1), device=weight.device)

    for _ in range(n_iterations):
        v = torch.matmul(u.t(), weight_matrix).t()
        v = v / torch.norm(v)
        u = torch.matmul(weight_matrix, v)
        u = u / torch.norm(u)
    return u, v


def stable_rank(weight):
    """
    Calculate the stable rank of a weight matrix/tensor.
    Works for both Linear and Conv2d layers.

    Stable rank is defined as: ||W||_F^2 / ||W||_2^2
    where ||W||_F is the Frobenius norm and ||W||_2 is the spectral norm.
    """
    # Reshape if it's a conv weight
    if len(weight.shape) == 4:  # Conv2d weights
        weight_matrix = weight.reshape(weight.shape[0], -1)
    else:  # Linear weights
        weight_matrix = weight

    # Calculate singular values
    singular_values = torch.linalg.svdvals(weight_matrix)

    # Stable rank is ||W||_F^2 / ||W||_2^2
    # = sum(σᵢ²) / σ₁²
    # where σᵢ are singular values and σ₁ is the largest singular value
    stable_rank = (torch.sum(singular_values**2) /
                   (singular_values[0]**2)).item()

    return stable_rank


def get_spectral_norms(model):
    """
    Compute spectral norms for all Linear and Conv2d layers in the model.
    """
    spectral_norms = {}
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight = layer.weight.data
            u, v = power_iteration(weight)

            if isinstance(layer, nn.Conv2d):
                # Reshape weight for Conv2d
                weight_matrix = weight.reshape(weight.shape[0], -1)
                spectral_norm = torch.matmul(
                    u.t(),
                    torch.matmul(weight_matrix, v)
                )
            else:
                # Linear layer
                spectral_norm = torch.matmul(
                    u.t(),
                    torch.matmul(weight, v)
                )
            spectral_norms[name] = spectral_norm.item()
    return tuple(spectral_norms)


def get_spectral_norm_gradients(model):
    """
    Compute spectral norms of gradients for all Linear and Conv2d layers in the model.
    Must be called after loss.backward().
    """
    spectral_norms = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if layer.weight.grad is not None:
                u, v = power_iteration(layer.weight.grad)

                if isinstance(layer, nn.Conv2d):
                    # Reshape weight for Conv2d
                    weight_matrix = layer.weight.grad.reshape(
                        layer.weight.grad.shape[0], -1)
                    spectral_norm = torch.matmul(
                        u.t(),
                        torch.matmul(weight_matrix, v)
                    )
                else:
                    # Linear layer
                    spectral_norm = torch.matmul(
                        u.t(),
                        torch.matmul(layer.weight.grad, v)
                    )
                spectral_norms.append((name, spectral_norm.item()))
    return tuple(spectral_norms)


def get_layer_l2_norms(model):
    """
    Compute L2 norms of all Linear and Conv2d layers in the model.
    """
    layer_norms = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            params = torch.cat([p.view(-1) for p in layer.parameters()])
            norm = torch.norm(params, p=2).item()
            layer_norms.append((name, norm))

    return tuple(layer_norms)


def get_avg_activation_norms(model, inputs):
    activation_norms = {}
    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            activation_norms[f'{name}_{module}'] = output.norm().item()
        return fn

    for name, layer in model.named_modules():
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Tanh):
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    model(*inputs)

    for hook in hooks:
        hook.remove()

    # Calculate overall average
    overall_avg = sum(activation_norms.values()) / len(activation_norms)
    activation_norms = [(name, norm) for name, norm in activation_norms.items()]
    return tuple(activation_norms), overall_avg


def get_stable_weight_ranks(model):
    stable_ranks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            stable_ranks.append((name, stable_rank(layer.weight.data)))
    return tuple(stable_ranks)


def get_hidden_stable_ranks(model, input_tensor):
    stable_ranks = []
    activation_outputs = {}

    def hook_fn(name):
        def fn(module, input, output):
            activation_outputs[f'{name}_{module}'] = output
        return fn
    
    # Register forward hooks for all activation functions
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass
    with torch.no_grad():
        model(*input_tensor)

    # Calculate stable rank for each activation output
    for name, output in activation_outputs.items():
        output_2d = output.reshape(output.size(0), -1)
        stable_ranks.append((name, stable_rank(output_2d)))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return tuple(stable_ranks)


def count_dormant_neurons(model, input_data):
    """
    Count dormant neurons in Linear and Conv2d layers.
    A neuron is considered dormant if it never activates (outputs zero) for any input in the batch.
    """
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    hooks = []
    
    # Register hooks for each layer
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        model(*input_data)
    
    dormant_stats = []
    
    # Count dormant neurons for each layer
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            output = activations[name]
            
            if isinstance(layer, nn.Conv2d):
                # For Conv2d, reshape (batch, channels, height, width) to 
                # (batch, channels, height*width)
                batch_size = output.size(0)
                channels = output.size(1)
                output = output.view(batch_size, channels, -1)
                
                # Check if a channel is completely dormant across all spatial locations
                # Sum across batch and spatial dimensions
                dormant = torch.sum(output == 0, dim=(0, 2))
                # A channel is dormant if all its spatial locations are zero for all batch items
                dormant_count = torch.sum(
                    dormant == (batch_size * output.size(2))
                ).item()
                total_channels = channels
                
            else:  # Linear layer
                # Check if a neuron is dormant across all batch items
                dormant = torch.sum(output == 0, dim=0)
                dormant_count = torch.sum(dormant == output.size(0)).item()
                total_channels = output.size(1)
            
            # Calculate percentage
            dormant_percentage = (dormant_count / total_channels) * 100
            
            dormant_stats.append((name, dormant_count, dormant_percentage))
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    
    return tuple(dormant_stats)

def get_statistics(model, input_data):
    return {
        'spectral_norm': get_spectral_norms(model),
        'l2_norm': get_layer_l2_norms(model),
        'activation_norm': get_avg_activation_norms(model, input_data),
        'stable_weight_rank': get_stable_weight_ranks(model),
        'hidden_stable_rank': get_hidden_stable_ranks(model, input_data),
        'dormant_units': count_dormant_neurons(model, input_data),
        'spectral_norm_grad': get_spectral_norm_gradients(model)
    }


if __name__ == '__main__':
    from src.utils.ppo_networks import CNNAgent
    
    input_tensor = torch.randn(1, 64, 64, 3)

    model = CNNAgent(input_tensor.shape, (3,))
    print('Model architecture:', model)

    spectral_norms = get_spectral_norms(model)
    print('Spectral norms:', spectral_norms)

    layer_l2_norms = get_layer_l2_norms(model)
    print('Layer L2 norms:', layer_l2_norms)

    avg_activation_norms = get_avg_activation_norms(
        model, (input_tensor, torch.randn(1, 1)))
    print('Average activation norms:', avg_activation_norms)

    stable_weight_ranks = get_stable_weight_ranks(model)
    print('Stable weight ranks:', stable_weight_ranks)
    
    hidden_stable_ranks = get_hidden_stable_ranks(model, (input_tensor, torch.randn(1, 1)))
    print('Hidden stable ranks:', hidden_stable_ranks)
    
    dormant_neurons = count_dormant_neurons(model, (input_tensor, torch.randn(1, 1)))
    print('Dormant neurons:', dormant_neurons)

    # loss = torch.nn.MSELoss()
    # model.zero_grad()
    # output = model(input_tensor, torch.randn(1, 1))
    # loss(output, torch.randn_like(output)).backward()

    # spectral_norm_gradients = get_spectral_norm_gradients(model, loss)
    # print('Spectral norm gradients:', spectral_norm_gradients)
