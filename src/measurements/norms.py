#!/usr/bin/env python3

import torch
import torch.nn as nn
from .utils import spectral_norm, stable_rank

def get_spectral_norms(model):
    spectral_norms = []
    for layer in model:
        # TODO: for nn.Conv2d, need to figure out correct reshaping
        if isinstance(layer, nn.Linear):
            u, v = power_iteration(layer.weight)
            spectral_norm = torch.matmul(u.t(), torch.matmul(layer.weight, v))
            spectral_norms.append(spectral_norm.item())
    return spectral_norms

def get_spectral_norm_gradients(model, loss):
    # Must be used after backward
    # loss.backward()

    spectral_norms = []
    for name, layer in model:
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            # Compute spectral norm
            u, v = power_iteration(layer.weight.grad)
            spectral_norm = torch.matmul(u.t(), torch.matmul(layer.weight.grad, v))
            spectral_norms.append((name, spectral_norm))

    return spectral_norms

def get_layer_l2_norms(model):
    layer_norms = []
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            params = torch.cat([p.view(-1) for p in layer.parameters()])
            norm = torch.norm(params, p=2).item()
            layer_norms.append((name, norm))
    return layer_norms

def get_avg_activation_norms(model, inputs):
    activation_norms = []
    hooks = []

    def hook_fn(module, input, output):
        activation_norms.append(output.norm().item())

    for layer in model:
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Tanh):
            hooks.append(layer.register_forward_hook(hook_fn))

    model(inputs)

    for hook in hooks:
        hook.remove()

    # TODO: Should return norm per-layer or average norM/
    # return [sum(norms) / len(norms) for norms in zip(*[activation_norms[i::
    return activation_norms


def get_stable_weight_ranks(model):
    stable_ranks = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            stable_ranks.append(stable_rank(weight))
    return stable_ranks

def get_hidden_stable_ranks(model, input_tensor):
    stable_ranks = []
    activation_outputs = []

    def hook_fn(module, input, output):
        activation_outputs.append(output)

    # Register forward hooks for all activation functions
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        model(input_tensor)

    # Calculate stable rank for each activation output
    for output in activation_outputs:
        output_2d = output.view(output.size(0), -1)
        stable_ranks.append(stable_rank(output_2d))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return stable_ranks

def count_dormant_neurons(model, input_data):
    dormant_counts = []
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    # Register hooks for each layer
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            layer.register_forward_hook(hook_fn(name))

    # Forward pass
    model(input_data)

    # Count dormant neurons
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            output = activations[name]
            dormant = torch.sum(output == 0, dim=0)
            dormant_count = torch.sum(dormant == output.size(0)).item()
            dormant_counts.append((name, dormant_count))

    return dormant_counts
