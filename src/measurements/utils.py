#!/usr/bin/env python3

import torch

def power_iteration(weight, n_iterations=10):
    u = torch.randn(weight.size(0), 1)
    v = torch.randn(1, weight.size(1))
    for _ in range(n_iterations):
        v = torch.matmul(u.t(), weight).t()
        v = v / torch.norm(v)
        u = torch.matmul(weight, v)
        u = u / torch.norm(u)
    return u, v

def stable_rank(weight):
    singular_values = torch.linalg.svdvals(weight)
    stable_rank = (torch.sum(singular_values**2) / (singular_values[0]**2)).item()
    return stable_rank
