"""Predict and choose the opening role permutation from visible positions."""

from itertools import permutations

import torch
from torch import nn


ORDERS = tuple(permutations(range(3)))


def selector():
    return nn.Sequential(nn.Linear(6,24),nn.Tanh(),nn.Linear(24,1))


@torch.inference_mode()
def choose_opening_roles(policy, positions):
    features = torch.tensor([[coord / 6 for chef in order for coord in positions[chef]]
                             for order in ORDERS], dtype=torch.float32)
    return ORDERS[int(policy(features).argmax())]
