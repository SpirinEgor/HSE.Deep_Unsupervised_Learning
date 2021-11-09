# borrow from https://github.com/rll/deepul

import torch
import numpy as np


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_fanin_init_bound(tensor: torch.Tensor) -> float:
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return bound


def fanin_init(tensor: torch.Tensor) -> torch.Tensor:
    bound = get_fanin_init_bound(tensor)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor: torch.Tensor) -> torch.Tensor:
    new_tensor = torch.empty_like(tensor)
    return fanin_init(new_tensor)
