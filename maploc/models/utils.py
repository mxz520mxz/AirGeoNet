import math
from typing import Optional

import torch
import numpy as np
import torch.nn.functional as F

def checkpointed(cls, do=True):
    """Adapted from the DISK implementation of Michał Tyszkiewicz."""
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls


def pad_feature(feats_map):
   
    height = feats_map.size(2)
    width  = feats_map.size(3)
    diagonal_length = int(np.ceil(np.sqrt(height**2 + width**2)))

    padding_vertical = int(np.ceil((diagonal_length - height) / 2))
    padding_horizontal = int(np.ceil((diagonal_length - width) / 2))

    pad_top = padding_vertical // 2
    pad_bottom = padding_vertical - pad_top
    pad_left = padding_horizontal // 2
    pad_right = padding_horizontal - pad_left

    padded_tensor = F.pad(feats_map, (pad_left, pad_right, pad_top, pad_bottom, 0, 0))

    return padded_tensor


class GlobalPooling(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "mean":
            self.fn = torch.nn.Sequential(
                torch.nn.Flatten(2), torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten()
            )
        elif kind == "max":
            self.fn = torch.nn.Sequential(
                torch.nn.Flatten(2), torch.nn.AdaptiveMaxPool1d(1), torch.nn.Flatten()
            )
        else:
            raise ValueError(f"Unknown pooling type {kind}.")

    def forward(self, x):
        return self.fn(x)


# @torch.jit.script
# def make_grid(
#     w: float,
#     h: float,
#     step_x: float = 1.0,
#     step_y: float = 1.0,
#     orig_x: float = 0,
#     orig_y: float = 0,
#     y_up: bool = False,
#     device: Optional[torch.device] = None,
# ) -> torch.Tensor:
#     x, y = torch.meshgrid(
#         [
#             torch.arange(orig_x, w + orig_x, step_x, device=device),
#             torch.arange(orig_y, h + orig_y, step_y, device=device),
#         ],
#         indexing="xy",
#     )
#     if y_up:
#         y = y.flip(-2)
#     grid = torch.stack((x, y), -1)
#     return grid


@torch.jit.script
def make_grid(
    w: float,
    h: float,
    step_x: float = 1.0,
    step_y: float = 1.0,
    orig_x: float = 0,
    orig_y: float = 0,
    y_up: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    
    x_range = torch.arange(orig_x, w + orig_x, step_x, device=device)
    y_range = torch.arange(orig_y, h + orig_y, step_y, device=device)

    x = torch.linspace(orig_x, w + orig_x, steps=len(x_range), device=device).unsqueeze(1).expand(len(x_range), len(y_range))
    y = torch.linspace(orig_y, h + orig_y, steps=len(y_range), device=device).unsqueeze(0).expand(len(x_range), len(y_range))

    if y_up:
        y = y.flip(-2)
    grid = torch.stack((x, y), -1)

 
    return grid

@torch.jit.script
def rotmat2d(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([c, -s, s, c], -1).reshape(angle.shape + (2, 2))
    return R


@torch.jit.script
def rotmat2d_grad(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([-s, -c, c, -s], -1).reshape(angle.shape + (2, 2))
    return R


def deg2rad(x):
    return x * math.pi / 180


def rad2deg(x):
    return x * 180 / math.pi




