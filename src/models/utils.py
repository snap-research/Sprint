import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

def make_grid_mask_size(batch_size, n_patch_h, n_patch_w, device):
    # Create tensors directly on target device to avoid CUDA graphs skipping
    grid_h = torch.arange(n_patch_h, dtype=torch.long, device=device)
    grid_w = torch.arange(n_patch_w, dtype=torch.long, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.cat(
        [grid[0].reshape(1,-1), grid[1].reshape(1,-1)], dim=0
    ).repeat(batch_size,1,1)
    mask = torch.ones((batch_size, n_patch_h*n_patch_w), device=device, dtype=torch.bfloat16)
    # Create size tensor directly on device without CPU intermediate
    size = torch.empty((batch_size, 2), device=device, dtype=torch.long)
    size[:, 0] = n_patch_h
    size[:, 1] = n_patch_w
    size = size[:, None, :]
    return grid, mask, size

def make_3dgrid_mask_size(batch_size, n_patch_h, n_patch_w, n_patch_t, device):
    # Create tensors directly on target device to avoid CUDA graphs skipping
    grid_h = torch.arange(n_patch_h, dtype=torch.long, device=device)
    grid_w = torch.arange(n_patch_w, dtype=torch.long, device=device)
    grid_t = torch.arange(n_patch_t, dtype=torch.long, device=device)
    grid = torch.meshgrid(grid_t, grid_w, grid_h, indexing='xy')
    grid = torch.cat(
        [grid[0].reshape(1,-1), grid[1].reshape(1,-1), grid[2].reshape(1,-1)], dim=0
    ).repeat(batch_size,1,1)
    mask = torch.ones((batch_size, n_patch_h*n_patch_w*n_patch_t), device=device, dtype=torch.bfloat16)
    # Create size tensor directly on device without CPU intermediate
    size = torch.empty((batch_size, 3), device=device, dtype=torch.long)
    size[:, 0] = n_patch_t
    size[:, 1] = n_patch_h
    size[:, 2] = n_patch_w
    size = size[:, None, :]
    return grid, mask, size

@torch.no_grad()
def prepare_image_ids(batch_size, temp, height, width, start_time_stamp=0):
    latent_image_ids = torch.zeros(temp, height, width, 3)

    # Temporal Rope
    latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(start_time_stamp, start_time_stamp + temp)[:, None, None]

    # height Rope
    height_pos = torch.arange(height).float()

    latent_image_ids[..., 1] = latent_image_ids[..., 1] + height_pos[None, :, None]

    # width rope
    width_pos = torch.arange(width).float()

    latent_image_ids[..., 2] = latent_image_ids[..., 2] + width_pos[None, None, :]
    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
    latent_image_ids = rearrange(latent_image_ids, 'b t h w c -> b (t h w) c')
    return latent_image_ids

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height, device):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (
        crop_top + resize_height,
        crop_left + resize_width,
    )