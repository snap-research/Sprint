import torch
import math
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from collections import OrderedDict
from safetensors.torch import load_file, save_file
import re


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def preprocess_raw_image(x, enc_type, resolution=256):
    if 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    else:
        raise NotImplementedError
    return x

# @torch.no_grad()
# def update_ema(ema_model, model, decay=0.9999):
#     """
#     Step the EMA model towards the current model.
#     """
#     ema_params = OrderedDict(ema_model.named_parameters())
#     model_params = OrderedDict(model.named_parameters())

#     for name, param in model_params.items():
#         # name = name.replace("module.", "")
#         # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
#         ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    #for name, param in model_params.items():
    for (name1, param1), (name2, param2) in zip(ema_params.items(), model_params.items()):
        if name1 == name2:
            ema_params[name1].mul_(decay).add_(param2.data, alpha=1 - decay)
        else:
            name2 = name2.replace("module.", "")
            name2 = name2.replace("_orig_mod.", "")
            ema_params[name2].mul_(decay).add_(param2.data, alpha=1 - decay)

@torch.no_grad()
def update_ema_fsdp(ema_model, model, accelerator, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    with torch.no_grad():
        msd = accelerator.get_state_dict(model)
        # msd = model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            if k in msd:
                model_v = msd[k].detach().to(ema_v.device, dtype=ema_v.dtype)
                ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512) # currently only support 256x256 and 512x512

    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        architectures.append(architecture)
        encoder_types.append(encoder_type)

        if 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        encoders.append(encoder)
    return encoders, encoder_types, architectures

def find_model(arch):
    if arch == 'SiT':
        from src.models.sit import SiT_models
    else:
        raise NotImplementedError
    return SiT_models

def init_from_ckpt(
    model, checkpoint_dir, ignore_keys=None, verbose=False
) -> None:
    if checkpoint_dir.endswith(".safetensors"):
        try:
            model_state_dict=load_file(checkpoint_dir)
        except:
            model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    else:
        model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    model_new_ckpt=dict()
    for i in model_state_dict.keys():
        model_new_ckpt[i] = model_state_dict[i]

    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(model_new_ckpt.keys())

    # Handle _orig_mod prefix mismatches from torch.compile
    if checkpoint_keys != model_keys:
        # Check if model has _orig_mod prefix but checkpoint doesn't
        model_has_orig_mod = any(k.startswith('_orig_mod.') for k in model_keys)
        checkpoint_has_orig_mod = any(k.startswith('_orig_mod.') for k in checkpoint_keys)

        if model_has_orig_mod and not checkpoint_has_orig_mod:
            # Add _orig_mod prefix to checkpoint keys
            new_model_new_ckpt = {}
            for k, v in model_new_ckpt.items():
                new_key = f'_orig_mod.{k}'
                new_model_new_ckpt[new_key] = v
            model_new_ckpt = new_model_new_ckpt
            if verbose:
                print("Added '_orig_mod.' prefix to checkpoint keys to match compiled model.")

        elif not model_has_orig_mod and checkpoint_has_orig_mod:
            # Remove _orig_mod prefix from checkpoint keys
            new_model_new_ckpt = {}
            for k, v in model_new_ckpt.items():
                if k.startswith('_orig_mod.'):
                    new_key = k[len('_orig_mod.'):]
                    new_model_new_ckpt[new_key] = v
                else:
                    new_model_new_ckpt[k] = v
            model_new_ckpt = new_model_new_ckpt
            if verbose:
                print("Removed '_orig_mod.' prefix from checkpoint keys to match non-compiled model.")

    keys = list(model_new_ckpt.keys())
    for k in keys:
        if ignore_keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    print("Deleting key {} from state_dict.".format(k))
                    del model_new_ckpt[k]
    missing, unexpected = model.load_state_dict(model_new_ckpt, strict=False)
    if verbose:
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    if verbose:
        print("")

def change_keys_from_ckpt(
    checkpoint_dir, ignore_keys=None, verbose=False, save_path=None
) -> dict:
    if checkpoint_dir.endswith(".safetensors"):
        try:
            model_state_dict=load_file(checkpoint_dir)
        except:
            model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    else:
        model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    model_new_ckpt = dict(model_state_dict)

    # Remap specific prefixes safely:
    # - "mixer_blocks" -> "encoder_blocks"
    # - "blocks" segment -> "middle_blocks" (but do NOT touch "decoder_blocks")
    remapped_ckpt = {}
    for k, v in model_new_ckpt.items():
        new_k = k
        # Replace segment named 'mixer_blocks' only when it appears as its own path segment
        new_k = re.sub(r'(?:(?<=^)|(?<=\.))mixer_blocks(?=\.|$)', 'encoder_blocks', new_k)
        # Replace segment named 'blocks' (standalone) with 'middle_blocks';
        # this will not match 'decoder_blocks' since that is a different segment name
        new_k = re.sub(r'(?:(?<=^)|(?<=\.))blocks(?=\.|$)', 'middle_blocks', new_k)
        if verbose and new_k != k:
            print(f"Remap key: {k} -> {new_k}")
        remapped_ckpt[new_k] = v
    model_new_ckpt = remapped_ckpt

    if ignore_keys:
        keys = list(model_new_ckpt.keys())
        for k in keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    if verbose:
                        print("Deleting key {} from state_dict.".format(k))
                    del model_new_ckpt[k]
    # Optionally save remapped checkpoint to .safetensors
    if save_path is not None:
        if not save_path.endswith('.safetensors'):
            save_path = save_path + '.safetensors'
        cpu_ckpt = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in model_new_ckpt.items()}
        save_file(cpu_ckpt, save_path)
        if verbose:
            print(f"Saved remapped checkpoint to {save_path}")
    return model_new_ckpt

#################################################################################
#                              Token dropping Functions                         #
#################################################################################

def mask_out_token(x, ids_keep):
    """
    Mask out tokens in the input tensor x based on ids_keep.
    Args:
        x: Input tensor of shape (B, C, L) or (B, L).
        ids_keep: Indices of tokens to keep.
    Returns:
        x_masked: Tensor with masked tokens.
    """
    B, L, D = x.shape
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked

def mask_out_pos_token(pos, ids_keep):
    """
    Mask out positional tokens in the input tensor pos based on ids_keep.
    Args:
        pos: Input tensor of shape (B, L, D).
        ids_keep: Indices of tokens to keep.
    Returns:
        pos_masked: Tensor with masked positional tokens.
    """
    B, _, L, D = pos.shape
    pos_masked = torch.gather(pos, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, D))
    return pos_masked

def mask_out_pos_token_wan(pos, ids_keep):
    """
    Mask out positional tokens in the input tensor pos based on ids_keep.
    Args:
        pos: Input tensor of shape (B, L, D).
        ids_keep: Indices of tokens to keep.
    Returns:
        pos_masked: Tensor with masked positional tokens.
    """
    B, _, L, D = pos.shape
    pos_masked = torch.gather(pos, dim=1, index=ids_keep.unsqueeze(2).unsqueeze(-1).repeat(1, 1, 1, D))
    return pos_masked

#################################################################################
#                              Masking Functions                                #
#################################################################################

def prepare_mask_and_input(
        tokens_h=None,
        tokens_w=None,
        tokens_t=None,
        images=None,
        mask_type=None,
        mask_ratio=None,
    ):
    """
    Prepare mask_dict based on mask_type and mask_ratio.
    Returns: mask_dict
    """
    if mask_type == 'structured_with_random_offset':
        if mask_ratio == 0.75:
            group_size = 2
            selection_per_group = 1
        elif mask_ratio == 0.625:
            group_size = 4
            selection_per_group = 6
        elif mask_ratio == 0.875:
            group_size = 4
            selection_per_group = 2
        elif mask_ratio == 0.5:
            group_size = 2
            selection_per_group = 2
        mask_dict = get_structured_mask_with_random_offset(tokens_h=tokens_h, tokens_w=tokens_w, x=images, group_size=group_size, selection_per_group=selection_per_group)
    elif mask_type == 'structured_with_random_offset_3D':
        if mask_ratio == 0.75:
            group_size = 2
            group_size_t = 2
            selection_per_group = 2
        else:
            raise ValueError(f"Unknown mask_ratio: {mask_ratio}")
        mask_dict = get_structured_mask_with_random_offset_3D(tokens_h=tokens_h, tokens_w=tokens_w, tokens_t=tokens_t, x=images, group_size=group_size, group_size_t=group_size_t, selection_per_group=selection_per_group)
    elif mask_type == 'random':
        mask_dict = get_random_mask(x=images, mask_ratio=mask_ratio)
    else:
        raise NotImplementedError

    return mask_dict

def get_random_mask(x, mask_ratio):
    if x.ndim == 3:
        B, seq_length, _ = x.shape
    elif x.ndim == 4:
        B, C, H, W = x.shape
        seq_length = (H // 2) * (W // 2)
    elif x.ndim == 5:
        B, C, T, H, W = x.shape
        seq_length = T * (H // 2) * (W // 2)
    else:
        raise ValueError("Input x must be 3D, 4D or 5D tensor.")
    len_keep = int(seq_length * (1 - mask_ratio))
    noise = torch.rand((B, seq_length), device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_remove = ids_shuffle[:, len_keep:]
    mask = torch.ones((B, seq_length), device=x.device)
    mask[:, :len_keep] = 0  # 0 is keep, 1 is remove
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return {
        'mask':        mask,               # [B, N]  (True = masked)
        'ids_keep':    ids_keep,           # [B, M]
        'ids_remove':  ids_remove,         # [B, N-M]
        'ids_restore': ids_restore,        # [B, N]
        'ids_shuffle': None,
        'offsets':     None,
    }

def get_structured_mask_with_random_offset(
    tokens_h: int | None = None,
    tokens_w: int | None = None,
    x: torch.Tensor = None,
    group_size: int = 2,
    selection_per_group: int = 1,
    *,
    generator: torch.Generator | None = None,
):
    """
    Parallelized version supporting variable group_size and selection_per_group.

    Args:
      x: [B, N, D] tokens produced from an HxWxC grid (N = H*W).
      tokens_h, tokens_w: optional grid size in tokens. If None, infer square side.
      group_size: size of each square group (e.g., 2 for 2x2 groups, 4 for 4x4 groups).
      selection_per_group: number of tokens to select from each group.
      generator: optional torch.Generator for reproducible randomness.

    Returns:
      ids_keep:      LongTensor [B, num_groups*selection_per_group]    selected tokens
      ids_remove:    LongTensor [B, N - num_groups*selection_per_group]   complement of ids_keep (ascending)
      ids_restore:   LongTensor [B, N]       inverse permutation of cat([ids_keep, ids_remove], dim=1)
      mask:          BoolTensor  [B, N]      True for masked (dropped), False for kept
      offsets:       (selected_rel_h, selected_rel_w) tensors with selected positions per group
      ids_shuffle:   None (kept for API compatibility)
    """
    if x is not None:
        if x.ndim == 3:
            B, N, _ = x.shape
            H = W = int(N ** 0.5)
            assert H * W == N, "Sequence length must be a perfect square for block masking."
        elif x.ndim == 4:
            B, C, H, W = x.shape
            H, W = H // 2, W // 2 # TODO: replace hard code for patch size 2x2
            N = H * W
        else:
            raise ValueError("Input x must be 3D or 4D tensor.")

    device = x.device
    # Infer H=W if not provided
    if tokens_h is None or tokens_w is None:
        side = int(math.isqrt(N))
        assert side * side == N, "N must be a perfect square or pass tokens_h/tokens_w."
        tokens_h = tokens_w = side

    H, W = tokens_h, tokens_w
    assert (H % group_size == 0) and (W % group_size == 0), "H and W must be divisible by group_size."
    assert selection_per_group <= group_size * group_size, "selection_per_group cannot exceed group_size^2"

    Hs, Ws = H // group_size, W // group_size  # number of groups along H and W
    num_groups = Hs * Ws
    M = num_groups * selection_per_group  # total number of kept tokens

    # Batch-parallel random selection for all groups
    # Shape: [B, Hs, Ws, group_size^2] - random values for each position in each group
    random_vals = torch.rand(B, Hs, Ws, group_size * group_size, device=device, generator=generator)

    # Get the indices of the top selection_per_group values for each group
    # selected_indices: [B, Hs, Ws, selection_per_group] - relative indices within each group
    _, selected_indices = torch.topk(random_vals, selection_per_group, dim=-1)

    # Convert relative indices to (rel_h, rel_w) coordinates within each group
    # selected_indices contains values in [0, group_size^2), convert to (h,w) coords
    selected_rel_h = selected_indices // group_size  # [B, Hs, Ws, selection_per_group]
    selected_rel_w = selected_indices % group_size   # [B, Hs, Ws, selection_per_group]

    # Convert to global coordinates
    # Create base coordinates for each group
    group_base_h = torch.arange(Hs, device=device).view(1, Hs, 1, 1) * group_size  # [1, Hs, 1, 1]
    group_base_w = torch.arange(Ws, device=device).view(1, 1, Ws, 1) * group_size  # [1, 1, Ws, 1]

    # Add group base coordinates to relative coordinates
    global_h = group_base_h + selected_rel_h  # [B, Hs, Ws, selection_per_group]
    global_w = group_base_w + selected_rel_w  # [B, Hs, Ws, selection_per_group]

    # Convert to linear indices and flatten to get ids_keep
    global_indices = global_h * W + global_w  # [B, Hs, Ws, selection_per_group]
    ids_keep = global_indices.reshape(B, M)  # [B, M] where M = Hs*Ws*selection_per_group

    # For compatibility, create offsets info (optional, mainly for debugging)
    selected_positions_per_group = (selected_rel_h, selected_rel_w)  # Store as tensors instead of lists

    # Complement indices (ascending by construction)
    all_idx = torch.arange(H * W, device=device).unsqueeze(0).expand(B, -1)  # [B, N]
    temp_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    temp_mask[torch.arange(B, device=device)[:, None], ids_keep] = False
    ids_remove = all_idx.masked_select(temp_mask).view(B, N - M)  # [B, N-M], ascending

    # Binary mask: True = masked (dropped), False = kept
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[torch.arange(B, device=device)[:, None], ids_keep] = False

    # Inverse permutation to restore original order after perm = cat([ids_keep, ids_remove], 1)
    perm = torch.cat([ids_keep, ids_remove], dim=1)  # [B, N]
    ids_restore = torch.empty_like(perm)
    ids_restore.scatter_(1, perm, torch.arange(N, device=device).unsqueeze(0).expand(B, -1))

    return {
        'mask':        mask,               # [B, N]  (True = masked)
        'ids_keep':    ids_keep,           # [B, M]
        'ids_remove':  ids_remove,         # [B, N-M]
        'ids_restore': ids_restore,        # [B, N]
        'ids_shuffle': None,
        'offsets':     selected_positions_per_group,  # Per-group selected positions for each batch
    }

def get_structured_mask_with_random_offset_3D(
    tokens_h: int | None = None,
    tokens_w: int | None = None,
    tokens_t: int | None = None,
    x: torch.Tensor = None,
    group_size: int = 2,
    group_size_t: int = 1,
    selection_per_group: int = 1,
    *,
    generator: torch.Generator | None = None,
):
    """
    3D version for videos. Groups tokens over time-height-width and selects within each group.

    Args:
      x: One of
         - [B, N, D] tokens produced from a T x H x W grid (N = T*H*W)
         - [B, C, T, H, W] video volume (will be mapped to tokens by patch rules)
         - [B, C, H, W] image volume (treated as T=1)
      tokens_t, tokens_h, tokens_w: optional sizes in tokens. If None, infer assuming square spatial grid
                                     for the spatial part and T=1 unless tokens_t is provided.
      group_size_t, group_size: temporal and spatial group sizes (group_size applies to both H and W).
      selection_per_group: number of tokens to select from each 3D group.
      generator: optional torch.Generator for reproducible randomness.

    Returns:
      mask:        BoolTensor [B, N] (True = masked)
      ids_keep:    LongTensor [B, num_groups * selection_per_group]
      ids_remove:  LongTensor [B, N - num_groups * selection_per_group]
      ids_restore: LongTensor [B, N]
      ids_shuffle: None
      offsets:     (selected_rel_t, selected_rel_h, selected_rel_w) per group
    """
    if x is None:
        raise ValueError("x must be provided.")

    device = x.device

    # Infer token grid sizes based on input rank
    if x.ndim == 5:
        B, C, T_img, H_img, W_img = x.shape
        # Assume spatial patch size 2x2 as in 2D function; temporal patch size 1
        T = T_img
        H = H_img // 2
        W = W_img // 2
        N = T * H * W
    elif x.ndim == 4:
        B, C, H_img, W_img = x.shape
        T = 1
        H = H_img // 2
        W = W_img // 2
        N = T * H * W
    elif x.ndim == 3:
        B, N, _ = x.shape
        if tokens_t is None:
            # Treat as a single frame by default
            T = 1
            spatial = N
        else:
            T = tokens_t
            assert N % T == 0, "Sequence length must be divisible by tokens_t."
            spatial = N // T
        if (tokens_h is None) or (tokens_w is None):
            side = int(math.isqrt(spatial))
            assert side * side == spatial, "Spatial tokens must form a square grid or pass tokens_h/tokens_w."
            H = W = side
        else:
            H, W = tokens_h, tokens_w
    else:
        raise ValueError("Input x must be 3D, 4D or 5D tensor.")

    # Allow overriding with explicit tokens_t/h/w if provided
    if tokens_t is not None:
        T = tokens_t
    if (tokens_h is not None) and (tokens_w is not None):
        H, W = tokens_h, tokens_w

    # Validations
    assert (T % group_size_t == 0), "T must be divisible by group_size_t."
    assert (H % group_size == 0) and (W % group_size == 0), "H and W must be divisible by group_size."

    Ts, Hs, Ws = T // group_size_t, H // group_size, W // group_size
    num_groups = Ts * Hs * Ws
    group_volume = group_size_t * group_size * group_size
    assert selection_per_group <= group_volume, "selection_per_group cannot exceed group_size_t*group_size*group_size"

    M = num_groups * selection_per_group
    N = T * H * W

    # Random values per position within each 3D group
    random_vals = torch.rand(
        B,
        Ts,
        Hs,
        Ws,
        group_volume,
        device=device,
        generator=generator,
    )

    # Top-k indices within each group's flattened local coordinates
    _, selected_indices = torch.topk(random_vals, selection_per_group, dim=-1)

    # Map flattened local index -> (rel_t, rel_h, rel_w)
    rel_t = selected_indices // (group_size * group_size)
    rem = selected_indices % (group_size * group_size)
    rel_h = rem // group_size
    rel_w = rem % group_size

    # Base coordinates for each group
    base_t = torch.arange(Ts, device=device).view(1, Ts, 1, 1, 1) * group_size_t
    base_h = torch.arange(Hs, device=device).view(1, 1, Hs, 1, 1) * group_size
    base_w = torch.arange(Ws, device=device).view(1, 1, 1, Ws, 1) * group_size

    # Global coordinates
    global_t = base_t + rel_t
    global_h = base_h + rel_h
    global_w = base_w + rel_w

    # Linearize to token indices with W-fastest, then H, then T
    global_indices = ((global_t * H) + global_h) * W + global_w  # [B, Ts, Hs, Ws, selection_per_group]
    ids_keep = global_indices.reshape(B, M)

    # Offsets for debugging/analysis
    selected_positions_per_group = (rel_t, rel_h, rel_w)

    # Complement indices (ascending by construction)
    all_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    temp_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    temp_mask[torch.arange(B, device=device)[:, None], ids_keep] = False
    ids_remove = all_idx.masked_select(temp_mask).view(B, N - M)

    # Binary mask: True = masked (dropped), False = kept
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[torch.arange(B, device=device)[:, None], ids_keep] = False

    # Inverse permutation to restore original order after perm = cat([ids_keep, ids_remove], 1)
    perm = torch.cat([ids_keep, ids_remove], dim=1)
    ids_restore = torch.empty_like(perm)
    ids_restore.scatter_(1, perm, torch.arange(N, device=device).unsqueeze(0).expand(B, -1))

    return {
        'mask':        mask,
        'ids_keep':    ids_keep,
        'ids_remove':  ids_remove,
        'ids_restore': ids_restore,
        'ids_shuffle': None,
        'offsets':     selected_positions_per_group,
    }
