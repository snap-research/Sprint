import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from functools import partial
from einops import rearrange, repeat
from torch.nn.attention import SDPBackend, sdpa_kernel
from timm.layers.mlp import SwiGLU, Mlp

from src.utils.utils import mask_out_pos_token, mask_out_token, init_from_ckpt
from src.models.rope import rotate_half, VisionRotaryEmbedding
from src.models.utils import modulate, make_grid_mask_size, get_parameter_dtype
from src.models.norms import create_norm

#################################################################################
#           Embedding Layers for Patches, Timesteps and Class Labels            #
#################################################################################

class PatchEmbedder(nn.Module):
    """
    Embeds latent features into vector representations
    """
    def __init__(self,
        input_dim,
        embed_dim,
        bias: bool = True,
        norm_layer: Optional[Callable] = None,
    ):
        super().__init__()

        self.proj = nn.Linear(input_dim, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)    # (B, L, patch_size ** 2 * C) -> (B, L, D)
        x = self.norm(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1]).to(device=t.device)], dim=-1)
        return embedding.to(dtype=t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                  Attention                                    #
#################################################################################
class Attention(nn.Module):

    def __init__(self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        rel_pos_embed: Optional[str] = None,
        add_rel_pe_to_v: bool = False,
        **block_kwargs
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if q_norm == 'layernorm' and qk_norm_weight == True:
            q_norm = 'w_layernorm'
        if k_norm == 'layernorm' and qk_norm_weight == True:
            k_norm = 'w_layernorm'

        self.q_norm = create_norm(q_norm, self.head_dim)
        self.k_norm = create_norm(k_norm, self.head_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rel_pos_embed = None if rel_pos_embed==None else rel_pos_embed.lower()
        self.add_rel_pe_to_v = add_rel_pe_to_v

    def forward(self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # (B, n_h, N, D_h)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rel_pos_embed in ['rope', 'xpos']:  # multiplicative rel_pos_embed
            if self.add_rel_pe_to_v:
                v = v * freqs_cos + rotate_half(v) * freqs_sin
            q = q * freqs_cos + rotate_half(q) * freqs_sin
            k = k * freqs_cos + rotate_half(k) * freqs_sin

        if mask is None:
            attn_mask = None
        else:
            attn_mask = mask[:, None, None, :]  # (B, N) -> (B, 1, 1, N)
            attn_mask = (attn_mask == attn_mask.transpose(-2, -1))  # (B, 1, 1, N) x (B, 1, N, 1) -> (B, 1, N, N)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if x.device.type == "cpu":
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    scale=self.scale
                ).to(x.dtype)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#################################################################################
#                               Basic SiT Blocks                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        swiglu=False,
        swiglu_large=True,
        rel_pos_embed=None,
        add_rel_pe_to_v=False,
        norm_layer: str = 'layernorm',
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        qkv_bias=True,
        ffn_bias=True,
        adaln_bias=True,
        adaln_type='normal',
        adaln_lora_dim: int = None,
        c_embed_dim: int = None,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = create_norm(norm_layer, hidden_size)
        self.norm2 = create_norm(norm_layer, hidden_size)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, rel_pos_embed=rel_pos_embed,
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight,
            qkv_bias=qkv_bias, add_rel_pe_to_v=add_rel_pe_to_v,
            **block_kwargs
        )
        self.c_embed_dim = c_embed_dim if c_embed_dim is not None else hidden_size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if swiglu:
            if swiglu_large:
                self.mlp = SwiGLU(in_features=hidden_size, hidden_features=mlp_hidden_dim, bias=ffn_bias)
            else:
                self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=ffn_bias)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), bias=ffn_bias)
        if adaln_type == 'normal':
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.c_embed_dim, 6 * hidden_size, bias=adaln_bias)
            )
        elif adaln_type == 'lora':
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=adaln_bias),
                nn.Linear(adaln_lora_dim, 6 * hidden_size, bias=adaln_bias)
            )
        elif adaln_type == 'swiglu':
            self.adaLN_modulation = SwiGLU(
                in_features=hidden_size, hidden_features=(hidden_size//4)*3, out_features=6*hidden_size, bias=adaln_bias
            )

    def forward(self, x, c, freqs_cos, freqs_sin, mask=None, global_adaln=0.0):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaLN_modulation(c) + global_adaln).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask, freqs_cos, freqs_sin)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, norm_layer: str = 'layernorm', adaln_bias=True, adaln_type='normal', c_embed_dim: int = None):
        super().__init__()
        self.norm_final = create_norm(norm_type=norm_layer, dim=hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.c_embed_dim = c_embed_dim if c_embed_dim is not None else hidden_size
        if adaln_type == 'swiglu':
            self.adaLN_modulation = SwiGLU(in_features=hidden_size, hidden_features=hidden_size//2, out_features=2*hidden_size, bias=adaln_bias)
        else:   # adaln_type in ['normal', 'lora']
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.c_embed_dim, 2 * hidden_size, bias=adaln_bias)
            )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                                 Core SiT Model                                #
#################################################################################


class SiT(nn.Module):
    def __init__(
        self,
        context_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = False,
        use_sit: bool = True,
        use_checkpoint: bool=False,
        use_swiglu: bool = False,
        use_swiglu_large: bool = False,
        rel_pos_embed: Optional[str] = 'rope',
        norm_type: str = "layernorm",
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        adaln_bias: bool = True,
        adaln_type: str = "normal",
        adaln_lora_dim: int = None,
        rope_theta: float = 10000.0,
        custom_freqs: str = 'normal',
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
        online_rope: bool = False,
        add_rel_pe_to_v: bool = False,
        pretrain_ckpt: str = None,
        ignore_keys: list = None,
        finetune: bool = False,
        time_shifting: int = 1,
        # REPA
        representation_align: bool = False,
        representation_depth: int = 8,
        dim_projection = 768,
        projector_hidden_dim = 2048,
        # SPRINT
        decoder_depth: int = 2,
        middle_depth: int = 24,
        encoder_depth: int = 2,
        use_mask_restore: bool = True,
        renoise: bool = True,
        residual_type: str = 'concat_linear',
        cfg_mask_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.context_size = context_size
        self.hidden_size = hidden_size
        assert not (learn_sigma and use_sit)
        self.learn_sigma = learn_sigma
        self.use_sit = use_sit
        self.use_checkpoint = use_checkpoint
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = self.in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.adaln_type = adaln_type
        self.online_rope = online_rope
        self.time_shifting = time_shifting
        self.representation_align = representation_align
        self.representation_depth = representation_depth

        # SPRINT
        self.decoder_depth = decoder_depth
        self.middle_depth = middle_depth
        self.encoder_depth = encoder_depth
        self.use_mask_restore = use_mask_restore
        self.renoise = renoise
        self.residual_type = residual_type
        self.cfg_mask_prob = cfg_mask_prob

        # Conditioning layers
        self.x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Rotary embedding
        self.rel_pos_embed = VisionRotaryEmbedding(
            head_dim=hidden_size//num_heads, theta=rope_theta, custom_freqs=custom_freqs, online_rope=online_rope,
            max_pe_len_h=max_pe_len_h, max_pe_len_w=max_pe_len_w, decouple=decouple, ori_max_pe_len=ori_max_pe_len,
        )

        # AdaLN modulation
        if adaln_type == 'lora':
            self.global_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=adaln_bias)
            )
        else:
            self.global_adaLN_modulation = None

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([TransformerBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type,
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim, **kwargs
        ) for _ in range(self.encoder_depth)])

        # Middle blocks
        self.middle_blocks = nn.ModuleList([TransformerBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type,
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim, **kwargs
        ) for _ in range(self.middle_depth)])

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([TransformerBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type,
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim, **kwargs
        ) for _ in range(self.decoder_depth)])

        # Learnable mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=False)

        # REPA
        if self.representation_align:
            self.projectors = nn.Sequential(
                nn.Linear(hidden_size, projector_hidden_dim),
                nn.SiLU(),
                nn.Linear(projector_hidden_dim, projector_hidden_dim),
                nn.SiLU(),
                nn.Linear(projector_hidden_dim, dim_projection),
            )
        else:
            self.projectors = None

        if self.residual_type == 'concat_linear':
            self.renoise_linear = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        elif self.residual_type == 'baseline':
            self.renoise_linear = None
        else:
            raise ValueError(f"Invalid renoise type: {self.residual_type}")

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm_layer=norm_type, adaln_bias=adaln_bias, adaln_type=adaln_type)

        self.initialize_weights(pretrain_ckpt=pretrain_ckpt, ignore=ignore_keys)
        self.finetune = finetune
        if self.finetune:
            self.finetune_model(freeze=ignore_keys)

    def initialize_weights(self, pretrain_ckpt=None, ignore=None):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.encoder_blocks:
            if self.adaln_type in ['normal', 'lora']:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif self.adaln_type == 'swiglu':
                nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)

        for block in self.middle_blocks:
            if self.adaln_type in ['normal', 'lora']:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif self.adaln_type == 'swiglu':
                nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)

        for block in self.decoder_blocks:
            if self.adaln_type in ['normal', 'lora']:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif self.adaln_type == 'swiglu':
                nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)

        if self.adaln_type == 'lora':
            nn.init.constant_(self.global_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.global_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.adaln_type == 'swiglu':
            nn.init.constant_(self.final_layer.adaLN_modulation.fc2.weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation.fc2.bias, 0)
        else:   # adaln_type in ['normal', 'lora']
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize ignore keys
        keys = list(self.state_dict().keys())
        ignore_keys = []
        if ignore != None:
            for ign in ignore:
                for key in keys:
                    if ign in key:
                        ignore_keys.append(key)
        ignore_keys = list(set(ignore_keys))
        if pretrain_ckpt != None:
            init_from_ckpt(self, pretrain_ckpt, ignore_keys, verbose=True)

    def flatten(self, x):
        x = x.reshape(x.shape[0], -1, self.context_size//self.patch_size, self.patch_size, self.context_size//self.patch_size, self.patch_size)
        x = rearrange(x, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
        x = x.permute(0, 2, 1)  # (b, h, c)
        return x

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        h = w = self.context_size
        p = self.patch_size

        x = rearrange(x, "b (h w) c -> b h w c", h=h//p, w=w//p) # (B, h//2 * w//2, 16) -> (B, h//2, w//2, 16)
        imgs = rearrange(x, "b h w (c p1 p2) -> b c (h p1) (w p2)", p1=p, p2=p) # (B, h//2, w//2, 16) -> (B, h, w, 4)
        return imgs

    def restore_full_sequence(self, x_masked, mask_dict, mask_tokens=None):
        """
        Restore full sequence by adding learnable mask tokens to unselected positions.

        Args:
            x_masked: (B, L_masked, hidden_size) - masked tokens from main model
            mask_dict: Dict containing 'ids_keep', 'ids_restore', 'ids_remove'

        Returns:
            x_full: (B, L_full, pred_hidden_size) - full sequence with mask tokens
        """
        B, L_masked, D = x_masked.shape
        L_full = (self.context_size // self.patch_size) ** 2
        L_remove = L_full - L_masked

        if mask_tokens is None:
            mask_tokens = self.mask_token.repeat(B, L_remove, 1)  # (B, L_remove, pred_hidden_size)
        x_concat = torch.cat([x_masked, mask_tokens], dim=1)  # (B, L_full, pred_hidden_size)

        # Restore original order using ids_restore
        ids_restore = mask_dict['ids_restore'].unsqueeze(-1).repeat(1, 1, D)
        x_full = torch.gather(x_concat, dim=1, index=ids_restore)  # (B, L_full, pred_hidden_size)
        return x_full

    def forward(self,
                x,
                t,
                y,
                mask_dict=None,
                path_drop_guidance=False,
    ):
        """
        Forward pass of SiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        # Path Drop Guidance at inference-only!
        if path_drop_guidance:
            assert not self.training
            return self.forward_pdg(x, t, y)

        block_idx = 0
        zs = None
        x = self.flatten(x)                             # (B, C, H, W) -> (B, N, D)
        x = self.x_embedder(x)                          # (B, N, D) -> (B, N, D')

        grid, _, _ = make_grid_mask_size(x.shape[0], self.context_size // self.patch_size, self.context_size // self.patch_size, x.device)
        freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        # Apply token masking for rope
        if mask_dict is not None:
            freqs_cos_masked = mask_out_pos_token(freqs_cos, mask_dict['ids_keep']).contiguous()
            freqs_sin_masked = mask_out_pos_token(freqs_sin, mask_dict['ids_keep']).contiguous()
        else:
            freqs_cos_masked = freqs_cos
            freqs_sin_masked = freqs_sin

        # Conditioning layers
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        # AdaLN modulation
        if self.global_adaLN_modulation != None:
            global_adaln = self.global_adaLN_modulation(c)
        else:
            global_adaln = 0.0

        B, N, D = x.shape

        # Encoder blocks
        if self.encoder_blocks is not None:
            for encoder_block in self.encoder_blocks:
                if not self.use_checkpoint:
                    x = encoder_block(x, c, freqs_cos, freqs_sin, None, global_adaln)
                else:
                    x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(encoder_block), x, c, freqs_cos, freqs_sin, None, global_adaln)
                block_idx += 1
                if (block_idx+1) == self.representation_depth and self.representation_align:
                    zs = [projector(x.reshape(-1, D)).reshape(B, N, -1) for projector in self.projectors]

        # Dense-shallow path
        x_clone = x.clone()

        # Apply token drop
        if mask_dict is not None:
            x = mask_out_token(x, mask_dict['ids_keep'])

        B, N, D = x.shape

        # Sparse-deep path
        for i, block in enumerate(self.middle_blocks):                   # (B, N, D)
            if not self.use_checkpoint:
                x = block(x, c, freqs_cos_masked, freqs_sin_masked, None, global_adaln)
            else:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, freqs_cos_masked, freqs_sin_masked, None, global_adaln)
            block_idx += 1
            if (block_idx+1) == self.representation_depth and self.representation_align:
                #zs = [projector(x.reshape(-1, D)).reshape(B, N, -1) for projector in self.projectors]
                zs = [self.projectors(x)]

        # Restore full sequence with mask tokens
        if mask_dict is not None and self.use_mask_restore:
            x = self.restore_full_sequence(x, mask_dict)

        B, N, D = x.shape
        if self.training and self.cfg_mask_prob > 0:
            sample_mask = torch.rand(B, device=x.device) < self.cfg_mask_prob  # (B,)
            mask_tokens_expanded = self.mask_token.expand(B, N, D)  # (B, N, D)
            x = torch.where(sample_mask.unsqueeze(1).unsqueeze(2), mask_tokens_expanded, x)

        if self.residual_type == 'concat_linear':
            x = torch.cat([x, x_clone], dim=-1)
            x = self.renoise_linear(x)
        elif self.residual_type == 'baseline':
            x = x
        else:
            raise NotImplementedError(f"Invalid renoise type: {self.residual_type}")

        # Decoder blocks
        for i, dec_block in enumerate(self.decoder_blocks):
            if self.use_mask_restore:
                if not self.use_checkpoint:
                    x = dec_block(x, c, freqs_cos, freqs_sin, None, global_adaln)
                else:
                    x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(dec_block), x, c, freqs_cos, freqs_sin, None, global_adaln)
            else:
                if not self.use_checkpoint:
                    x = dec_block(x, c, freqs_cos_masked, freqs_sin_masked, None, global_adaln)
                else:
                    x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(dec_block), x, c, freqs_cos_masked, freqs_sin_masked, None, global_adaln)

        x = self.final_layer(x, c)  # (B, N_masked, p ** 2 * C_out)
        x = self.unpatchify(x)
        return x, zs

    def forward_pdg(
        self,
        x,
        t,
        y,
    ):
        x = self.flatten(x)                             # (B, C, H, W) -> (B, N, D)
        x = self.x_embedder(x)                          # (B, N, D) -> (B, N, D')

        grid, _, _ = make_grid_mask_size(x.shape[0], self.context_size // self.patch_size, self.context_size // self.patch_size, x.device)
        freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        # AdaLN modulation
        if self.global_adaLN_modulation != None:
            global_adaln = self.global_adaLN_modulation(c)
        else:
            global_adaln = 0.0

        B, N, D = x.shape

        # Encoder blocks
        if self.encoder_blocks is not None:
            for encoder_block in self.encoder_blocks:
                x = encoder_block(x, c, freqs_cos, freqs_sin, None, global_adaln)

        # Dense-shallow path
        x_clone = x.clone()

        x_cond, x_uncond = x.chunk(2)
        freqs_cos_cond, freqs_cos_uncond = freqs_cos.chunk(2)
        freqs_sin_cond, freqs_sin_uncond = freqs_sin.chunk(2)
        c_cond, c_uncond = c.chunk(2)

        B, N, D = x_cond.shape

        for i, block in enumerate(self.middle_blocks):                   # (B, N, D)
            x_cond = block(x_cond, c_cond, freqs_cos_cond, freqs_sin_cond, None, global_adaln)

        x_uncond = self.mask_token.repeat(B, N, 1)
        x = torch.cat([x_cond, x_uncond], dim=0)

        if self.residual_type == 'concat_linear':
            x = torch.cat([x, x_clone], dim=-1)
            x = self.renoise_linear(x)
        else:
            raise ValueError(f"Invalid residual type: {self.residual_type}")

        for i, dec_block in enumerate(self.decoder_blocks):
            x = dec_block(x, c, freqs_cos, freqs_sin, None, global_adaln)

        x = self.final_layer(x, c)  # (B, N_masked, p ** 2 * C_out)
        x = self.unpatchify(x)
        return x, None

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


    def finetune_model(self, freeze):
        for unf in freeze:
            for name, param in self.named_parameters():
                if unf in name: # LN means Layer Norm
                    param.requires_grad = False



#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL(**kwargs):
    return SiT(hidden_size=1152, num_heads=16, **kwargs)

def SiT_L(**kwargs):
    return SiT(hidden_size=1024, num_heads=16, **kwargs)

def SiT_B(**kwargs):
    return SiT(hidden_size=768, num_heads=12, **kwargs)

def SiT_S(**kwargs):
    return SiT(hidden_size=384, num_heads=6, **kwargs)


SiT_models = {
    'XL': SiT_XL,
    'L':  SiT_L,
    'B':  SiT_B,
    'S':  SiT_S
}