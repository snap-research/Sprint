import torch
import logging

from torch import nn


logger = logging.getLogger(__name__)

def apply_compile(model: nn.Module, compile_config=None):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for name, module in model.encoder_blocks.named_children():
        module = torch.compile(
            module, backend=compile_config.backend,
        )
        model.encoder_blocks.register_module(name, module)

    for name, module in model.middle_blocks.named_children():
        module = torch.compile(
            module, backend=compile_config.backend,
        )
        model.middle_blocks.register_module(name, module)

    for name, module in model.decoder_blocks.named_children():
        module = torch.compile(
            module, backend=compile_config.backend,
        )
        model.decoder_blocks.register_module(name, module)

    logger.info("Compiling each TransformerBlock with torch.compile")

def apply_compile_wan(model: nn.Module, compile_config=None):
    """
    Apply torch.compile to each WanAttentionBlock, which makes compilation efficient due to
    repeated structure.
    """
    for name, module in model.blocks.named_children():
        module = torch.compile(
            module, backend='inductor', dynamic=False
        )
        model.blocks.register_module(name, module)

    logger.info("Compiling each WanAttentionBlock with torch.compile")

def apply_fsdp(
    model: nn.Module,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    dp_mesh_dim_names = ('dp_shard_cp')
    dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    fully_shard(model, **fsdp_config)