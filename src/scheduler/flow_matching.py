import torch
import numpy as np
import torch.nn.functional as F
import torch.cuda.amp as amp

from src.utils.utils import mask_out_token, prepare_mask_and_input


#################################################################################
#                              Flow Matching Functions                         #
#################################################################################

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score

def compute_diffusion(t_cur):
    return 2 * t_cur

class FlowMatching:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            accelerator=None,
            **kwargs,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_loss(self, model, images, model_kwargs=None, zs=None,
                mask_config=None):
        if model_kwargs == None:
            model_kwargs = {}

        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], ))
            while time_input.ndim < images.ndim:
                time_input = time_input.unsqueeze(-1)
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0],))
            while rnd_normal.ndim < images.ndim:
                rnd_normal = rnd_normal.unsqueeze(-1)
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        # Get config for masking
        if mask_config.mask_ratio > 0:
            mask_dict = prepare_mask_and_input(images=images, mask_type=mask_config.mask_type, mask_ratio=mask_config.mask_ratio)
        else:
            mask_dict = None

        with self.accelerator.autocast():
            model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs, mask_dict=mask_dict)

        denoising_loss = (model_output - model_target) ** 2
        denoising_loss = mean_flat(denoising_loss).mean()

        # projection loss
        if self.encoders is not None and zs_tilde is not None:
            if zs[0].shape[1] != zs_tilde[0].shape[1]:
                zs = [mask_out_token(z, mask_dict['ids_keep']) for z in zs]

            # Vectorized normalization
            z_norm = F.normalize(zs[0], dim=-1)
            z_tilde_norm = F.normalize(zs_tilde[0], dim=-1)

            # (B, N, D) * (B, N, D) -> sum over D -> (B, N)
            loss_per_token = -(z_norm * z_tilde_norm).sum(dim=-1)

            # Mean over N and B dimensions
            proj_loss = loss_per_token.mean()
        else:
            proj_loss = torch.tensor(0.0, device=images.device)

        return denoising_loss, proj_loss

    @torch.no_grad()
    def sample_ode(
        self,
        model,
        latents,
        y,
        num_steps=20,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_drop_guidance=False,
    ):
        # setup conditioning
        if cfg_scale > 1.0:
            y_null = torch.tensor([1000] * y.size(0), device=y.device)
        _dtype = latents.dtype
        t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
        x_next = latents.to(torch.float64)
        device = x_next.device

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            kwargs = dict(y=y_cur, path_drop_guidance=path_drop_guidance)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs,
                )[0].to(torch.float64)
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            x_next = x_cur + (t_next - t_cur) * d_cur

        return x_next

    @torch.no_grad()
    def sample_sde(
        self,
        model,
        latents,
        y,
        num_steps=20,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_drop_guidance=False,
    ):
        # setup conditioning
        if cfg_scale > 1.0:
            y_null = torch.tensor([1000] * y.size(0), device=y.device)
        _dtype = latents.dtype
        t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
        t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
        x_next = latents.to(torch.float64)
        device = x_next.device

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            kwargs = dict(y=y_cur, path_drop_guidance=path_drop_guidance)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=self.path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

        # last step
        t_cur, t_next = t_steps[-2], t_steps[-1]
        dt = t_next - t_cur
        x_cur = x_next
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y
        kwargs = dict(y=y_cur, path_drop_guidance=path_drop_guidance)
        time_input = torch.ones(model_input.size(0)).to(
            device=device, dtype=torch.float64
            ) * t_cur

        # compute drift
        v_cur = model(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            )[0].to(torch.float64)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=self.path_type)
        diffusion = compute_diffusion(t_cur)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        mean_x = x_cur + dt * d_cur
        return mean_x