"""Model interfaces; numerical integration is delegated to Diffusers in stage 1."""
from .runtime import ROOT

import torch
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel


def prediction_to_x0(sample, prediction, alpha_bar, prediction_type="epsilon"):
    """Convert a VP prediction at x_t = sqrt(A)*x0 + sqrt(1-A)*epsilon."""
    alpha = torch.as_tensor(alpha_bar, dtype=sample.dtype, device=sample.device).sqrt()
    sigma = (1 - torch.as_tensor(alpha_bar, dtype=sample.dtype, device=sample.device)).sqrt()
    if prediction_type == "epsilon":
        return (sample - sigma * prediction) / alpha
    if prediction_type == "sample":
        return prediction
    if prediction_type == "v_prediction":
        return alpha * sample - sigma * prediction
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def vp_to_edm(sample, alpha_bar):
    """Return y=x_t/alpha and sigma_EDM=sigma_VP/alpha; this is not an EDM model."""
    a = torch.as_tensor(alpha_bar, dtype=sample.dtype, device=sample.device)
    return sample / a.sqrt(), ((1 - a) / a).sqrt()


def load_pipeline(config, device):
    unet = UNet2DModel.from_pretrained(
        config["model_id"], revision=config["revision"],
        cache_dir=ROOT / ".cache" / "huggingface",
        torch_dtype=torch.float32, use_safetensors=True, low_cpu_mem_usage=False,
    )
    scheduler = DDIMScheduler.from_pretrained(
        config["model_id"], revision=config["revision"],
        cache_dir=ROOT / ".cache" / "huggingface",
        prediction_type=config["prediction_type"],
        timestep_spacing=config["timestep_spacing"],
        clip_sample=config["clip_sample"],
        set_alpha_to_one=config["set_alpha_to_one"],
        steps_offset=config["steps_offset"],
    )
    pipe = DDIMPipeline(unet=unet, scheduler=scheduler)
    pipe.unet.requires_grad_(False).eval()
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.inference_mode()
def sample_ddim(pipe, noise, steps):
    """Use an explicitly stored x_T and count real UNet calls with a hook."""
    calls = 0

    def count_call(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = pipe.unet.register_forward_hook(count_call)
    try:
        image = noise.to(device=pipe.device, dtype=pipe.unet.dtype).clone()
        pipe.scheduler.set_timesteps(steps)
        for t in pipe.scheduler.timesteps:
            prediction = pipe.unet(image, t).sample
            image = pipe.scheduler.step(
                prediction, t, image, eta=0.0, use_clipped_model_output=False,
            ).prev_sample
        return (image / 2 + 0.5).clamp(0, 1), calls
    finally:
        handle.remove()
