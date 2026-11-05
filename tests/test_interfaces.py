import pytest
import torch
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel

from stage1.model import prediction_to_x0, sample_ddim, vp_to_edm
from stage1.experiment import fixed_noise


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
def test_prediction_recovers_known_clean_image(prediction_type):
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn(2, 3, 8, 8, generator=generator)
    epsilon = torch.randn(clean.shape, generator=generator)
    a = torch.tensor(0.37)
    sample = a.sqrt() * clean + (1 - a).sqrt() * epsilon
    prediction = {"epsilon": epsilon, "sample": clean, "v_prediction": a.sqrt() * epsilon - (1 - a).sqrt() * clean}[prediction_type]
    torch.testing.assert_close(prediction_to_x0(sample, prediction, a, prediction_type), clean)


def test_vp_to_edm_preserves_the_same_noisy_image():
    clean, epsilon = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    a = torch.tensor(0.11)
    sample = a.sqrt() * clean + (1 - a).sqrt() * epsilon
    y, sigma = vp_to_edm(sample, a)
    torch.testing.assert_close(y, clean + sigma * epsilon)
    torch.testing.assert_close(y / (1 + sigma**2).sqrt(), sample)


def test_baseline_matches_reference_pipeline_and_counts_calls():
    torch.manual_seed(1)
    unet = UNet2DModel(sample_size=8, in_channels=3, out_channels=3, layers_per_block=1,
                       block_out_channels=(32,), down_block_types=("DownBlock2D",),
                       up_block_types=("UpBlock2D",))
    pipe = DDIMPipeline(unet, DDIMScheduler(num_train_timesteps=20))
    pipe.set_progress_bar_config(disable=True)
    noise = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(17))
    actual, calls = sample_ddim(pipe, noise, 4)
    expected = pipe(batch_size=2, generator=torch.Generator().manual_seed(17),
                    num_inference_steps=4, eta=0.0, output_type="np").images
    expected = torch.from_numpy(expected).permute(0, 3, 1, 2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert calls == 4
    assert len(unet._forward_hooks) == 0


def test_noise_is_reproducible_and_independent_of_generation_batch_size():
    first = fixed_noise({"seed": 23, "num_samples": 9, "batch_size": 2})
    second = fixed_noise({"seed": 23, "num_samples": 9, "batch_size": 5})
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first, fixed_noise({"seed": 24, "num_samples": 9}))
