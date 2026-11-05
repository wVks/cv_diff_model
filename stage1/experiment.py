import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .runtime import ROOT, digest_tensor, environment, synchronize, write_json
from .model import load_pipeline, sample_ddim

import torch


def fixed_noise(config):
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    return torch.randn(config["num_samples"], 3, 32, 32, generator=generator, dtype=torch.float32)


def timed_sample(pipe, noise, steps, device):
    # CPU->device transfer, postprocessing and file I/O are outside the timed region.
    noise = noise.to(device)
    synchronize(device)
    started = time.perf_counter()
    image, calls = sample_ddim(pipe, noise, steps)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return image, calls, elapsed


def validate_against_pipeline(pipe, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(2, 3, 32, 32, generator=generator)
    ours, calls = sample_ddim(pipe, noise, 4)
    reference = pipe(
        batch_size=2, generator=torch.Generator(device="cpu").manual_seed(seed),
        num_inference_steps=4, eta=0.0, use_clipped_model_output=False, output_type="np",
    ).images
    reference = torch.from_numpy(reference).permute(0, 3, 1, 2)
    error = (ours.cpu() - reference).abs().max().item()
    if error > 2e-5:
        raise AssertionError(f"Baseline differs from DDIMPipeline: max abs error {error}")
    return {"max_absolute_error": error, "steps": 4, "num_images": 2, "actual_unet_calls": calls}


def benchmark(config, device, output_dir):
    pipe = load_pipeline(config, device)
    check = validate_against_pipeline(pipe, config["seed"], device)
    noise = fixed_noise(config)[:config["batch_size"]]
    timed_sample(pipe, noise, 3, device)
    times = []
    benchmark_steps = min(10, config["steps"])
    for repeat in range(config["benchmark_repeats"]):
        _, calls, seconds = timed_sample(pipe, noise, benchmark_steps, device)
        times.append(seconds)
        print(f"Benchmark {repeat + 1}: batch={len(noise)}, steps={calls}, seconds={seconds:.3f}", flush=True)
    estimate = statistics.median(times) * config["steps"] / benchmark_steps * math.ceil(config["num_samples"] / len(noise))
    result = {
        "config": config, "environment": environment(device), "pipeline_check": check,
        "benchmark_steps": benchmark_steps, "batch_size": len(noise),
        "repeat_seconds": times, "median_seconds": statistics.median(times),
        "estimated_generation_seconds": estimate,
        "estimate_excludes": ["model and dataset downloads", "FID", "file I/O"],
    }
    write_json(ROOT / output_dir / "benchmark.json", result)
    print(f"Estimated generation: {estimate / 60:.1f} minutes; FID is additional", flush=True)
    return result


def generate(config, device, output_dir):
    output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    noise = fixed_noise(config)
    manifest = {"config": config, "noise_sha256": digest_tensor(noise), "environment": environment(device)}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output belongs to a different configuration/environment; use a new --output")
    write_json(manifest_path, manifest)
    torch.save(noise, output_dir / "noise.pt")
    pipe = load_pipeline(config, device)
    pipeline_check = validate_against_pipeline(pipe, config["seed"], device)
    timed_sample(pipe, noise[:config["batch_size"]], 3, device)
    batches, durations, calls = [], [], []
    for start in range(0, len(noise), config["batch_size"]):
        path = output_dir / f"batch_{start:05d}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
        else:
            images, nfe, seconds = timed_sample(pipe, noise[start:start + config["batch_size"]], config["steps"], device)
            saved = {"images": (images * 255).round().to(torch.uint8).cpu(), "seconds": seconds, "calls": nfe}
            temporary = path.with_suffix(".tmp")
            torch.save(saved, temporary)
            temporary.replace(path)
        batches.append(saved["images"])
        durations.append(saved["seconds"])
        calls.append(saved["calls"])
        completed = start + len(saved["images"])
        estimate_remaining = statistics.mean(durations) * (math.ceil((len(noise) - completed) / config["batch_size"]))
        print(f"Generation: {completed}/{len(noise)}; batch {saved['seconds']:.2f}s; remaining ~{estimate_remaining / 60:.1f} min", flush=True)
    generated = torch.cat(batches)
    torch.save(generated, output_dir / "generated.pt")
    from .metrics import save_grid
    save_grid(generated, output_dir / "generated_grid.png")
    result = {
        "status": "generated", "date_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "environment": environment(device), "noise_sha256": manifest["noise_sha256"],
        "generated_sha256": digest_tensor(generated), "pipeline_check": pipeline_check,
        "generation": {
            "num_images": len(generated), "nfe_per_image": config["steps"],
            "actual_batched_unet_calls": sum(calls), "sample_model_evaluations": len(generated) * config["steps"],
            "batch_count": len(batches), "batch_seconds": durations,
            "total_seconds": sum(durations), "seconds_per_image": sum(durations) / len(generated),
            "images_per_second": len(generated) / sum(durations),
            "timing_scope": "device transfer excluded; denoising, clone and float [0,1] postprocess included; device synchronized; warmup and disk I/O excluded",
        },
    }
    write_json(output_dir / "results.json", result)
    return result, generated


def evaluate(config, device, output_dir):
    from .metrics import compute_fid
    path = ROOT / output_dir
    result = json.loads((path / "results.json").read_text())
    if result["config"] != config:
        raise ValueError("Evaluation config differs from generation config")
    generated = torch.load(path / "generated.pt", map_location="cpu", weights_only=True)
    if digest_tensor(generated) != result["generated_sha256"]:
        raise ValueError("Generated images do not match the recorded SHA256")
    result["fid"] = compute_fid(config, generated, path, device)
    result["status"] = "complete"
    write_json(path / "results.json", result)
    print(json.dumps(result["fid"], indent=2), flush=True)
    return result
