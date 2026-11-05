import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import torch


def read_config(path):
    config = json.loads(Path(path).read_text())
    for key in ("num_samples", "num_real", "steps", "batch_size", "feature_batch_size", "benchmark_repeats"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["steps"] > 1000 or config["num_real"] > 10000:
        raise ValueError("steps <= 1000 and num_real <= 10000 are required")
    if config["eta"] != 0 or config["prediction_type"] != "epsilon":
        raise ValueError("This baseline uses deterministic DDIM and epsilon prediction")
    return config


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu or a CUDA runtime")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(requested)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def environment(device):
    packages = ["torch", "torchvision", "diffusers", "huggingface-hub", "numpy", "scipy", "torch-fidelity"]
    return {
        "os": platform.system(), "os_release": platform.release(),
        "machine": platform.machine(), "python": platform.python_version(),
        "device": str(device), "dtype": "float32",
        "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else "Apple GPU (MPS)" if device.type == "mps" else platform.processor(),
        "versions": {p: importlib.metadata.version(p) for p in packages},
    }
