"""Standard FID: torch-fidelity's TF-compatible Inception, 2048 features."""
from .runtime import ROOT, digest_tensor, write_json

import json
import hashlib
import io
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric


def real_images(config):
    revision = "0b2714987fa478483af9968de7c934580d0bb9a2"
    filename = "plain_text/test-00000-of-00001.parquet"
    path = hf_hub_download("uoft-cs/cifar10", filename, repo_type="dataset", revision=revision,
                           cache_dir=ROOT / ".cache" / "datasets")
    records = pq.read_table(path, columns=["img", "label"]).to_pylist()
    if len(records) != 10000 or np.bincount([r["label"] for r in records], minlength=10).tolist() != [1000] * 10:
        raise ValueError("Unexpected CIFAR-10 test split")
    indices = np.random.default_rng(config["real_seed"]).permutation(len(records))[:config["num_real"]]
    pixels = []
    for i in indices:
        with Image.open(io.BytesIO(records[i]["img"]["bytes"])) as image:
            if image.size != (32, 32):
                raise ValueError("Unexpected image size in CIFAR-10")
            pixels.append(np.asarray(image.convert("RGB")))
    images = torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).contiguous()
    source = {"repository": "uoft-cs/cifar10", "revision": revision, "file": filename,
              "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    return images, indices.tolist(), source


@torch.inference_mode()
def extract_features(extractor, images, batch_size, device):
    result = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        result.append(extractor(batch)[0].cpu())
        if start == 0 or (start // batch_size + 1) % 20 == 0:
            print(f"FID features: {min(start + batch_size, len(images))}/{len(images)}", flush=True)
    return torch.cat(result)


def compute_fid(config, generated, output_dir, device):
    output_dir = ROOT / output_dir
    real, indices, source = real_images(config)
    write_json(output_dir / "real_indices.json", indices)
    save_grid(real, output_dir / "real_grid.png")
    started = time.perf_counter()
    extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
    hashes = {"real": digest_tensor(real), "generated": digest_tensor(generated)}
    features = {}
    for name, images in (("real", real), ("generated", generated)):
        path = output_dir / f"features_{name}.npy"
        meta = path.with_suffix(".json")
        expected = {"image_sha256": hashes[name], "extractor": "torch-fidelity-0.3.0-inception-v3-compat-2048", "device": str(device)}
        if path.exists() and meta.exists() and json.loads(meta.read_text()) == expected:
            features[name] = torch.from_numpy(np.load(path))
        else:
            print(f"Extracting {name} features on {device}", flush=True)
            features[name] = extract_features(extractor, images, config["feature_batch_size"], device)
            np.save(path, features[name].numpy())
            write_json(meta, expected)
    print("Computing FID covariance square root (CPU float64)", flush=True)
    statistics = {k: fid_features_to_statistics(v.double()) for k, v in features.items()}
    value = fid_statistics_to_metric(statistics["real"], statistics["generated"], verbose=False)
    return {
        **value, "num_real": len(real), "num_generated": len(generated),
        "real_split": "cifar10-test", "real_seed": config["real_seed"],
        "data_source": source,
        "feature_dimension": 2048, "extractor": "torch-fidelity/inception-v3-compat",
        "feature_device": str(device), "input_dtype": "uint8", "input_range": [0, 255],
        "statistics_dtype": "float64",
        "input_sha256": hashes, "evaluation_seconds": time.perf_counter() - started,
    }


def save_grid(images, path, count=64):
    images = images[:count].cpu().permute(0, 2, 3, 1).numpy()
    columns = min(8, len(images))
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 32, rows * 32), "white")
    for i, image in enumerate(images):
        canvas.paste(Image.fromarray(image), ((i % columns) * 32, (i // columns) * 32))
    canvas.resize((canvas.width * 3, canvas.height * 3), Image.Resampling.NEAREST).save(path)
