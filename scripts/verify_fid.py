"""Independent float64 PSD calculation to verify the library's matrix square root."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage1.runtime import write_json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/baseline")
    args = parser.parse_args()
    path = ROOT / args.output
    results = json.loads((path / "results.json").read_text())
    real = np.load(path / "features_real.npy").astype(np.float64)
    generated = np.load(path / "features_generated.npy").astype(np.float64)
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("Nonfinite Inception features")
    mean_real, mean_generated = real.mean(0), generated.mean(0)
    covariance_real, covariance_generated = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
    values, vectors = np.linalg.eigh(covariance_real)
    tolerance = 1e-8
    if values.min() < -tolerance:
        raise ValueError("Real covariance is not positive semidefinite")
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    sandwich = root @ covariance_generated @ root
    eigenvalues = np.linalg.eigvalsh((sandwich + sandwich.T) / 2)
    if eigenvalues.min() < -tolerance:
        raise ValueError("Covariance sandwich is not positive semidefinite")
    independent = float(np.sum((mean_real - mean_generated)**2) + np.trace(covariance_real) + np.trace(covariance_generated) - 2 * np.sqrt(np.maximum(eigenvalues, 0)).sum())
    library = results["fid"]["frechet_inception_distance"]
    error = abs(independent - library)
    report = {"library_fid": library, "independent_psd_fid": independent,
              "absolute_difference": error, "tolerance": 1e-4,
              "features_finite": True, "min_covariance_real_eigenvalue": float(values.min()),
              "min_sandwich_eigenvalue": float(eigenvalues.min()), "passed": error < 1e-4}
    write_json(path / "fid_verification.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("FID matrix square root verification failed")


if __name__ == "__main__":
    main()
