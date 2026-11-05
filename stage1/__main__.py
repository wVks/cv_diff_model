import argparse
from .runtime import ROOT, read_config, select_device


def main():
    parser = argparse.ArgumentParser(description="Stage 1: deterministic DDIM baseline, no training")
    parser.add_argument("command", choices=["benchmark", "generate", "evaluate", "run"])
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "baseline.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output", type=str, default="artifacts/baseline")
    args = parser.parse_args()
    config = read_config(args.config)
    device = select_device(args.device)
    print(f"Device: {device}; no model training", flush=True)
    from .experiment import benchmark, generate, evaluate
    if args.command == "benchmark":
        benchmark(config, device, args.output)
    if args.command in ("generate", "run"):
        generate(config, device, args.output)
    if args.command in ("evaluate", "run"):
        evaluate(config, device, args.output)


if __name__ == "__main__":
    main()
