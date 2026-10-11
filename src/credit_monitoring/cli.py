"""Composable local commands with machine-readable summaries."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Academic credit model monitoring")
    parser.add_argument("command", choices=["download", "train", "demo", "verify"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    from credit_monitoring.data import prepare, read_splits
    from credit_monitoring.pipeline import load_config, run_demo, verify
    from credit_monitoring.train import train

    seed = load_config(args.config).get("seed", 42) if args.command != "verify" else 42
    if args.command == "download":
        result = prepare(args.data_dir, args.report_dir, seed)
    elif args.command == "train":
        prepare(args.data_dir, args.report_dir, seed)
        splits = read_splits(args.data_dir)
        result = train(
            splits["reference"], splits["validation"], args.artifact_dir, args.report_dir, seed
        )
    elif args.command == "demo":
        result = run_demo(
            args.data_dir, args.artifact_dir, args.report_dir, args.config, args.tracking_uri
        )
        verify(args.report_dir)
        result = {
            "batch_id": result["batch_id"],
            "report": str(args.report_dir / "index.html"),
            "scenarios": {run["scenario"]: run["status"] for run in result["runs"]},
        }
    else:
        result = verify(args.report_dir)
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
