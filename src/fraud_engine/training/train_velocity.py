from __future__ import annotations

import argparse
import json

from fraud_engine.training.train_chargeback import train_velocity as _train_velocity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.training.train_velocity")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="data/velocity_dataset.json")
    parser.add_argument("--version", default="v1.5.0-rc1")
    parser.add_argument("--stage", default="challenger")
    args = parser.parse_args(argv)
    result = _train_velocity(args.dataset, version=args.version, stage=args.stage)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
