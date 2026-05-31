#!/usr/bin/env python3
"""Export sklearn model to CoreML."""
import argparse
import pickle
from pathlib import Path

try:
    import coremltools as ct
except Exception:
    raise SystemExit(
        "Missing optional dependency coremltools.\n"
        "Install with:\n"
        "python -m pip install coremltools scikit-learn"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Input model path does not exist: {model_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("rb") as f:
        model = pickle.load(f)

    mlmodel = ct.converters.sklearn.convert(
        model,
        input_features=["text"],
        output_feature_names=["classLabel"],
    )
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
