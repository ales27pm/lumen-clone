#!/usr/bin/env python3
"""Train a lightweight intent classifier."""
import argparse
import json
import pickle
from pathlib import Path


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise SystemExit(f"Dataset file not found: {path}")
    X: list[str] = []
    y: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            intent = str(row.get("intent", "")).strip()
            if not text or not intent:
                raise SystemExit(f"Invalid row {idx}: expected non-empty 'text' and 'intent'.")
            X.append(text)
            y.append(intent)
    if not X:
        raise SystemExit("Dataset has no usable rows.")
    if len(set(y)) < 2:
        raise SystemExit("Dataset must contain at least 2 unique intent labels.")
    return X, y


def build_model(seed: int):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except Exception:
        raise SystemExit(
            "Missing optional dependency scikit-learn.\n"
            "Install with:\n"
            "python -m pip install scikit-learn"
        )

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1200, random_state=seed)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    X, y = load_dataset(Path(args.dataset))
    model = build_model(int(args.seed))
    model.fit(X, y)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(model, f)

    print(f"saved {model_out}")
    print(f"rows={len(X)} labels={len(set(y))}")


if __name__ == "__main__":
    main()
