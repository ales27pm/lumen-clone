#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

SEEDS = {
    "weather": ["should I bring an umbrella", "is it gross outside", "weather here"],
    "maps": ["closest gas station", "show this on map"],
    "outlook": ["read unread emails", "check my inbox"],
    "memory": ["do you remember"],
    "rag": ["search my files"],
    "chat": ["explain this code", "what is recursion"],
    "webSearch": ["search web for swift concurrency"],
    "emailDraft": ["draft an email to sam"],
    "messageDraft": ["send a text to alex"],
    "phoneCall": ["call mom"],
    "contactSearch": ["find bob contact"],
    "calendar": ["create event tomorrow"],
    "reminder": ["remind me at 5"],
    "photos": ["find photos from june"],
    "camera": ["take a photo"],
    "health": ["health summary"],
    "motion": ["am i walking"],
    "files": ["read this file"],
    "trigger": ["create trigger every day"],
    "alarm": ["set alarm for 7"],
    "note": ["save this note"],
    "unknown": ["asdf qwerty"],
}

HARD_NEGATIVES = [
    ("explain calendar APIs in Swift", "chat"),
    ("what is an alarm clock", "chat"),
    ("how does outlook protocol work", "chat"),
]


def load_schema_labels(schema_path: Path) -> set[str]:
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    labels = data.get("properties", {}).get("intent", {}).get("enum", [])
    if not labels:
        raise SystemExit(f"Schema enum is empty: {schema_path}")
    return set(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="intent_dataset.jsonl")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--schema", default=str(Path(__file__).with_name("intents.schema.json")))
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    schema_labels = load_schema_labels(Path(args.schema))
    unknown_labels = sorted(set(SEEDS.keys()) - schema_labels)
    if unknown_labels:
        raise SystemExit(f"Generated labels not in schema enum: {', '.join(unknown_labels)}")

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    for _ in range(args.repeat):
        for intent, examples in SEEDS.items():
            rows.append({"text": rng.choice(examples), "intent": intent})

    for text, intent in HARD_NEGATIVES:
        if intent not in schema_labels:
            raise SystemExit(f"Hard negative label not in schema enum: {intent}")
        rows.append({"text": text, "intent": intent})

    rng.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            text = str(row.get("text", "")).strip()
            intent = str(row.get("intent", "")).strip()
            if not text or not intent:
                raise SystemExit("Each row must include non-empty 'text' and 'intent'.")
            f.write(json.dumps({"text": text, "intent": intent}, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
