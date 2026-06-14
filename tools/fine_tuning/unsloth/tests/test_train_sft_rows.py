from __future__ import annotations

import unittest
from pathlib import Path

from tools.fine_tuning.unsloth.train_sft import build_sft_rows


class FallbackTokenizer:
    def __init__(self) -> None:
        self.last_messages = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
    ):
        del add_generation_prompt, return_assistant_tokens_mask
        self.last_messages = messages
        rendered = []
        masks = []
        token_id = 1
        for message in messages:
            for _ in f"{message['role']}: {message['content']}".split():
                rendered.append(token_id)
                masks.append(1 if message["role"] == "assistant" else 0)
                token_id += 1
        if tokenize and return_dict:
            return {
                "input_ids": rendered,
                "attention_mask": [1] * len(rendered),
                "assistant_masks": masks,
            }
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class SftRowsTests(unittest.TestCase):
    def test_assistant_only_loss_pretokenizes_with_masked_labels(self) -> None:
        rows = build_sft_rows(
            [
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "done"},
                    ]
                }
            ],
            tokenizer=FallbackTokenizer(),
            assistant_only_loss=True,
            path=Path("train_sft.jsonl"),
        )

        self.assertEqual(["input_ids", "attention_mask", "labels"], list(rows[0].keys()))
        self.assertEqual(len(rows[0]["input_ids"]), len(rows[0]["attention_mask"]))
        self.assertTrue(all(mask == 1 for mask in rows[0]["attention_mask"]))
        self.assertIn(-100, rows[0]["labels"])
        self.assertEqual(rows[0]["input_ids"][-1], rows[0]["labels"][-1])

    def test_non_assistant_only_loss_renders_text_column(self) -> None:
        rows = build_sft_rows(
            [
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "done"},
                    ]
                }
            ],
            tokenizer=FallbackTokenizer(),
            assistant_only_loss=False,
            path=Path("train_sft.jsonl"),
        )

        self.assertEqual(["text"], list(rows[0].keys()))
        self.assertIn("assistant: done", rows[0]["text"])

    def test_non_string_content_is_normalized_for_chat_templates(self) -> None:
        tokenizer = FallbackTokenizer()
        rows = build_sft_rows(
            [
                {
                    "messages": [
                        {"role": "user", "content": "status"},
                        {"role": "assistant", "content": {"ok": True}},
                    ]
                }
            ],
            tokenizer=tokenizer,
            assistant_only_loss=True,
            path=Path("train_sft.jsonl"),
        )

        self.assertEqual('{"ok": true}', tokenizer.last_messages[-1]["content"])
        self.assertEqual(["input_ids", "attention_mask", "labels"], list(rows[0].keys()))

    def test_missing_assistant_message_fails_before_trainer_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "assistant message"):
            build_sft_rows(
                [{"messages": [{"role": "user", "content": "hello"}]}],
                tokenizer=FallbackTokenizer(),
                assistant_only_loss=True,
                path=Path("train_sft.jsonl"),
            )


if __name__ == "__main__":
    unittest.main()
