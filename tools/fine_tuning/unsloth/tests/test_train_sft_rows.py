from __future__ import annotations

import unittest
from pathlib import Path

from tools.fine_tuning.unsloth.train_sft import build_sft_rows


class FallbackTokenizer:
    pass


class SftRowsTests(unittest.TestCase):
    def test_assistant_only_loss_keeps_conversational_messages_column(self) -> None:
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

        self.assertEqual(["messages"], list(rows[0].keys()))
        self.assertEqual("assistant", rows[0]["messages"][-1]["role"])

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
        rows = build_sft_rows(
            [
                {
                    "messages": [
                        {"role": "user", "content": "status"},
                        {"role": "assistant", "content": {"ok": True}},
                    ]
                }
            ],
            tokenizer=FallbackTokenizer(),
            assistant_only_loss=True,
            path=Path("train_sft.jsonl"),
        )

        self.assertEqual('{"ok": true}', rows[0]["messages"][-1]["content"])

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
