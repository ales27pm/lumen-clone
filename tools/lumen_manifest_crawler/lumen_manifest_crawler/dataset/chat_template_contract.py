from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


CHAT_TEMPLATE_CONTRACT_SCHEMA_VERSION = "lumen.qwen3-chat-template-contract/1.1.0"
PINNED_QWEN3_CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
NON_THINKING_GENERATION_PREFIX = "<think>\n\n</think>\n\n"
NON_THINKING_USER_DIRECTIVE = "/no_think"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def chat_template_contract() -> dict[str, Any]:
    payload = {
        "schemaVersion": CHAT_TEMPLATE_CONTRACT_SCHEMA_VERSION,
        "modelFamily": "qwen3",
        "chatTemplateSHA256": PINNED_QWEN3_CHAT_TEMPLATE_SHA256,
        "templateKwargs": {"enable_thinking": False},
        "userDirective": NON_THINKING_USER_DIRECTIVE,
        "userDirectiveSHA256": hashlib.sha256(
            NON_THINKING_USER_DIRECTIVE.encode("utf-8")
        ).hexdigest(),
        "userDirectiveOwnership": "final_user_message",
        "generationPrefixSHA256": hashlib.sha256(
            NON_THINKING_GENERATION_PREFIX.encode("utf-8")
        ).hexdigest(),
        "generationPrefixOwnership": "prompt",
    }
    return {**payload, "contractSHA256": _canonical_sha256(payload)}


def non_thinking_template_kwargs() -> dict[str, bool]:
    return {"enable_thinking": False}


def strip_terminal_non_thinking_directive(value: str) -> str:
    match = re.search(
        r"(?:^|\s)/no_think\s*$",
        value,
        flags=re.IGNORECASE,
    )
    return value[: match.start()].rstrip() if match is not None else value.rstrip()


def canonical_non_thinking_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a runtime-aligned copy with the controlled directive on the final user turn."""

    copied: list[dict[str, Any]] = []
    last_user_index: int | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError("Qwen3 chat-template messages must be mappings")
        item = dict(message)
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Qwen3 chat-template messages require string role and content")
        if role == "user":
            last_user_index = index
        copied.append(item)
    if last_user_index is None:
        raise ValueError("Qwen3 non-thinking contract requires a user message")

    user_content = strip_terminal_non_thinking_directive(
        copied[last_user_index]["content"]
    )
    user_content = (
        f"{user_content}\n\n{NON_THINKING_USER_DIRECTIVE}"
        if user_content
        else NON_THINKING_USER_DIRECTIVE
    )
    copied[last_user_index]["content"] = user_content
    return copied


def apply_non_thinking_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> Any:
    configured = kwargs.pop("enable_thinking", False)
    if configured is not False:
        raise ValueError("Qwen3 chat-template contract requires enable_thinking=false")
    return tokenizer.apply_chat_template(
        canonical_non_thinking_messages(messages),
        enable_thinking=False,
        **kwargs,
    )


def _tokenizer_chat_template(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        getter = getattr(tokenizer, "get_chat_template", None)
        if callable(getter):
            template = getter()
    if not isinstance(template, str) or not template:
        raise RuntimeError("Tokenizer does not expose the controlled Qwen3 chat template")
    return template


def verify_chat_template_contract(
    value: Mapping[str, Any] | Any,
    *,
    tokenizer: Any | None = None,
) -> str:
    expected = chat_template_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("Chat-template contract drifted from the pinned Qwen3 policy")

    if tokenizer is not None:
        template = _tokenizer_chat_template(tokenizer)
        if hashlib.sha256(template.encode("utf-8")).hexdigest() != expected[
            "chatTemplateSHA256"
        ]:
            raise RuntimeError("Loaded tokenizer chat template drifted from the pinned digest")

        probe_prompt = [
            {"role": "system", "content": "contract-system"},
            {"role": "user", "content": "contract-user"},
        ]
        rendered_prompt = apply_non_thinking_chat_template(
            tokenizer,
            probe_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered_full = apply_non_thinking_chat_template(
            tokenizer,
            [
                *probe_prompt,
                {"role": "assistant", "content": "contract-answer"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        if (
            not isinstance(rendered_prompt, str)
            or not isinstance(rendered_full, str)
            or "contract-user\n\n/no_think<|im_end|>" not in rendered_prompt
            or not rendered_prompt.endswith(NON_THINKING_GENERATION_PREFIX)
            or not rendered_full.startswith(rendered_prompt)
            or not rendered_full[len(rendered_prompt) :].startswith("contract-answer")
        ):
            raise RuntimeError(
                "Qwen3 prompt and completed-conversation prefixes are not identical"
            )

    return str(expected["contractSHA256"])
