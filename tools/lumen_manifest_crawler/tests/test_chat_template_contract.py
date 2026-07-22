from __future__ import annotations

import hashlib

import pytest

from lumen_manifest_crawler.dataset import chat_template_contract as contract_module


class ContractTokenizer:
    def __init__(self, template: str = "controlled-template") -> None:
        self.chat_template = template
        self.calls: list[dict[str, object]] = []
        self.rendered_messages: list[list[dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        copied = [dict(message) for message in messages]
        self.rendered_messages.append(copied)
        last_user = next(
            message for message in reversed(copied) if message.get("role") == "user"
        )
        rendered = (
            "prompt-prefix"
            + str(last_user.get("content") or "")
            + "<|im_end|><|im_start|>assistant\n"
        )
        if kwargs.get("add_generation_prompt"):
            return rendered + contract_module.NON_THINKING_GENERATION_PREFIX
        assistant = messages[-1] if messages else {}
        if assistant.get("role") == "assistant":
            return (
                rendered
                + contract_module.NON_THINKING_GENERATION_PREFIX
                + str(assistant.get("content") or "")
                + "<|im_end|>"
            )
        return rendered


def test_contract_is_self_hashed_and_fail_closed() -> None:
    contract = contract_module.chat_template_contract()
    assert contract_module.verify_chat_template_contract(contract) == contract[
        "contractSHA256"
    ]

    mutated = dict(contract)
    mutated["templateKwargs"] = {"enable_thinking": True}
    with pytest.raises(ValueError, match="drifted"):
        contract_module.verify_chat_template_contract(mutated)


@pytest.mark.parametrize(
    ("content", "expected"),
    (
        ("No structured response is requested.", "absent"),
        (contract_module.STRUCTURED_OUTPUT_INSTRUCTION, "exact_once"),
        (
            contract_module.STRUCTURED_OUTPUT_INSTRUCTION
            + "\n"
            + contract_module.STRUCTURED_OUTPUT_INSTRUCTION,
            "drifted",
        ),
        (
            contract_module.STRUCTURED_OUTPUT_INSTRUCTION.replace(
                "Response format contract:",
                "Response-format contract:",
            ),
            "drifted",
        ),
        ("Response_format_contract: emit JSON.", "drifted"),
    ),
)
def test_structured_output_instruction_status_fails_closed(
    content: str,
    expected: str,
) -> None:
    assert contract_module.structured_output_instruction_status(content) == expected


def test_wrapper_forces_non_thinking_for_every_render() -> None:
    tokenizer = ContractTokenizer()
    contract_module.apply_non_thinking_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert tokenizer.calls == [
        {
            "enable_thinking": False,
            "tokenize": False,
            "add_generation_prompt": True,
        }
    ]
    assert tokenizer.rendered_messages[0][-1]["content"] == "hello\n\n/no_think"

    with pytest.raises(ValueError, match="enable_thinking=false"):
        contract_module.apply_non_thinking_chat_template(
            tokenizer,
            [{"role": "user", "content": "hello"}],
            enable_thinking=True,
        )


def test_runtime_directive_is_final_user_owned_and_idempotent() -> None:
    source = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Explain /no_think as text, then /think"},
        {"role": "assistant", "content": "answer"},
    ]

    controlled = contract_module.canonical_non_thinking_messages(source)
    repeated = contract_module.canonical_non_thinking_messages(controlled)

    assert controlled[-2]["content"].endswith("/think\n\n/no_think")
    assert repeated == controlled
    assert source[-2]["content"].endswith("/think")


def test_loaded_tokenizer_digest_and_prompt_prefix_must_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = ContractTokenizer()
    monkeypatch.setattr(
        contract_module,
        "PINNED_QWEN3_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
    )
    contract = contract_module.chat_template_contract()

    contract_module.verify_chat_template_contract(contract, tokenizer=tokenizer)
    assert len(tokenizer.calls) == 2
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)

    tokenizer.chat_template = "drifted-template"
    with pytest.raises(RuntimeError, match="digest"):
        contract_module.verify_chat_template_contract(contract, tokenizer=tokenizer)
