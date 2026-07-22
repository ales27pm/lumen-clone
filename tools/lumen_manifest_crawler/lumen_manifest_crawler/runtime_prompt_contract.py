from __future__ import annotations

import hashlib


RUNTIME_PROMPT_COMPOSER_POLICY_ID = "lumen.runtime-prompt-composer/1.1.0"
RUNTIME_PROMPT_COMPOSER_POLICY = (
    f"{RUNTIME_PROMPT_COMPOSER_POLICY_ID}|"
    "attested-fleet-prompt-must-byte-match-exact-slot-contract|"
    "attested-runtime-grounding-only|"
    "caller-context-is-a-separate-hashed-component|"
    "effective-prompt-hash-required-for-runtime-qualification"
)
RUNTIME_PROMPT_COMPOSER_POLICY_SHA256 = hashlib.sha256(
    RUNTIME_PROMPT_COMPOSER_POLICY.encode("utf-8")
).hexdigest()

FLEET_SYSTEM_PROMPT_CONTRACT_SCHEMA_VERSION = (
    "lumen.fleet-system-prompt-contract/1.0.0"
)
RUNTIME_GROUNDING_PROMPT_CONTRACT_SCHEMA_VERSION = (
    "lumen.runtime-grounding-prompt-contract/1.0.0"
)
SHIPPED_RUNTIME_QUALIFICATION_SCHEMA_VERSION = (
    "lumen.shipped-runtime-prompt-qualification/1.0.0"
)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
