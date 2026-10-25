"""Malformed training output must return validation failures, not abort a batch."""

import json
import sys

import pytest

from lumen_manifest_crawler import validators
from test_public_training_output_contract import (
    ROUTE_SYSTEM,
    VALID_ROUTE,
    base,
    failures_for,
)


@pytest.fixture(params=["long_integer", "deep_nesting"])
def parser_limit_literal(request):
    if request.param == "long_integer":
        limit = sys.get_int_max_str_digits()
        if limit == 0:
            pytest.skip("This interpreter explicitly disables the integer conversion limit")
        return "9" * (limit + 1)
    # A roughly 20 KiB input reaches the Python 3.12 decoder's own depth guard.
    # Do not change either interpreter limit to make this case pass or fail.
    return "[" * 10_000 + "0" + "]" * 10_000


def malformed_output(agent, literal):
    if agent == "executor":
        return '{"action":{"tool":"fixture.numeric","args":{"value":' + literal + "}}}"
    return VALID_ROUTE[:-1] + ',"strictProbe":' + literal + "}"


def add_preference(datasets, agent, chosen):
    datasets[agent].train_dpo.append({
        "prompt": [
            {"role": "system", "content": ROUTE_SYSTEM if agent == "cortex" else "Return a native output."},
            {"role": "user", "content": "Use the synthetic fixture."},
        ],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": "synthetic rejected alternative"},
    })


@pytest.mark.parametrize("agent", ["executor", "cortex"])
@pytest.mark.parametrize("split", ["train_sft", "val_sft"])
def test_public_sft_reports_parser_limits_as_invalid_json(agent, split, parser_limit_literal):
    datasets = base()
    getattr(datasets[agent], split)[0]["messages"][2]["content"] = malformed_output(agent, parser_limit_literal)
    failures = failures_for(datasets)
    assert any(failure.code == f"{agent}_non_json_output" for failure in failures), failures


@pytest.mark.parametrize("agent", ["executor", "cortex"])
def test_public_chosen_dpo_reports_parser_limits_as_invalid_json(agent, parser_limit_literal):
    datasets = base()
    add_preference(datasets, agent, malformed_output(agent, parser_limit_literal))
    failures = failures_for(datasets)
    prefix = "cortex_dpo" if agent == "cortex" else "executor"
    assert any(failure.code == f"{prefix}_non_json_output" for failure in failures), failures


@pytest.mark.parametrize("surface", ["train_sft", "val_sft", "chosen_dpo"])
@pytest.mark.parametrize("status", [[], {}])
def test_public_cortex_rejects_container_status_without_hashing_it(surface, status):
    datasets = base()
    route = json.loads(VALID_ROUTE)
    route["status"] = status
    output = json.dumps(route)
    if surface == "chosen_dpo":
        add_preference(datasets, "cortex", output)
        prefix = "cortex_dpo"
    else:
        getattr(datasets["cortex"], surface)[0]["messages"][2]["content"] = output
        prefix = "cortex"
    failures = failures_for(datasets)
    assert any(failure.code == f"{prefix}_route_contract_invalid" for failure in failures), failures


def test_canonicalization_keeps_unparseable_output_for_strict_validation(parser_limit_literal):
    output = malformed_output("executor", parser_limit_literal)
    assert validators._canonical_model_output(output) == output


def test_required_argument_helpers_do_not_rethrow_parser_limits(parser_limit_literal):
    output = malformed_output("executor", parser_limit_literal)
    assert validators._should_enforce_required_args(output) is True
    assert isinstance(validators._assistant_mentions_required_args(output, {"value"}), bool)
