import tools.check_release_hardening as release_hardening


def _states_for_directive(directive: str) -> list[bool]:
    return release_hardening.debug_stack_for_lines(
        [
            directive,
            "UnavailableGGUFNativeBridge()",
            "#else",
            "UnavailableGGUFNativeBridge()",
            "#endif",
        ]
    )


def test_parenthesized_debug_condition_is_debug_only():
    assert _states_for_directive("#if (DEBUG)") == [True, True, False, False, False]


def test_parenthesized_negated_debug_condition_is_release_branch():
    assert _states_for_directive("#if !(DEBUG)") == [False, False, True, True, False]


def test_spaced_parenthesized_negated_debug_condition_is_release_branch():
    assert _states_for_directive("#if ! ( DEBUG )") == [False, False, True, True, False]
