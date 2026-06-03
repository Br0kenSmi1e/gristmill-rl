import pytest

from transformer_policy.types import PolicySample, Stage1Attempt, T, Token


def test_token_factory_sorts_payload_for_stable_equality():
    left = T("FACTOR", tensor=3, position=1)
    right = Token(kind="FACTOR", payload=(("position", 1), ("tensor", 3)))

    assert left == right
    assert left.payload_dict() == {"position": 1, "tensor": 3}


def test_token_rejects_empty_kind():
    with pytest.raises(ValueError, match="token kind must not be empty"):
        T("")


def test_token_rejects_unsupported_payload_value():
    with pytest.raises(TypeError, match="unsupported token payload"):
        T("FACTOR", tensor=None)


def test_policy_sample_requires_consistent_terminal_shape():
    stopped = PolicySample(stopped=True, log_prob=0.25)

    assert stopped.def_index is None
    assert stopped.action_space is None
    assert stopped.decision is None
    assert stopped.def_attempts == ()
    assert stopped.decision_tokens == ()


def test_policy_sample_rejects_stopped_with_decision():
    with pytest.raises(ValueError, match="stopped sample must not contain a decision"):
        PolicySample(
            stopped=True,
            def_index=0,
            action_space=object(),
            decision={"candidate_index": 0, "left_mask": [], "right_mask": []},
            log_prob=0.0,
        )


def test_policy_sample_rejects_rewrite_without_decision():
    with pytest.raises(ValueError, match="rewrite sample requires def_index"):
        PolicySample(stopped=False, log_prob=0.0)


def test_stage1_attempt_records_acceptance_and_log_prob():
    attempt = Stage1Attempt(def_index=2, log_prob=-0.5, accepted=False)

    assert attempt.def_index == 2
    assert attempt.log_prob == -0.5
    assert not attempt.accepted


def test_package_exports_policy_types():
    import transformer_policy

    assert transformer_policy.__all__ == (
        "Token",
        "T",
        "Stage1Attempt",
        "PolicySample",
        "TransformerPolicy",
    )
