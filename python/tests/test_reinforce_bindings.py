import pytest

from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from tests.test_bindings import actionable_json, first_padded_choice


def test_rewrite_state_row_log_total_flops_updates_after_row_apply():
    comp = TensorComputation.from_json_string(actionable_json())
    row = RewriteStateRow.from_states([RewriteState.from_computation(comp)])
    before = row.log_total_flops()[0]
    spaces = row.query_action_spaces_for_row([0], [True])
    choice = first_padded_choice(spaces.snapshots()[0])
    validated = row.validate_actions_for_row(spaces, [choice], [True])

    applied = row.apply_validated_actions_for_row(validated)
    after = row.log_total_flops()[0]

    assert applied == [True]
    assert after != pytest.approx(before)
