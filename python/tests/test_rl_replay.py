import numpy as np
import pytest

from gristmill_rl.actions import SampledAction
from gristmill_rl.replay import EpisodeTrace, ReplayBuffer, RootTraceRecord


def sample_action(candidate_index: int) -> SampledAction:
    return SampledAction(
        decision={
            "candidate_index": candidate_index,
            "left_mask": [True],
            "right_mask": [True],
        },
        prior=0.5,
    )


def root_record(state_log_flops: float, candidate_index: int = 0) -> RootTraceRecord:
    return RootTraceRecord(
        state_snapshot={"definitions": []},
        action_space_snapshot={"candidate_templates": []},
        sampled_actions=[sample_action(candidate_index)],
        visit_distribution=np.asarray([1.0], dtype=np.float32),
        state_log_flops=state_log_flops,
        start_from=0,
    )


def multi_action_root_record() -> RootTraceRecord:
    return RootTraceRecord(
        state_snapshot={"definitions": [{"name": "x"}]},
        action_space_snapshot={"candidate_templates": [{"id": 0}, {"id": 1}]},
        sampled_actions=[sample_action(0), sample_action(1)],
        visit_distribution=np.asarray([2.0, 6.0], dtype=np.float32),
        state_log_flops=10.0,
        start_from=0,
    )


def large_count_root_record() -> RootTraceRecord:
    return RootTraceRecord(
        state_snapshot={"definitions": [{"name": "x"}]},
        action_space_snapshot={"candidate_templates": [{"id": 0}, {"id": 1}]},
        sampled_actions=[sample_action(0), sample_action(1)],
        visit_distribution=np.asarray([1e300, 1e300], dtype=np.float64),
        state_log_flops=10.0,
        start_from=0,
    )


def test_episode_trace_completes_value_targets():
    trace = EpisodeTrace()
    trace.append(root_record(10.0, 0))
    trace.append(root_record(8.5, 1))

    items = trace.complete(final_log_flops=7.0)

    assert [item.value_target for item in items] == [3.0, 1.5]
    assert items[0].sampled_actions[0].decision["candidate_index"] == 0
    assert items[1].sampled_actions[0].decision["candidate_index"] == 1


def test_episode_trace_normalizes_multi_action_visit_counts():
    items = EpisodeTrace([multi_action_root_record()]).complete(final_log_flops=7.0)

    np.testing.assert_allclose(items[0].policy_target, [0.25, 0.75])


def test_episode_trace_normalizes_large_finite_visit_counts():
    items = EpisodeTrace([large_count_root_record()]).complete(final_log_flops=7.0)

    assert items[0].policy_target.dtype == np.float32
    assert np.all(np.isfinite(items[0].policy_target))
    np.testing.assert_allclose(items[0].policy_target, [0.5, 0.5])


@pytest.mark.parametrize(
    "visit_distribution",
    [
        np.asarray([], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([-1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.5, 0.5], dtype=np.float32),
    ],
)
def test_episode_trace_rejects_invalid_direct_constructor_records(visit_distribution):
    record = root_record(10.0)
    invalid_record = RootTraceRecord(
        state_snapshot=record.state_snapshot,
        action_space_snapshot=record.action_space_snapshot,
        sampled_actions=record.sampled_actions,
        visit_distribution=visit_distribution,
        state_log_flops=record.state_log_flops,
        start_from=record.start_from,
    )

    with pytest.raises(ValueError):
        EpisodeTrace([invalid_record]).complete(final_log_flops=7.0)


@pytest.mark.parametrize(
    "visit_distribution",
    [
        np.asarray([], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([-1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.5, 0.5], dtype=np.float32),
    ],
)
def test_episode_trace_append_rejects_invalid_records(visit_distribution):
    record = root_record(10.0)
    invalid_record = RootTraceRecord(
        state_snapshot=record.state_snapshot,
        action_space_snapshot=record.action_space_snapshot,
        sampled_actions=record.sampled_actions,
        visit_distribution=visit_distribution,
        state_log_flops=record.state_log_flops,
        start_from=record.start_from,
    )

    with pytest.raises(ValueError):
        EpisodeTrace().append(invalid_record)


def test_episode_trace_rejects_non_vector_visit_distribution():
    record = root_record(10.0)
    invalid_record = RootTraceRecord(
        state_snapshot=record.state_snapshot,
        action_space_snapshot=record.action_space_snapshot,
        sampled_actions=record.sampled_actions,
        visit_distribution=np.asarray([[1.0]], dtype=np.float32),
        state_log_flops=record.state_log_flops,
        start_from=record.start_from,
    )

    with pytest.raises(ValueError):
        EpisodeTrace([invalid_record]).complete(final_log_flops=7.0)


def test_episode_trace_completed_items_do_not_share_record_mutables():
    record = multi_action_root_record()
    items = EpisodeTrace([record]).complete(final_log_flops=7.0)
    item = items[0]

    record.state_snapshot["definitions"].append({"name": "mutated"})
    record.action_space_snapshot["candidate_templates"].append({"id": 2})
    record.sampled_actions.append(sample_action(2))
    record.sampled_actions[0].decision["candidate_index"] = 99
    record.sampled_actions[0].decision["left_mask"].append(False)

    assert item.state_snapshot == {"definitions": [{"name": "x"}]}
    assert item.action_space_snapshot == {"candidate_templates": [{"id": 0}, {"id": 1}]}
    assert [action.decision["candidate_index"] for action in item.sampled_actions] == [0, 1]
    assert item.sampled_actions[0].decision["left_mask"] == [True]


def test_replay_buffer_evicts_oldest_and_samples_without_replacement():
    replay = ReplayBuffer(capacity=2, seed=0)
    first, second, third = (
        root_record(10.0, 0),
        root_record(11.0, 1),
        root_record(12.0, 2),
    )
    for record in [first, second, third]:
        replay.extend(EpisodeTrace([record]).complete(final_log_flops=9.0))

    assert len(replay) == 2
    batch = replay.sample(batch_size=2)

    assert {item.sampled_actions[0].decision["candidate_index"] for item in batch} == {1, 2}


def test_replay_buffer_extend_copies_items():
    replay = ReplayBuffer(capacity=1, seed=0)
    item = EpisodeTrace([multi_action_root_record()]).complete(final_log_flops=7.0)[0]

    replay.extend([item])
    item.state_snapshot["definitions"].append({"name": "mutated"})
    item.action_space_snapshot["candidate_templates"].append({"id": 2})
    item.sampled_actions[0].decision["candidate_index"] = 99
    item.policy_target[0] = 1.0

    stored_item = replay.sample(batch_size=1)[0]

    assert stored_item.state_snapshot == {"definitions": [{"name": "x"}]}
    assert stored_item.action_space_snapshot == {"candidate_templates": [{"id": 0}, {"id": 1}]}
    assert stored_item.sampled_actions[0].decision["candidate_index"] == 0
    np.testing.assert_allclose(stored_item.policy_target, [0.25, 0.75])


def test_replay_buffer_sample_copies_items():
    replay = ReplayBuffer(capacity=1, seed=0)
    replay.extend(EpisodeTrace([multi_action_root_record()]).complete(final_log_flops=7.0))

    sampled_item = replay.sample(batch_size=1)[0]
    sampled_item.state_snapshot["definitions"].append({"name": "mutated"})
    sampled_item.action_space_snapshot["candidate_templates"].append({"id": 2})
    sampled_item.sampled_actions[0].decision["candidate_index"] = 99
    sampled_item.policy_target[0] = 1.0

    next_sample = replay.sample(batch_size=1)[0]

    assert next_sample.state_snapshot == {"definitions": [{"name": "x"}]}
    assert next_sample.action_space_snapshot == {"candidate_templates": [{"id": 0}, {"id": 1}]}
    assert next_sample.sampled_actions[0].decision["candidate_index"] == 0
    np.testing.assert_allclose(next_sample.policy_target, [0.25, 0.75])
