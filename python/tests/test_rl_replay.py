import numpy as np

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


def test_episode_trace_completes_value_targets():
    trace = EpisodeTrace()
    trace.append(root_record(10.0, 0))
    trace.append(root_record(8.5, 1))

    items = trace.complete(final_log_flops=7.0)

    assert [item.value_target for item in items] == [3.0, 1.5]
    assert items[0].sampled_actions[0].decision["candidate_index"] == 0
    assert items[1].sampled_actions[0].decision["candidate_index"] == 1


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
