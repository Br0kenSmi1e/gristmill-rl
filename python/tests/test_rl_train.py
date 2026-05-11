import json
import subprocess
import sys

import numpy as np

from gristmill_rl.checkpoint import load_checkpoint
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel
from gristmill_rl.rollout import _proposal_for_node
from gristmill_rl.search import SearchNode

from .rl_fixtures import actionable_comp
from .rl_fixtures import actionable_json
from .rl_fixtures import actionable_space


class CountingComp:
    def __init__(self, inner):
        self.inner = inner
        self.next_action_space_calls = 0

    def next_action_space(self, start_from):
        self.next_action_space_calls += 1
        return self.inner.next_action_space(start_from)

    def clone(self):
        return self.inner.clone()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _checkpoint_output_vector(path):
    loaded = load_checkpoint(path)
    comp, space = actionable_space()
    features = extract_features(
        comp_snapshot=comp.snapshot(),
        action_space_snapshot=space.snapshot(),
        start_from=0,
        log_total_flops=comp.log_total_flops(),
        config=loaded.feature_config,
    )
    outputs = loaded.model(features)
    return np.concatenate(
        [
            np.ravel(np.asarray(outputs.candidate_logits)),
            np.ravel(np.asarray(outputs.left_logits)),
            np.ravel(np.asarray(outputs.right_logits)),
            np.asarray([outputs.value], dtype=np.float32),
        ]
    )


def test_train_proposal_reuses_search_node_action_space():
    comp = CountingComp(actionable_comp())
    node = SearchNode(comp=comp, start_from=0)
    model = PolicyValueModel(rng_seed=0)

    node.expand(
        proposal_fn=_proposal_for_node(
            node,
            model=model,
            feature_config=FeatureConfig(),
            rng=np.random.default_rng(0),
            actions_per_node=1,
            sample_attempts=4,
        )
    )

    assert comp.next_action_space_calls == 1
    assert node.action_space is not None
    assert len(node.sampled_actions) == 1


def test_train_cli_completes_tiny_run(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["episodes"] == 1
    assert metrics["replay_size"] >= 1
    assert metrics["last_total_loss"] > 0.0
    assert metrics["params_changed"]


def test_train_cli_writes_checkpoint(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--seed",
            "0",
            "--hidden-dim",
            "16",
            "--checkpoint-out",
            str(checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["checkpoint_in"] is None
    assert metrics["checkpoint_out"] == str(checkpoint_path)
    assert checkpoint_path.exists()
    assert (checkpoint_path / "metadata.json").exists()
    assert (checkpoint_path / "state").exists()
    assert metrics["replay_size"] >= 1


def test_train_cli_loads_checkpoint_and_continues_training(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_in_path = tmp_path / "checkpoint-in"
    checkpoint_out_path = tmp_path / "checkpoint-out"
    fresh_checkpoint_path = tmp_path / "fresh-control"
    input_path.write_text(actionable_json())

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--seed",
            "0",
            "--hidden-dim",
            "16",
            "--checkpoint-out",
            str(checkpoint_in_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--seed",
            "0",
            "--hidden-dim",
            "16",
            "--checkpoint-in",
            str(checkpoint_in_path),
            "--checkpoint-out",
            str(checkpoint_out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["checkpoint_in"] == str(checkpoint_in_path)
    assert metrics["checkpoint_out"] == str(checkpoint_out_path)
    assert checkpoint_in_path.exists()
    assert checkpoint_out_path.exists()
    assert metrics["replay_size"] >= 1
    assert metrics["params_changed"]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--seed",
            "0",
            "--hidden-dim",
            "16",
            "--checkpoint-out",
            str(fresh_checkpoint_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    input_output = _checkpoint_output_vector(checkpoint_in_path)
    resumed_output = _checkpoint_output_vector(checkpoint_out_path)
    fresh_output = _checkpoint_output_vector(fresh_checkpoint_path)
    assert not np.allclose(resumed_output, input_output)
    assert not np.allclose(resumed_output, fresh_output)
