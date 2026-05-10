import json
import subprocess
import sys

import numpy as np

from gristmill_rl.features import FeatureConfig
from gristmill_rl.model import PolicyValueModel
from gristmill_rl.search import SearchNode
from gristmill_rl.train import _proposal_for_node

from .rl_fixtures import actionable_json
from .rl_fixtures import actionable_comp


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
