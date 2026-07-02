import json

import pytest

from gristmill_symbolics.cli.plot_training_curves import (
    GRISTMILL_OPTIMIZED_LOG_FLOPS,
    _plot_final_flops,
    load_metrics,
    main,
    objective_errorbar,
)


def _metric(update: int, *, reward_std: float, batch_size: int = 4):
    return {
        "update_index": update,
        "batch_size": batch_size,
        "reward_mean": 0.0,
        "reward_std": reward_std,
        "objective_loss_mean": 1.0 + update,
        "surrogate_loss": 0.5,
        "final_flops_best": 10.0 - update,
        "params_changed": True,
    }


def test_load_metrics_reads_nonempty_jsonl_lines(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_metric(0, reward_std=2.0)),
                "",
                json.dumps(_metric(1, reward_std=4.0)),
            ]
        )
    )

    metrics = load_metrics(path)

    assert [item["update_index"] for item in metrics] == [0, 1]


def test_objective_errorbar_uses_reward_standard_error():
    metrics = [
        _metric(0, reward_std=2.0, batch_size=4),
        _metric(1, reward_std=9.0, batch_size=9),
    ]

    assert objective_errorbar(metrics).tolist() == pytest.approx([1.0, 3.0])


def test_plot_training_curves_cli_writes_png(tmp_path):
    input_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "curves.png"
    input_path.write_text(
        "\n".join(
            json.dumps(_metric(update, reward_std=2.0))
            for update in range(3)
        )
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--title",
            "smoke",
        ]
    )

    assert exit_code == 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_final_flops_plot_includes_gristmill_optimized_baseline():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    try:
        _plot_final_flops(axis, [0, 1], [60.0, 55.0])
        y_values = [line.get_ydata()[0] for line in axis.lines]
    finally:
        plt.close(figure)

    assert any(
        value == pytest.approx(GRISTMILL_OPTIMIZED_LOG_FLOPS)
        for value in y_values
    )


def test_plot_training_curves_rejects_non_positive_batch_size():
    metrics = [_metric(0, reward_std=1.0, batch_size=0)]

    with pytest.raises(ValueError, match="batch_size"):
        objective_errorbar(metrics)
