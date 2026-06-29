from __future__ import annotations

from pathlib import Path

from gristmill_symbolics.profiling.profile_summary import summarize_run


def test_summarize_run_reports_peak_gpu_memory_time_and_rollout_phases(
    tmp_path: Path,
):
    (tmp_path / "stdout.jsonl").write_text(
        '{"final_flops_best": 12.0, "update_index": 0}\n',
        encoding="utf-8",
    )
    (tmp_path / "stderr.log").write_text(
        "\n".join(
            [
                'Compiling jit(sample_action) with global shapes and types ...',
                '{"event":"rollout_phase","phase":"sample_action","elapsed_ms":12.5,'
                '"action_token_len_max":4096}',
                '{"event":"rollout_phase","phase":"score_action_grad","elapsed_ms":7.5,'
                '"state_token_len_max":3072,"definition_count_max":128}',
                "\tUser time (seconds): 10.00",
                "\tSystem time (seconds): 2.00",
                "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:06.00",
                "\tMaximum resident set size (kbytes): 123456",
                "\tExit status: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "nvidia-smi.csv").write_text(
        "\n".join(
            [
                "timestamp, memory.used [MiB], utilization.gpu [%], power.draw [W]",
                "2026/06/29 10:00:00.000, 100 MiB, 0 %, 20 W",
                "2026/06/29 10:00:00.250, 3456 MiB, 80 %, 55 W",
                "2026/06/29 10:00:00.500, 2000 MiB, 20 %, 30 W",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "status.txt").write_text("0\n", encoding="utf-8")

    summary = summarize_run(tmp_path)

    assert summary.status == 0
    assert summary.final_metrics == '{"final_flops_best": 12.0, "update_index": 0}'
    assert summary.compile_count == 1
    assert summary.max_rss_kbytes == 123456
    assert summary.peak_gpu_memory_mib == 3456.0
    assert summary.rollout_phase_totals_ms == {
        "sample_action": 12.5,
        "score_action_grad": 7.5,
    }
    assert summary.max_action_token_len == 4096
    assert summary.max_state_token_len == 3072
    assert summary.max_definition_count == 128


def test_summarize_run_extracts_oom_hlo_shapes(tmp_path: Path):
    (tmp_path / "stderr.log").write_text(
        "RESOURCE_EXHAUSTED: Autotuning failed for HLO: "
        "%input_reduce_fusion = f32[8,3542,8]{2,1,0} fusion(%a.1), "
        "calls=(param_0: f32[8,3542,3542,8]) -> f32[8,3542,8]\n",
        encoding="utf-8",
    )

    summary = summarize_run(tmp_path)

    assert "f32[8,3542,3542,8]" in summary.oom_hlo_shapes
    assert "f32[8,3542,8]" in summary.oom_hlo_shapes
