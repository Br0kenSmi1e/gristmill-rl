from __future__ import annotations

from pathlib import Path

from gristmill_symbolics.profiling.profile_summary import format_summary, summarize_run


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


def test_summarize_run_extracts_failed_allocation_context(tmp_path: Path):
    (tmp_path / "stderr.log").write_text(
        "jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while "
        "trying to allocate 2.24GiB. [executable_name='jit_score_target'] "
        "[tf-allocator-allocation-error='']\n",
        encoding="utf-8",
    )

    summary = summarize_run(tmp_path)

    assert summary.failed_executable_name == "jit_score_target"
    assert summary.failed_allocation_bytes == int(2.24 * 1024**3)
    assert summary.failed_allocation_text == "2.24GiB"

    formatted = format_summary(summary)
    assert "failed_executable=jit_score_target" in formatted
    assert "failed_allocation=2.24 GiB requested (2.24GiB)" in formatted


def test_summarize_run_reports_largest_xla_shapes(tmp_path: Path):
    xla_dir = tmp_path / "xla"
    xla_dir.mkdir()
    (xla_dir / "module_0001.jit_score_target.before_optimizations.txt").write_text(
        "\n".join(
            [
                "ENTRY %main {",
                "  %scores = f32[8,5000,5000]{2,1,0} dot(%q, %k)",
                "  %weights = f32[8,5000,5000]{2,1,0} exponential(%scores)",
                "  %activations = bf16[8,5000,5000]{2,1,0} convert(%scores)",
                "  %defs = f32[8,128,64]{2,1,0} parameter(0)",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(tmp_path)

    assert summary.xla_files_scanned == 1
    assert summary.largest_xla_shapes[0].shape == "f32[8,5000,5000]"
    assert summary.largest_xla_shapes[0].estimated_bytes == 8 * 5000 * 5000 * 4
    assert summary.largest_xla_shapes[0].count == 2
    assert summary.largest_xla_shapes[0].sources == (
        "xla/module_0001.jit_score_target.before_optimizations.txt",
    )
    assert summary.largest_xla_shapes[1].shape == "bf16[8,5000,5000]"

    formatted = format_summary(summary)
    assert "762.94 MiB  count=2  f32[8,5000,5000]" in formatted
    assert "381.47 MiB  count=1  bf16[8,5000,5000]" in formatted


def test_summarize_run_reports_largest_xla_buffer_allocations(tmp_path: Path):
    xla_dir = tmp_path / "xla"
    xla_dir.mkdir()
    (xla_dir / "module_0002.jit_score_target.buffer-assignment.txt").write_text(
        "\n".join(
            [
                "BufferAssignment:",
                "allocation 7: size 2405181685, shape f32[8,5000,5000]{2,1,0}",
                "allocation 8: size 536870912, shape f32[8,128,1024]{2,1,0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(tmp_path)

    assert summary.largest_xla_allocations[0].size_bytes == 2405181685
    assert summary.largest_xla_allocations[0].shape == "f32[8,5000,5000]"
    assert summary.largest_xla_allocations[0].source == (
        "xla/module_0002.jit_score_target.buffer-assignment.txt:2"
    )

    formatted = format_summary(summary)
    assert "2.24 GiB  f32[8,5000,5000]" in formatted
