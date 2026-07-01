from pathlib import Path

from gristmill_symbolics.profiling.training_profile import (
    TrainingProfileCase,
    build_train_command,
    parse_elapsed_seconds,
    parse_max_rss_kbytes,
    summarize_case,
)


def _case(tmp_path: Path, *, cprofile: bool = False) -> TrainingProfileCase:
    return TrainingProfileCase(
        input_path=tmp_path / "working_eqn.json",
        run_root=tmp_path / "runs",
        run_name="updates2-batch4",
        updates=2,
        batch_size=4,
        max_steps=64,
        seed=7,
        state_token_pad_to=5000,
        action_token_pad_to=5000,
        definition_pad_to=128,
        candidate_pad_to=2048,
        side_term_pad_to=256,
        d_model=32,
        num_attention_layers=1,
        num_attention_heads=4,
        sample_ms=250,
        cprofile=cprofile,
        xla_dump=False,
        require_gpu=True,
        python_executable="/venv/bin/python",
        repo_root=tmp_path,
    )


def test_build_train_command_uses_static_padding_flags(tmp_path):
    command = build_train_command(_case(tmp_path))

    assert command[:3] == ["/usr/bin/time", "-v", "/venv/bin/python"]
    assert command[3:5] == ["-m", "gristmill_symbolics.cli.train"]
    assert "--updates" in command
    assert command[command.index("--updates") + 1] == "2"
    assert command[command.index("--batch-size") + 1] == "4"
    assert command[command.index("--state-token-pad-to") + 1] == "5000"
    assert command[command.index("--action-token-pad-to") + 1] == "5000"
    assert command[command.index("--candidate-pad-to") + 1] == "2048"
    assert command[command.index("--side-term-pad-to") + 1] == "256"
    assert command[command.index("--d-model") + 1] == "32"
    assert command[command.index("--num-attention-heads") + 1] == "4"


def test_build_train_command_writes_cprofile_to_run_dir(tmp_path):
    command = build_train_command(_case(tmp_path, cprofile=True))

    profile_path = tmp_path / "runs" / "updates2-batch4" / "profile.prof"
    assert command[3:7] == ["-m", "cProfile", "-o", str(profile_path)]
    assert command[7:9] == ["-m", "gristmill_symbolics.cli.train"]


def test_parse_time_v_elapsed_seconds():
    assert parse_elapsed_seconds("Elapsed (wall clock) time: 17:34.14\n")
    assert parse_elapsed_seconds("Elapsed (wall clock) time: 17:34.14\n") == (
        17 * 60 + 34.14
    )
    assert parse_elapsed_seconds("Elapsed (wall clock) time: 1:02:03\n") == (
        3600 + 2 * 60 + 3
    )


def test_parse_time_v_max_rss():
    stderr = "Maximum resident set size (kbytes): 6736540\n"

    assert parse_max_rss_kbytes(stderr) == 6736540


def test_summarize_case_reads_wall_time_gpu_and_xla(tmp_path):
    case = _case(tmp_path)
    run_dir = tmp_path / "runs" / "updates2-batch4"
    run_dir.mkdir(parents=True)
    (run_dir / "status.txt").write_text("0\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text(
        "\n".join(
            [
                "WARNING: Compiling train_step with global shapes.",
                "Elapsed (wall clock) time: 0:02.50",
                "Maximum resident set size (kbytes): 123456",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "stdout.jsonl").write_text(
        '{"reward_mean": 1.0}\n{"reward_mean": 2.0}\n',
        encoding="utf-8",
    )
    (run_dir / "nvidia-smi.csv").write_text(
        "\n".join(
            [
                "timestamp, index, name, utilization.gpu [%], "
                "memory.used [MiB]",
                "2026/07/01 00:00:00, 0, RTX 4060 Ti, 10 %, 512 MiB",
                "2026/07/01 00:00:01, 0, RTX 4060 Ti, 30 %, 640 MiB",
            ]
        ),
        encoding="utf-8",
    )
    xla_dir = run_dir / "xla"
    xla_dir.mkdir()
    (xla_dir / "module.txt").write_text("custom-call cudnn\n", encoding="utf-8")

    summary = summarize_case(case, run_dir)

    assert summary.status == 0
    assert summary.elapsed_seconds == 2.5
    assert summary.max_rss_kbytes == 123456
    assert summary.peak_gpu_memory_mib == 640
    assert summary.avg_gpu_util_percent == 20
    assert summary.peak_gpu_util_percent == 30
    assert summary.compile_count == 1
    assert summary.cudnn_hlo_count == 1
    assert summary.final_metrics == '{"reward_mean": 2.0}'
