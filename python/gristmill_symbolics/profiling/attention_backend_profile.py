from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import resource
import shlex
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from .memory_profile_runner import (
    _run_nvidia_smi_snapshot,
    _start_nvidia_smi_sampler,
    _stop_sampler,
    _write_git_artifacts,
)


_BACKENDS = ("explicit", "xla", "cudnn")
_DTYPES = ("float32", "bfloat16", "float16")


@dataclass(frozen=True)
class AttentionProfileConfig:
    backend: str
    run_root: Path
    run_name: str | None
    batch_size: int
    seq_len: int
    d_model: int
    seed: int
    dtype: str
    sample_ms: int
    xla_dump: bool
    xla_dump_all_passes: bool
    repo_root: Path


def run_attention_profile(config: AttentionProfileConfig) -> Path:
    run_dir = _make_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_git_artifacts(config.repo_root, run_dir)
    _run_nvidia_smi_snapshot(run_dir / "nvidia-before.txt")
    (run_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")

    env = _profile_env(config, run_dir)
    old_env = {key: os.environ.get(key) for key in env}
    os.environ.update(env)

    sampler = _start_nvidia_smi_sampler(run_dir / "nvidia-smi.csv", config.sample_ms)
    status = 127
    try:
        with (run_dir / "stdout.jsonl").open("w", encoding="utf-8") as stdout:
            with (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    try:
                        _write_jax_env(run_dir)
                        metrics = _run_attention_value_and_grad(config)
                    except Exception:
                        traceback.print_exc()
                        status = 1
                    else:
                        print(json.dumps(metrics, sort_keys=True))
                        status = 0
                    _write_resource_usage()
    finally:
        _stop_sampler(sampler)
        (run_dir / "status.txt").write_text(f"{status}\n", encoding="utf-8")
        _run_nvidia_smi_snapshot(run_dir / "nvidia-after.txt")
        _restore_env(old_env)

    return run_dir


def _make_run_dir(config: AttentionProfileConfig) -> Path:
    if config.run_name:
        name = config.run_name
    else:
        now = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%d-%H%M%S")
        suffix = f"attention-{config.backend}-b{config.batch_size}-l{config.seq_len}"
        if config.xla_dump:
            suffix += "-xla"
        if config.xla_dump_all_passes:
            suffix += "-all-passes"
        name = f"{now}-{suffix}"
    return config.run_root / name


def _profile_env(config: AttentionProfileConfig, run_dir: Path) -> dict[str, str]:
    env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_LOG_COMPILES": "1",
    }
    if config.xla_dump:
        xla_dir = run_dir / "xla"
        xla_dir.mkdir(parents=True, exist_ok=True)
        flags = [f"--xla_dump_to={xla_dir}", "--xla_dump_hlo_as_text"]
        if config.xla_dump_all_passes:
            flags.append("--xla_dump_hlo_pass_re=.*")
        existing = os.environ.get("XLA_FLAGS")
        dump_flags = " ".join(flags)
        env["XLA_FLAGS"] = f"{existing} {dump_flags}" if existing else dump_flags
    return env


def _restore_env(old_env: dict[str, str | None]) -> None:
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write_jax_env(run_dir: Path) -> None:
    import jax

    lines = [
        f"jax {jax.__version__}",
        f"devices {jax.devices()}",
    ]
    (run_dir / "jax-env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_attention_value_and_grad(config: AttentionProfileConfig) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    dtype = _dtype(config.dtype, jnp)
    key = jax.random.PRNGKey(config.seed)
    q_key, k_key, v_key = jax.random.split(key, 3)
    shape = (config.batch_size, config.seq_len, config.d_model)
    q = jax.random.normal(q_key, shape, dtype=dtype)
    k = jax.random.normal(k_key, shape, dtype=dtype)
    v = jax.random.normal(v_key, shape, dtype=dtype)

    @jax.jit
    def value_and_grad(q, k, v):
        def loss_fn(q, k, v):
            out = _attention(config.backend, q, k, v)
            return jnp.mean(out.astype(jnp.float32) ** 2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_abs_sum = sum(jnp.sum(jnp.abs(grad.astype(jnp.float32))) for grad in grads)
        return loss, grad_abs_sum

    start = time.perf_counter()
    loss, grad_abs_sum = value_and_grad(q, k, v)
    loss.block_until_ready()
    grad_abs_sum.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(
        json.dumps(
            {
                "event": "rollout_phase",
                "phase": "attention_value_and_grad",
                "elapsed_ms": elapsed_ms,
                "state_token_len_max": config.seq_len,
                "active_count": config.batch_size,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return {
        "backend": config.backend,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "d_model": config.d_model,
        "dtype": config.dtype,
        "elapsed_ms": elapsed_ms,
        "loss": float(loss),
        "grad_abs_sum": float(grad_abs_sum),
    }


def _attention(backend: str, q, k, v):
    import jax
    import jax.numpy as jnp

    if backend == "explicit":
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = jnp.einsum("btd,bsd->bts", q, k) * scale
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bts,bsd->btd", weights, v)
    q4 = q[:, :, None, :]
    k4 = k[:, :, None, :]
    v4 = v[:, :, None, :]
    out = jax.nn.dot_product_attention(
        q4,
        k4,
        v4,
        implementation=backend,
    )
    return out[:, :, 0, :]


def _dtype(name: str, jnp):
    return {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }[name]


def _write_resource_usage() -> None:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_kbytes = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        max_rss_kbytes //= 1024
    print(f"\tMaximum resident set size (kbytes): {max_rss_kbytes}", file=sys.stderr)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _default_repo_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "python":
        return cwd.parent
    return cwd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a synthetic attention backend memory profile run."
    )
    parser.add_argument("--backend", choices=_BACKENDS, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/gristmill-attention-profile"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--seq-len", type=_positive_int, default=5000)
    parser.add_argument("--d-model", type=_positive_int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=_DTYPES, default="float32")
    parser.add_argument("--sample-ms", type=_positive_int, default=250)
    parser.add_argument("--xla-dump", action="store_true")
    parser.add_argument(
        "--xla-dump-all-passes",
        action="store_true",
        help=(
            "Dump text for every XLA HLO pass. This implies --xla-dump and can "
            "write a large xla/ directory."
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = AttentionProfileConfig(
        backend=args.backend,
        run_root=args.run_root,
        run_name=args.run_name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        seed=args.seed,
        dtype=args.dtype,
        sample_ms=args.sample_ms,
        xla_dump=args.xla_dump or args.xla_dump_all_passes,
        xla_dump_all_passes=args.xla_dump_all_passes,
        repo_root=args.repo_root.resolve(),
    )
    run_dir = run_attention_profile(config)
    status = int((run_dir / "status.txt").read_text(encoding="utf-8").strip())
    print(run_dir)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
