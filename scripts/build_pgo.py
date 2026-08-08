#!/usr/bin/env python3
"""Build and smoke-test a host-native Volkrix binary with Rust/LLVM PGO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import shlex
import subprocess
import sys
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PgoFailure(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    print("+", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if result.returncode != 0:
        detail = f"\n{result.stdout}" if capture and result.stdout else ""
        raise PgoFailure(f"command failed with status {result.returncode}: {command}{detail}")
    return result.stdout.strip() if capture and result.stdout else ""


def rust_host() -> str:
    for line in run(["rustc", "-vV"], capture=True).splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ")
    raise PgoFailure("rustc -vV did not report a host triple")


def find_llvm_profdata() -> pathlib.Path:
    override = os.environ.get("LLVM_PROFDATA")
    if override:
        candidate = pathlib.Path(override).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise PgoFailure(f"LLVM_PROFDATA is not a file: {candidate}")

    executable = "llvm-profdata.exe" if os.name == "nt" else "llvm-profdata"
    sysroot = pathlib.Path(run(["rustc", "--print", "sysroot"], capture=True))
    bundled = sysroot / "lib" / "rustlib" / rust_host() / "bin" / executable
    if bundled.is_file():
        return bundled
    discovered = shutil.which("llvm-profdata")
    if discovered:
        return pathlib.Path(discovered).resolve()
    raise PgoFailure(
        "llvm-profdata was not found; install the matching Rust tool with "
        "'rustup component add llvm-tools-preview' or set LLVM_PROFDATA"
    )


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_path(target_dir: pathlib.Path) -> pathlib.Path:
    suffix = ".exe" if os.name == "nt" else ""
    return target_dir / "release" / f"volkrix{suffix}"


def evaluator_argument(value: str) -> tuple[str, str | None]:
    if value == "classical":
        return value, None
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_file():
        raise PgoFailure(f"EvalFile is not a file: {path}")
    return str(path), sha256(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new final binary path")
    parser.add_argument(
        "--evalfile",
        required=True,
        help="'classical' or an existing network file used for profiling and smoke tests",
    )
    parser.add_argument(
        "--work-dir",
        default=str(ROOT / "target" / "pgo"),
        help="new directory for raw profiles and the two isolated Cargo target trees",
    )
    parser.add_argument("--bench-depth", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--base-rustflags",
        default="",
        help="additional identical rustflags for generate and use builds",
    )
    args = parser.parse_args(argv)
    if args.bench_depth <= 0 or args.threads <= 0:
        parser.error("--bench-depth and --threads must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if os.environ.get("RUSTFLAGS") or os.environ.get("CARGO_ENCODED_RUSTFLAGS"):
            raise PgoFailure(
                "ambient Rust flags are set; unset them and use --base-rustflags "
                "so the build is recorded"
            )
        evaluator, evaluator_sha = evaluator_argument(args.evalfile)
        output = pathlib.Path(args.output).expanduser().resolve()
        work_dir = pathlib.Path(args.work_dir).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise PgoFailure(f"refusing to overwrite output: {output}")
        if not output.parent.is_dir():
            raise PgoFailure(f"output parent directory does not exist: {output.parent}")
        if work_dir.exists() or work_dir.is_symlink():
            raise PgoFailure(f"refusing to reuse PGO work directory: {work_dir}")

        llvm_profdata = find_llvm_profdata()
        raw_profiles = work_dir / "raw"
        generate_target = work_dir / "generate-target"
        use_target = work_dir / "use-target"
        raw_profiles.mkdir(parents=True)

        base_flags = shlex.split(args.base_rustflags)
        generate_flags = [*base_flags, f"-Cprofile-generate={raw_profiles}"]
        generate_env = os.environ.copy()
        generate_env.update(
            {
                "CARGO_TARGET_DIR": str(generate_target),
                "CARGO_ENCODED_RUSTFLAGS": "\x1f".join(generate_flags),
            }
        )
        build_command = [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--package",
            "volkrix",
            "--bin",
            "volkrix",
        ]
        run(build_command, env=generate_env)
        instrumented = binary_path(generate_target)

        profile_env = os.environ.copy()
        profile_env["LLVM_PROFILE_FILE"] = str(raw_profiles / "%m-%p.profraw")
        bench_base = [
            str(instrumented),
            "bench",
            "--depth",
            str(args.bench_depth),
            "--hash-mb",
            "64",
            "--evalfile",
            evaluator,
        ]
        run(bench_base + ["--threads", "1"], env=profile_env)
        if args.threads > 1:
            run(bench_base + ["--threads", str(args.threads)], env=profile_env)
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "uci_smoke.py"),
                "--engine",
                str(instrumented),
                "--evalfile",
                evaluator,
                "--threads",
                str(args.threads),
            ],
            env=profile_env,
        )

        raw_files = sorted(raw_profiles.glob("*.profraw"))
        if not raw_files:
            raise PgoFailure("instrumented workloads produced no .profraw files")
        merged = work_dir / "merged.profdata"
        run(
            [
                str(llvm_profdata),
                "merge",
                "-o",
                str(merged),
                *[str(path) for path in raw_files],
            ]
        )

        use_flags = [
            *base_flags,
            f"-Cprofile-use={merged}",
            "-Cllvm-args=-pgo-warn-missing-function",
        ]
        use_env = os.environ.copy()
        use_env.update(
            {
                "CARGO_TARGET_DIR": str(use_target),
                "CARGO_ENCODED_RUSTFLAGS": "\x1f".join(use_flags),
            }
        )
        run(build_command, env=use_env)
        built = binary_path(use_target)
        shutil.copy2(built, output)

        transcript = work_dir / "release-smoke.log"
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "uci_smoke.py"),
                "--engine",
                str(output),
                "--evalfile",
                evaluator,
                "--threads",
                str(args.threads),
                "--transcript",
                str(transcript),
            ]
        )
        manifest = {
            "schema": "volkrix-pgo-build-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "rustc": run(["rustc", "-vV"], capture=True),
            "llvm_profdata": str(llvm_profdata),
            "base_rustflags": base_flags,
            "evaluator": evaluator,
            "evaluator_sha256": evaluator_sha,
            "bench_depth": args.bench_depth,
            "threads": args.threads,
            "raw_profile_count": len(raw_files),
            "merged_profile_sha256": sha256(merged),
            "output": str(output),
            "output_sha256": sha256(output),
        }
        (work_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"PGO build passed smoke testing: {output}")
        print(f"provenance: {work_dir / 'manifest.json'}")
        return 0
    except PgoFailure as error:
        print(f"PGO build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
