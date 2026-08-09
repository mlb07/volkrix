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


def run_bytes(command: list[str], *, cwd: pathlib.Path = ROOT) -> bytes:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise PgoFailure(
            f"command failed with status {result.returncode}: {command}"
            + (f"\n{detail}" if detail else "")
        )
    return result.stdout


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


def source_provenance(root: pathlib.Path = ROOT) -> dict[str, object]:
    """Return a binary-safe identity for the exact Git-visible source tree."""
    head = run_bytes(["git", "rev-parse", "HEAD"], cwd=root).decode("ascii").strip()
    tracked_diff = run_bytes(["git", "diff", "--binary", "HEAD", "--", "."], cwd=root)
    untracked_output = run_bytes(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
    )
    untracked_paths = sorted(path for path in untracked_output.split(b"\0") if path)

    untracked_digest = hashlib.sha256()
    untracked_digest.update(b"volkrix-untracked-v1\0")
    for raw_path in untracked_paths:
        path = root / os.fsdecode(raw_path)
        if path.is_symlink():
            kind = b"symlink"
            content = os.fsencode(os.readlink(path))
        elif path.is_file():
            kind = b"file"
            content = path.read_bytes()
        else:
            raise PgoFailure(f"untracked source path is not a file or symlink: {path}")
        for field in (raw_path, kind, content):
            untracked_digest.update(len(field).to_bytes(8, "big"))
            untracked_digest.update(field)

    tracked_diff_sha = hashlib.sha256(tracked_diff).hexdigest()
    untracked_sha = untracked_digest.hexdigest()
    source_digest = hashlib.sha256()
    source_digest.update(b"volkrix-source-tree-v1\0")
    source_digest.update(head.encode("ascii"))
    source_digest.update(bytes.fromhex(tracked_diff_sha))
    source_digest.update(bytes.fromhex(untracked_sha))
    source_tree_sha = source_digest.hexdigest()
    dirty = bool(tracked_diff or untracked_paths)
    source_id = head if not dirty else f"{head}-dirty-{source_tree_sha[:12]}"

    cargo_lock = root / "Cargo.lock"
    if not cargo_lock.is_file():
        raise PgoFailure(f"Cargo.lock is missing: {cargo_lock}")
    return {
        "source_commit": head,
        "source_dirty": dirty,
        "source_id": source_id,
        "source_tree_sha256": source_tree_sha,
        "tracked_diff_sha256": tracked_diff_sha,
        "untracked_sha256": untracked_sha,
        "untracked_paths": [os.fsdecode(path) for path in untracked_paths],
        "cargo_lock_sha256": sha256(cargo_lock),
    }


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
        source = source_provenance()
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
                "VOLKRIX_SOURCE_ID": str(source["source_id"]),
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
                "VOLKRIX_SOURCE_ID": str(source["source_id"]),
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
        if source_provenance() != source:
            raise PgoFailure("source tree changed during the PGO build; refusing provenance")
        manifest = {
            "schema": "volkrix-pgo-build-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "rustc": run(["rustc", "-vV"], capture=True),
            "llvm_profdata": str(llvm_profdata),
            **source,
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
