#!/usr/bin/env python3
"""Reproducible, resumable FastChess gauntlets for Volkrix.

The lab deliberately uses only the Python standard library. A prepared lab is
immutable: all executables, networks, books, the resolved configuration, and
each exact command are checksummed before any games are started.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import queue
import re
import shutil
import shlex
import socket
import subprocess
import sys
import threading
import time
from typing import Any


SCHEMA = "volkrix-strength-lab-v1"
RESULTS = {"1-0", "0-1", "1/2-1/2"}
EXPECTED_TERMINATIONS = {"normal", "adjudication"}
FAILURE_CLASSES = {
    "abandoned": "crash_or_stall",
    "disconnect": "crash_or_disconnect",
    "stalled connection": "hang_or_stall",
    "time forfeit": "time_forfeit",
    "illegal move": "illegal_move",
    "unterminated": "interrupted",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TAG_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:\\.|[^"\\])*)"\]\s*$')
UCI_ERROR_RE = re.compile(
    r"(?im)^(?:error\b|info string (?:error|failed)\b|"
    r".*(?:unknown|invalid|unsupported|unrecognized|no such)\s+(?:uci\s+)?option\b)"
)


class LabError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_new(path: pathlib.Path, data: bytes, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.chmod(mode)


def write_replace(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def write_new_atomic(path: pathlib.Path, data: bytes, mode: int = 0o444) -> None:
    """Atomically publishes a new file without replacing an existing target."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    write_new(temporary, data, mode)
    try:
        # Linking is an atomic create-if-absent operation. Unlike os.replace,
        # it cannot overwrite a pre-existing commit marker.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_new_verified(source: pathlib.Path, destination: pathlib.Path, digest: str) -> None:
    """Copy one immutable input without buffering large books or networks in memory."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as input_file, os.fdopen(descriptor, "wb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if sha256_file(destination) != digest:
        destination.unlink(missing_ok=True)
        raise LabError(f"frozen input copy changed while reading: {source}")


def freeze_inputs(manifest: dict[str, Any], output: pathlib.Path) -> None:
    """Make a prepared lab self-contained and rewrite every runtime path to its copy."""
    inputs = output / "inputs"
    inputs.mkdir()
    assets = [
        manifest["fastchess"],
        manifest["candidate"],
        manifest["openings"],
        *manifest["opponents"],
        *manifest.get("assets", []),
    ]
    executable_paths = {
        manifest["fastchess"]["path"],
        manifest["candidate"]["path"],
        *(engine["path"] for engine in manifest["opponents"]),
    }
    path_map: dict[str, str] = {}
    frozen_by_digest: dict[str, pathlib.Path] = {}
    for asset in assets:
        source_text = asset["path"]
        source = pathlib.Path(source_text)
        digest = asset["sha256"]
        destination = frozen_by_digest.get(digest)
        if destination is None:
            destination = inputs / digest
            copy_new_verified(source, destination, digest)
            frozen_by_digest[digest] = destination
        path_map[source_text] = str(destination.resolve())

    for destination in frozen_by_digest.values():
        destination.chmod(0o444)
    for source_text in executable_paths:
        pathlib.Path(path_map[source_text]).chmod(0o555)
    for asset in assets:
        asset["path"] = path_map[asset["path"]]

    for owner in [manifest["candidate"], *manifest["opponents"], *manifest["profiles"]]:
        for option_name, option_value in owner["options"].items():
            if option_value in path_map:
                owner["options"][option_name] = path_map[option_value]
    manifest["input_storage"] = "self-contained-copy-v1"


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise LabError(f"{label} must be a non-empty JSON array")
    return value


def safe_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_NAME_RE.fullmatch(value):
        raise LabError(f"{label} must match {SAFE_NAME_RE.pattern}")
    return value


def positive_int(value: Any, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LabError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise LabError(f"{label} must be <= {maximum}")
    return value


def positive_number(value: Any, label: str, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise LabError(f"{label} must be a positive finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise LabError(f"{label} must be a positive finite number") from error
    if not math.isfinite(result):
        raise LabError(f"{label} must be a positive finite number")
    if maximum is not None and result > maximum:
        raise LabError(f"{label} must be <= {maximum}")
    return result


def single_line_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise LabError(f"{label} must be a non-empty single-line string")
    return value


def scalar_options(value: Any, label: str) -> dict[str, str]:
    if value is None:
        return {}
    source = require_object(value, label)
    result: dict[str, str] = {}
    for key, option_value in source.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or "=" in key
            or "\n" in key
            or "\r" in key
        ):
            raise LabError(f"{label} contains an invalid UCI option name")
        if not isinstance(option_value, (str, int, float, bool)):
            raise LabError(f"{label}.{key} must be a JSON scalar")
        if isinstance(option_value, float) and not math.isfinite(option_value):
            raise LabError(f"{label}.{key} must be finite")
        text = str(option_value).lower() if isinstance(option_value, bool) else str(option_value)
        if "\n" in text or "\r" in text:
            raise LabError(f"{label}.{key} contains a newline")
        if text == "" and key != "SyzygyPath":
            raise LabError(
                f"{label}.{key} cannot be empty; FastChess cannot transmit an empty "
                "option value (only the verified default-empty SyzygyPath is supported)"
            )
        result[key] = text
    return result


def resolve_asset(
    item: dict[str, Any], label: str, base: pathlib.Path, executable: bool = False
) -> dict[str, Any]:
    raw_path = item.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise LabError(f"{label}.path is required")
    path = pathlib.Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise LabError(f"{label} is not a file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise LabError(f"{label} is not executable: {path}")
    digest = sha256_file(path)
    expected = item.get("sha256")
    if expected is not None:
        if not isinstance(expected, str) or not SHA_RE.fullmatch(expected.lower()):
            raise LabError(f"{label}.sha256 must be 64 hexadecimal characters")
        if digest != expected.lower():
            raise LabError(f"{label} checksum mismatch: expected {expected}, got {digest}")
    return {"path": str(path), "sha256": digest, "size": path.stat().st_size}


def resolve_engine(item: Any, label: str, base: pathlib.Path) -> dict[str, Any]:
    engine = require_object(item, label)
    name = safe_name(engine.get("name"), f"{label}.name")
    asset = resolve_asset(engine, label, base, executable=True)
    rating = engine.get("rating")
    if rating is not None:
        if isinstance(rating, bool) or not isinstance(rating, (int, float)):
            raise LabError(f"{label}.rating must be a finite number")
        try:
            finite_rating = math.isfinite(float(rating))
        except OverflowError:
            finite_rating = False
        if not finite_rating:
            raise LabError(f"{label}.rating must be a finite number")
    asset.update(
        {
            "name": name,
            "options": scalar_options(engine.get("options"), f"{label}.options"),
            "rating": rating,
        }
    )
    return asset


def resolve_config(config_path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LabError(f"failed to read config: {error}") from error
    config = require_object(raw, "config")
    if config.get("schema") != SCHEMA:
        raise LabError(f"config.schema must be {SCHEMA!r}")
    base = config_path.resolve().parent
    fastchess = resolve_asset(require_object(config.get("fastchess"), "fastchess"), "fastchess", base, True)
    candidate = resolve_engine(config.get("candidate"), "candidate", base)
    openings_raw = require_object(config.get("openings"), "openings")
    openings = resolve_asset(openings_raw, "openings", base)
    book_format = openings_raw.get("format")
    if book_format not in {"epd", "pgn"}:
        raise LabError("openings.format must be 'epd' or 'pgn'")
    openings.update({"format": book_format, "start": positive_int(openings_raw.get("start", 1), "openings.start")})

    opponents: list[dict[str, Any]] = []
    names = {candidate["name"]}
    for index, raw_opponent in enumerate(require_list(config.get("opponents"), "opponents")):
        opponent = resolve_engine(raw_opponent, f"opponents[{index}]", base)
        if opponent["name"] in names:
            raise LabError(f"duplicate engine name {opponent['name']!r}")
        names.add(opponent["name"])
        opponents.append(opponent)

    extra_assets: list[dict[str, Any]] = []
    asset_names: set[str] = set()
    raw_assets = config.get("assets", [])
    if not isinstance(raw_assets, list):
        raise LabError("assets must be a JSON array")
    for index, raw_asset in enumerate(raw_assets):
        item = require_object(raw_asset, f"assets[{index}]")
        name = safe_name(item.get("name"), f"assets[{index}].name")
        if name in asset_names:
            raise LabError(f"duplicate asset name {name!r}")
        asset_names.add(name)
        asset = resolve_asset(item, f"assets[{index}]", base)
        asset["name"] = name
        extra_assets.append(asset)

    frozen_asset_paths = {asset["path"] for asset in extra_assets}

    def freeze_eval_options(options: dict[str, str], owner: str) -> None:
        for option_name, option_value in options.items():
            if "evalfile" not in option_name.lower() or option_value in {"", "classical", "embedded"}:
                continue
            option_path = pathlib.Path(option_value).expanduser()
            if not option_path.is_absolute():
                option_path = base / option_path
            option_path = option_path.resolve()
            if str(option_path) not in frozen_asset_paths:
                raise LabError(
                    f"{owner} option {option_name} must reference a file listed in assets"
                )
            # FastChess launches each engine in its own directory. Persist the
            # verified absolute path, not a config-relative spelling that would
            # resolve differently at runtime.
            options[option_name] = str(option_path)

    for engine in [candidate, *opponents]:
        freeze_eval_options(engine["options"], engine["name"])

    profiles: list[dict[str, Any]] = []
    profile_names: set[str] = set()
    for index, raw_profile in enumerate(require_list(config.get("profiles"), "profiles")):
        profile = require_object(raw_profile, f"profiles[{index}]")
        name = safe_name(profile.get("name"), f"profiles[{index}].name")
        if name in profile_names:
            raise LabError(f"duplicate profile name {name!r}")
        profile_names.add(name)
        tc = profile.get("tc")
        if not isinstance(tc, str) or not tc.strip() or "\n" in tc or "\r" in tc:
            raise LabError(f"profiles[{index}].tc must be a FastChess time control")
        resolved_profile = {
            "name": name,
            "tc": tc,
            "pairs": positive_int(profile.get("pairs"), f"profiles[{index}].pairs"),
            "concurrency": positive_int(profile.get("concurrency", 1), f"profiles[{index}].concurrency", 256),
            "options": scalar_options(profile.get("options"), f"profiles[{index}].options"),
            "time_margin_ms": positive_int(profile.get("time_margin_ms", 1000), f"profiles[{index}].time_margin_ms", 60000),
        }
        freeze_eval_options(resolved_profile["options"], f"profile {name}")
        profiles.append(resolved_profile)

    adjudication = require_object(config.get("adjudication", {}), "adjudication")
    resign = single_line_string(
        adjudication.get("resign", "movecount=3 score=400"),
        "adjudication.resign",
    )
    draw = single_line_string(
        adjudication.get("draw", "movenumber=40 movecount=8 score=10"),
        "adjudication.draw",
    )
    # Validate the FastChess field grammar before launching any UCI preflights
    # or creating a prepared-lab directory.
    adjudication_fields(resign, "adjudication.resign")
    adjudication_fields(draw, "adjudication.draw")
    return {
        "schema": SCHEMA,
        "fastchess": fastchess,
        "candidate": candidate,
        "openings": openings,
        "opponents": opponents,
        "assets": extra_assets,
        "profiles": profiles,
        "adjudication": {
            "resign": resign,
            "draw": draw,
        },
        "preflight_timeout_seconds": positive_number(
            config.get("preflight_timeout_seconds", 30),
            "preflight_timeout_seconds",
            300,
        ),
    }


def effective_options(engine: dict[str, Any], profile: dict[str, Any]) -> dict[str, str]:
    options = dict(profile["options"])
    options.update(engine["options"])
    return options


def engine_arguments(engine: dict[str, Any], profile: dict[str, Any], role: str) -> list[str]:
    options = effective_options(engine, profile)
    args = [
        "-engine",
        f"name={engine['name']}",
        f"cmd={engine['path']}",
        f"dir={pathlib.Path(engine['path']).parent}",
    ]
    for name, value in sorted(options.items()):
        # FastChess parses each engine token as a non-empty key=value pair and rejects
        # option.SyzygyPath=. Configuration validation permits only this known
        # default-empty option, so omitting it preserves the preflighted state.
        if value == "":
            if name != "SyzygyPath":
                raise LabError(
                    f"cannot transmit empty FastChess option {name!r}; only the verified "
                    "default-empty SyzygyPath is supported"
                )
            continue
        args.append(f"option.{name}={value}")
    # Role is recorded in the manifest even though FastChess does not need it.
    assert role in {"candidate", "opponent"}
    return args


def adjudication_fields(value: str, label: str) -> list[str]:
    try:
        fields = shlex.split(value)
    except ValueError as error:
        raise LabError(f"{label} is invalid: {error}") from error
    if not fields or any(field.startswith("-") or "=" not in field for field in fields):
        raise LabError(f"{label} must contain only key=value fields")
    return fields


def job_command(manifest: dict[str, Any], profile: dict[str, Any], opponent: dict[str, Any], job_dir: pathlib.Path) -> list[str]:
    openings = manifest["openings"]
    command = [
        manifest["fastchess"]["path"], "-recover", "-repeat", "-games", "2",
        "-rounds", str(profile["pairs"]), "-strict", "-report", "penta=true",
        "-variant", "standard", "-concurrency", str(profile["concurrency"]),
        "-ratinginterval", "1", "-scoreinterval", "1", "-autosaveinterval", "2",
        "-openings", f"file={openings['path']}", f"format={openings['format']}",
        "order=sequential", f"start={openings['start']}",
    ]
    command += engine_arguments(manifest["candidate"], profile, "candidate")
    command += engine_arguments(opponent, profile, "opponent")
    command += [
        "-each", f"tc={profile['tc']}", "proto=uci", f"timemargin={profile['time_margin_ms']}",
        "-resign", *adjudication_fields(manifest["adjudication"]["resign"], "adjudication.resign"),
        "-draw", *adjudication_fields(manifest["adjudication"]["draw"], "adjudication.draw"),
        "-pgnout", f"file={job_dir / 'games.pgn'}", "append=false",
        "-log", f"file={job_dir / 'fastchess.log'}", "level=info", "engine=true", "append=false",
        "-config", f"outname={job_dir / 'recovery.json'}", "stats=true",
    ]
    return command


def uci_preflight(
    engine: dict[str, Any], options: dict[str, str], timeout_seconds: float
) -> tuple[str, bytes]:
    option_commands = [
        f"setoption name {name} value {value}" for name, value in sorted(options.items())
    ]
    commands = ["uci", *option_commands, "isready", "position startpos", "go depth 1", "quit"]
    stdin = "\n".join(commands) + "\n"
    process: subprocess.Popen[str] | None = None
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_queue: queue.Queue[str | None] = queue.Queue()
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise LabError(
                f"UCI preflight timed out for {engine['name']} after {timeout_seconds:g}s"
            )
        return duration

    def collect_stdout(stream: Any) -> None:
        try:
            for line in stream:
                stdout_lines.append(line)
                stdout_queue.put(line)
        finally:
            stdout_queue.put(None)

    def collect_stderr(stream: Any) -> None:
        stderr_lines.extend(stream)

    def send(lines: list[str]) -> None:
        assert process is not None and process.stdin is not None
        try:
            process.stdin.write("\n".join(lines) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            # A process may emit a decisive diagnostic and exit between the preceding
            # protocol response and this write. Give the reader threads a bounded chance
            # to drain that evidence so scheduling does not turn the same failure into a
            # nondeterministic generic broken-pipe report.
            drain_timeout = min(remaining(), 0.25)
            if stdout_thread is not None:
                stdout_thread.join(timeout=drain_timeout)
            if stderr_thread is not None:
                stderr_thread.join(timeout=drain_timeout)
            diagnostic = "".join(stdout_lines) + "\n" + "".join(stderr_lines)
            match = UCI_ERROR_RE.search(diagnostic)
            if match:
                raise LabError(
                    f"UCI preflight for {engine['name']} reported an option/error "
                    f"diagnostic: {match.group(0).strip()}"
                ) from error
            raise LabError(f"UCI preflight pipe failed for {engine['name']}: {error}") from error

    def wait_for(label: str, predicate: Any) -> None:
        while True:
            try:
                line = stdout_queue.get(timeout=remaining())
            except queue.Empty as error:
                raise LabError(
                    f"UCI preflight timed out waiting for {label} from {engine['name']}"
                ) from error
            if line is None:
                raise LabError(
                    f"UCI preflight for {engine['name']} exited before {label}"
                )
            stripped = line.strip()
            if UCI_ERROR_RE.search(stripped):
                raise LabError(
                    f"UCI preflight for {engine['name']} reported an option/error diagnostic: "
                    f"{stripped}"
                )
            if predicate(stripped):
                return

    try:
        process = subprocess.Popen(
            [engine["path"]],
            cwd=pathlib.Path(engine["path"]).parent,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout_thread = threading.Thread(
            target=collect_stdout, args=(process.stdout,), daemon=True
        )
        stderr_thread = threading.Thread(
            target=collect_stderr, args=(process.stderr,), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        send(["uci"])
        wait_for("uciok", lambda line: line == "uciok")
        send([*option_commands, "isready"])
        wait_for("readyok", lambda line: line == "readyok")
        send(["position startpos", "go depth 1"])
        wait_for("bestmove", lambda line: line.startswith("bestmove "))
        send(["quit"])
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=remaining())
        stdout_thread.join(timeout=remaining())
        stderr_thread.join(timeout=remaining())
    except subprocess.TimeoutExpired as error:
        raise LabError(
            f"UCI preflight timed out for {engine['name']} after {timeout_seconds:g}s"
        ) from error
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            # Drain reader threads after process exit, then explicitly close all
            # three pipe wrappers on both successful and exceptional paths.
            if stdout_thread is not None:
                stdout_thread.join(timeout=1)
            if stderr_thread is not None:
                stderr_thread.join(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    combined = stdout
    if stderr:
        combined += "\n--- stderr ---\n" + stderr
    assert process is not None and process.returncode is not None
    if process.returncode != 0:
        raise LabError(
            f"UCI preflight for {engine['name']} exited {process.returncode}: {combined[-2000:]}"
        )
    if UCI_ERROR_RE.search(combined):
        raise LabError(
            f"UCI preflight for {engine['name']} reported an option/error diagnostic: "
            f"{combined[-2000:]}"
        )
    transcript = (
        "=== stdin ===\n"
        + stdin
        + "=== stdout ===\n"
        + stdout
        + "=== stderr ===\n"
        + stderr
        + f"=== exit_code ===\n{process.returncode}\n"
    ).encode("utf-8")
    return stdin, transcript


def prepare_preflights(
    manifest: dict[str, Any], output: pathlib.Path
) -> list[dict[str, Any]]:
    preflight_root = output / "preflight"
    preflight_root.mkdir()
    records: list[dict[str, Any]] = []
    seen: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}
    for profile in manifest["profiles"]:
        for engine in [manifest["candidate"], *manifest["opponents"]]:
            options = effective_options(engine, profile)
            # The executable directory is part of startup state for engines
            # that discover sibling networks, so do not deduplicate by bytes
            # alone when two copies live in different directories.
            key = (engine["path"], tuple(sorted(options.items())))
            context = {"profile": profile["name"], "engine": engine["name"]}
            if key in seen:
                seen[key]["contexts"].append(context)
                continue
            stdin, transcript = uci_preflight(
                engine, options, manifest["preflight_timeout_seconds"]
            )
            preflight_id = f"{engine['name']}__{len(records) + 1:03d}"
            relative_log = f"preflight/{preflight_id}.log"
            write_new(output / relative_log, transcript)
            record = {
                "id": preflight_id,
                "engine": engine["name"],
                "engine_sha256": engine["sha256"],
                "options": options,
                "stdin": stdin,
                "contexts": [context],
                "log": relative_log,
                "log_sha256": hashlib.sha256(transcript).hexdigest(),
            }
            seen[key] = record
            records.append(record)
    return records


def verify_preflights(root: pathlib.Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("preflights")
    if not isinstance(records, list) or not records:
        raise LabError("lab manifest has no UCI preflight evidence")
    for record in records:
        relative = record.get("log")
        digest = record.get("log_sha256")
        if (
            not isinstance(relative, str)
            or pathlib.PurePosixPath(relative).is_absolute()
            or ".." in pathlib.PurePosixPath(relative).parts
            or not isinstance(digest, str)
            or not SHA_RE.fullmatch(digest)
        ):
            raise LabError("lab manifest has an invalid UCI preflight record")
        path = root / pathlib.PurePosixPath(relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise LabError(f"UCI preflight evidence changed or disappeared: {path}")


def prepare(config_path: pathlib.Path, output: pathlib.Path) -> None:
    manifest = resolve_config(config_path)
    if output.exists() or output.is_symlink():
        raise LabError(f"output already exists; prepared labs are immutable: {output}")
    output.mkdir(parents=True)
    try:
        manifest.update(
            {
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "source_config": str(config_path.resolve()),
                "python": sys.version,
            }
        )
        freeze_inputs(manifest, output)
        manifest["preflights"] = prepare_preflights(manifest, output)
        jobs: list[dict[str, Any]] = []
        jobs_root = output / "jobs"
        jobs_root.mkdir()
        for profile in manifest["profiles"]:
            for opponent in manifest["opponents"]:
                job_id = f"{profile['name']}__{opponent['name']}"
                job_dir = jobs_root / job_id
                job_dir.mkdir()
                command = job_command(manifest, profile, opponent, job_dir.resolve())
                job = {
                    "schema": SCHEMA,
                    "id": job_id,
                    "profile": profile,
                    "candidate": manifest["candidate"]["name"],
                    "opponent": opponent["name"],
                    "opponent_rating": opponent.get("rating"),
                    "expected_games": profile["pairs"] * 2,
                    "command": command,
                }
                job_bytes = canonical_json(job)
                write_new(job_dir / "job.json", job_bytes)
                write_new(job_dir / "command.sh", (shlex.join(command) + "\n").encode())
                jobs.append({"id": job_id, "job_sha256": hashlib.sha256(job_bytes).hexdigest()})
        manifest["jobs"] = jobs
        write_new(output / "manifest.json", canonical_json(manifest))
        write_new(output / "manifest.sha256", (sha256_file(output / "manifest.json") + "  manifest.json\n").encode())
    except BaseException:
        # Leave partial output in place: it is evidence and cannot accidentally
        # be mistaken for a prepared lab because manifest.sha256 is absent.
        raise
    print(f"prepared {len(manifest['jobs'])} immutable jobs at {output.resolve()}")


def load_lab(root: pathlib.Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    digest_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise LabError("lab is incomplete: manifest.json or manifest.sha256 is missing")
    recorded = digest_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(manifest_path)
    if recorded != actual:
        raise LabError(f"lab manifest checksum mismatch: expected {recorded}, got {actual}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise LabError("unsupported lab schema")
    return manifest


def verify_inputs(manifest: dict[str, Any]) -> None:
    assets = [
        manifest["fastchess"], manifest["candidate"], manifest["openings"],
        *manifest["opponents"], *manifest.get("assets", []),
    ]
    for asset in assets:
        path = pathlib.Path(asset["path"])
        if not path.is_file():
            raise LabError(f"frozen input disappeared: {path}")
        actual = sha256_file(path)
        if actual != asset["sha256"]:
            raise LabError(f"frozen input changed: {path}; expected {asset['sha256']}, got {actual}")


def read_games(path: pathlib.Path) -> list[dict[str, str]]:
    games: list[dict[str, str]] = []
    tags: dict[str, str] = {}
    if not path.exists():
        return games
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TAG_RE.match(line)
        if match:
            if match.group(1) == "Event" and tags:
                if "Result" in tags:
                    games.append(tags)
                tags = {}
            tags[match.group(1)] = match.group(2).replace(r'\"', '"').replace(r"\\", "\\")
    if tags and "Result" in tags:
        games.append(tags)
    return games


def summarize_games(path: pathlib.Path, candidate: str, expected_games: int | None) -> dict[str, Any]:
    games = read_games(path)
    summary: dict[str, Any] = {
        "games": len(games), "expected_games": expected_games,
        "candidate_wins": 0, "draws": 0, "candidate_losses": 0,
        "failures": {name: 0 for name in sorted(set(FAILURE_CLASSES.values()) | {"unknown_abnormal"})},
        "terminations": {},
    }
    points: list[float] = []
    rounds: dict[str, list[dict[str, str]]] = {}

    def candidate_point(game: dict[str, str]) -> float:
        result = game["Result"]
        if result == "1/2-1/2":
            return 0.5
        won = (result == "1-0" and game.get("White") == candidate) or (
            result == "0-1" and game.get("Black") == candidate
        )
        return 1.0 if won else 0.0

    for game in games:
        result = game.get("Result", "*")
        if result not in RESULTS:
            raise LabError(f"PGN contains unfinished or invalid result {result!r}")
        white = game.get("White")
        black = game.get("Black")
        if candidate not in {white, black}:
            raise LabError(f"PGN game does not contain candidate {candidate!r}")
        round_tag = game.get("Round", "").strip()
        if not round_tag:
            raise LabError("PGN game is missing a non-empty Round tag")
        rounds.setdefault(round_tag, []).append(game)
        if result == "1/2-1/2":
            summary["draws"] += 1
        elif (result == "1-0" and white == candidate) or (result == "0-1" and black == candidate):
            summary["candidate_wins"] += 1
        else:
            summary["candidate_losses"] += 1
        points.append(candidate_point(game))
        termination = game.get("Termination", "unknown").strip().lower()
        summary["terminations"][termination] = summary["terminations"].get(termination, 0) + 1
        if termination not in EXPECTED_TERMINATIONS:
            summary["failures"][FAILURE_CLASSES.get(termination, "unknown_abnormal")] += 1
    if not games:
        raise LabError("PGN contains no complete games")
    if expected_games is not None and len(games) != expected_games:
        raise LabError(f"PGN has {len(games)} complete games; expected {expected_games}")
    pair_scores: list[float] = []
    for round_tag, pair in rounds.items():
        if len(pair) != 2:
            raise LabError(
                f"Round {round_tag!r} has {len(pair)} games; expected exactly two"
            )
        candidate_colors = [
            "white" if game.get("White") == candidate else "black" for game in pair
        ]
        if sorted(candidate_colors) != ["black", "white"]:
            raise LabError(
                f"Round {round_tag!r} is not a color-reversed candidate pair"
            )
        opponents = {
            game.get("Black") if game.get("White") == candidate else game.get("White")
            for game in pair
        }
        if len(opponents) != 1 or None in opponents or candidate in opponents:
            raise LabError(f"Round {round_tag!r} does not use one consistent opponent")
        pair_scores.append(sum(candidate_point(game) for game in pair) / 2.0)
    if len(pair_scores) * 2 != len(points):
        raise LabError("PGN does not contain complete, uniquely tagged opening pairs")
    categories = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}
    for score in pair_scores:
        categories[score] += 1
    score = sum(points) / len(points)

    def elo(value: float) -> float:
        bounded = min(1.0 - 1e-9, max(1e-9, value))
        return 400.0 * math.log10(bounded / (1.0 - bounded))

    summary["score_percent"] = round(score * 100.0, 4)
    summary["elo_difference"] = round(elo(score), 3)
    summary["pentanomial"] = [categories[key] for key in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if len(pair_scores) > 1:
        variance = sum((value - score) ** 2 for value in pair_scores) / (len(pair_scores) - 1)
        standard_error = math.sqrt(variance / len(pair_scores))
        lower = max(1e-9, score - 1.96 * standard_error)
        upper = min(1.0 - 1e-9, score + 1.96 * standard_error)
        summary["elo_95_percent"] = [round(elo(lower), 3), round(elo(upper), 3)]
    else:
        summary["elo_95_percent"] = None
    return summary


def artifact_hashes(job_dir: pathlib.Path) -> dict[str, str]:
    names = ["games.pgn", "fastchess.log", "console.log", "recovery.json", "summary.json"]
    return {name: sha256_file(job_dir / name) for name in names if (job_dir / name).is_file()}


def verify_job(root: pathlib.Path, expected: dict[str, Any]) -> tuple[dict[str, Any], pathlib.Path]:
    job_dir = root / "jobs" / expected["id"]
    job_path = job_dir / "job.json"
    if sha256_file(job_path) != expected["job_sha256"]:
        raise LabError(f"job manifest changed: {expected['id']}")
    return json.loads(job_path.read_text(encoding="utf-8")), job_dir


def verified_completion(job_dir: pathlib.Path) -> dict[str, Any]:
    marker_path = job_dir / "completed.json"
    try:
        marker = require_object(
            json.loads(marker_path.read_text(encoding="utf-8")), "completed.json"
        )
        recorded = require_object(marker.get("artifacts"), "completed.json.artifacts")
    except (OSError, json.JSONDecodeError) as error:
        raise LabError(f"invalid completion marker {marker_path}: {error}") from error
    allowed = {
        "games.pgn", "fastchess.log", "console.log", "recovery.json", "summary.json"
    }
    if not {"games.pgn", "summary.json"} <= set(recorded):
        raise LabError(f"completion marker lacks required artifacts: {marker_path}")
    for name, digest in recorded.items():
        if name not in allowed or not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
            raise LabError(f"completion marker has invalid artifact entry {name!r}")
        artifact = job_dir / name
        if not artifact.is_file():
            raise LabError(f"completed artifact is missing: {artifact}")
        actual = sha256_file(artifact)
        if actual != digest:
            raise LabError(
                f"completed artifact changed: {artifact}; expected {digest}, got {actual}"
            )
    try:
        return require_object(
            json.loads((job_dir / "summary.json").read_text(encoding="utf-8")),
            "summary.json",
        )
    except (OSError, json.JSONDecodeError) as error:
        raise LabError(f"invalid completed summary in {job_dir}: {error}") from error


def run_lab(root: pathlib.Path, selected: set[str] | None, dry_run: bool) -> None:
    root = root.resolve(strict=True)
    manifest = load_lab(root)
    verify_inputs(manifest)
    verify_preflights(root, manifest)
    available = {job["id"] for job in manifest["jobs"]}
    if selected and not selected <= available:
        raise LabError(f"unknown jobs: {', '.join(sorted(selected - available))}")
    lock = root / ".run.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        os.write(descriptor, f"pid={os.getpid()} host={socket.gethostname()}\n".encode())
        os.close(descriptor)
    except FileExistsError as error:
        raise LabError(f"another run owns {lock}; remove it only after confirming that process is dead") from error
    try:
        for expected in manifest["jobs"]:
            if selected and expected["id"] not in selected:
                continue
            job, job_dir = verify_job(root, expected)
            if (job_dir / "completed.json").exists():
                verified_completion(job_dir)
                print(f"skip completed {job['id']}")
                continue
            recovery = job_dir / "recovery.json"
            command = (
                [manifest["fastchess"]["path"], "-config", f"file={recovery}", "stats=true"]
                if recovery.is_file() else job["command"]
            )
            if dry_run:
                print(f"{job['id']}: {shlex.join(command)}")
                continue
            print(f"running {job['id']}", flush=True)
            with (job_dir / "console.log").open("a", encoding="utf-8") as console:
                console.write(f"\n=== {dt.datetime.now(dt.timezone.utc).isoformat()} ===\n")
                console.flush()
                result = subprocess.run(command, cwd=job_dir, stdout=console, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                write_replace(job_dir / "last-exit.json", {"exit_code": result.returncode, "resumable": recovery.is_file()})
                raise LabError(f"{job['id']} exited {result.returncode}; rerun the same command to resume")
            summary = summarize_games(job_dir / "games.pgn", manifest["candidate"]["name"], job["expected_games"])
            if job.get("opponent_rating") is not None:
                summary["opponent_rating"] = job["opponent_rating"]
                summary["estimated_candidate_rating"] = round(
                    float(job["opponent_rating"]) + summary["elo_difference"], 3
                )
            write_replace(job_dir / "summary.json", summary)
            hashes = artifact_hashes(job_dir)
            write_replace(job_dir / "artifacts.sha256.json", hashes)
            write_new_atomic(job_dir / "completed.json", canonical_json({"completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "artifacts": hashes}))
        if not dry_run:
            aggregate(root, manifest)
    finally:
        lock.unlink(missing_ok=True)


def aggregate(root: pathlib.Path, manifest: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"schema": SCHEMA, "jobs": {}, "totals": {"games": 0, "candidate_wins": 0, "draws": 0, "candidate_losses": 0, "failures": {}}}
    for expected in manifest["jobs"]:
        _, job_dir = verify_job(root, expected)
        if not (job_dir / "completed.json").is_file():
            report["jobs"][expected["id"]] = {"status": "pending"}
            continue
        summary = verified_completion(job_dir)
        report["jobs"][expected["id"]] = {"status": "complete", **summary}
        for key in ("games", "candidate_wins", "draws", "candidate_losses"):
            report["totals"][key] += summary[key]
        for key, count in summary["failures"].items():
            report["totals"]["failures"][key] = report["totals"]["failures"].get(key, 0) + count
    write_replace(root / "report.json", report)
    return report


def verify_lab(root: pathlib.Path) -> None:
    root = root.resolve(strict=True)
    manifest = load_lab(root)
    verify_inputs(manifest)
    verify_preflights(root, manifest)
    for expected in manifest["jobs"]:
        _, job_dir = verify_job(root, expected)
        completed = job_dir / "completed.json"
        if not completed.exists():
            continue
        verified_completion(job_dir)
    print(f"verified immutable lab inputs and completed artifacts at {root}")


def status(root: pathlib.Path) -> None:
    root = root.resolve(strict=True)
    manifest = load_lab(root)
    verify_inputs(manifest)
    verify_preflights(root, manifest)
    report = aggregate(root, manifest)
    print(json.dumps(report, indent=2, sort_keys=True))


def summarize_pgn(path: pathlib.Path, candidate: str, output: pathlib.Path | None) -> None:
    summary = summarize_games(path.resolve(strict=True), candidate, None)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        if output.exists() or output.is_symlink():
            raise LabError(f"summary output already exists: {output}")
        write_new(output, encoded.encode())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=pathlib.Path, required=True)
    prepare_parser.add_argument("--output", type=pathlib.Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--lab", type=pathlib.Path, required=True)
    run_parser.add_argument("--job", action="append", dest="jobs")
    run_parser.add_argument("--dry-run", action="store_true")
    for command in ("status", "verify"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--lab", type=pathlib.Path, required=True)
    summary_parser = subparsers.add_parser("summarize-pgn")
    summary_parser.add_argument("--pgn", type=pathlib.Path, required=True)
    summary_parser.add_argument("--candidate", required=True)
    summary_parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.config, args.output)
        elif args.command == "run":
            run_lab(args.lab, set(args.jobs) if args.jobs else None, args.dry_run)
        elif args.command == "status":
            status(args.lab)
        elif args.command == "verify":
            verify_lab(args.lab)
        else:
            summarize_pgn(args.pgn, args.candidate, args.output)
    except (LabError, OSError, subprocess.SubprocessError) as error:
        print(f"strength lab error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
