#!/usr/bin/env python3
"""Bounded, resumable local SPSA campaigns using paired FastChess games."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "volkrix-local-spsa-v1"
PARAMETER_RE = re.compile(
    r"^info string tuning parameter (\S+) value (-?\d+) default (-?\d+) "
    r"min (-?\d+) max (-?\d+) step (\d+)$"
)
EMBEDDED_EVAL_RE = re.compile(r"^<embedded:([0-9a-f]{64}):(\d+)>$")
SCORE_RE = re.compile(
    r"Score of SPSA-plus vs SPSA-minus:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
)
RESULTS_RE = re.compile(
    r"Games:\s*\d+,\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)"
)
TAG_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:\\.|[^"\\])*)"\]\s*$')
EXPECTED_TERMINATIONS = {"normal", "adjudication"}
FAILURE_CLASSES = {
    "abandoned": "crash_or_stall",
    "disconnect": "crash_or_disconnect",
    "stalled connection": "hang_or_stall",
    "time forfeit": "time_forfeit",
    "illegal move": "illegal_move",
    "unterminated": "interrupted",
}


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def campaign_lock(lab: Path):
    lock = lab / ".campaign.lock"
    payload = {"pid": os.getpid(), "host": socket.gethostname()}
    for attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            if attempt or not lock.is_file():
                fail(f"campaign is locked: {lock}")
            same_host = False
            alive = False
            try:
                owner = json.loads(lock.read_text(encoding="utf-8"))
                same_host = owner.get("host") == socket.gethostname()
                if same_host:
                    try:
                        os.kill(int(owner["pid"]), 0)
                        alive = True
                    except ProcessLookupError:
                        alive = False
                    except PermissionError:
                        alive = True
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
            if not same_host or alive:
                fail(f"campaign is locked: {lock}")
            lock.unlink()
    else:
        fail(f"could not acquire campaign lock: {lock}")
    try:
        os.write(descriptor, (json.dumps(payload) + "\n").encode("utf-8"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def tuning_manifest(engine: Path, values: dict[str, int] | None = None) -> dict[str, Any]:
    commands = ["uci"]
    for name, value in sorted((values or {}).items()):
        commands.append(f"setoption name {name} value {value}")
    commands.extend(["setoption name TuneManifest", "quit"])
    completed = subprocess.run(
        [str(engine)],
        input="\n".join(commands) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"engine exited {completed.returncode} while reading TuneManifest")
    parameters: dict[str, dict[str, int]] = {}
    checksum = None
    eval_file_default = None
    for line in completed.stdout.splitlines():
        if line.startswith("info string error:"):
            fail(line)
        if line.startswith("info string tuning manifest version 1 checksum "):
            checksum = line.rsplit(" ", 1)[-1]
        if line.startswith("option name EvalFile type string default"):
            eval_file_default = line.removeprefix("option name EvalFile type string default").strip()
        match = PARAMETER_RE.match(line)
        if match:
            name, value, default, minimum, maximum, step = match.groups()
            parameters[name] = {
                "value": int(value),
                "default": int(default),
                "min": int(minimum),
                "max": int(maximum),
                "step": int(step),
            }
    if checksum is None or not parameters:
        fail("engine did not provide a complete TuneManifest; build with spsa-tuning")
    return {
        "version": 1,
        "checksum": checksum,
        "parameters": parameters,
        "eval_file_default": eval_file_default or "",
    }


def evaluator_preflight(engine: Path, eval_file: Path | None) -> dict[str, Any]:
    commands = ["uci"]
    if eval_file is not None:
        commands.append(f"setoption name EvalFile value {eval_file}")
    commands.extend(["isready", "position startpos", "go depth 1"])
    completed = subprocess.run(
        [str(engine)],
        input="\n".join(commands) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    lines = completed.stdout.splitlines()
    errors = [line for line in lines if line.startswith("info string error:")]
    bestmoves = [line for line in lines if line.startswith("bestmove ")]
    if completed.returncode != 0 or errors or "uciok" not in lines or "readyok" not in lines or not bestmoves:
        detail = errors[-1] if errors else f"exit={completed.returncode}"
        fail(f"evaluation preflight failed ({detail}); expected uciok, readyok, and bestmove")
    return {
        "mode": "embedded" if eval_file is None else "external",
        "bestmove": bestmoves[-1].split(maxsplit=1)[1],
        "eval_file_sha256": None if eval_file is None else sha256(eval_file),
    }


def freeze_file(source: Path, destination: Path, executable: bool = False) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    if not source.is_file():
        fail(f"not a file: {source}")
    shutil.copy2(source, destination)
    if executable:
        destination.chmod(destination.stat().st_mode | 0o111)
    return {
        "path": str(destination.relative_to(destination.parents[1])),
        "sha256": sha256(destination),
        "size": destination.stat().st_size,
    }


def resolve_asset(lab: Path, record: dict[str, Any]) -> Path:
    path = lab / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        fail(f"frozen asset changed or disappeared: {path}")
    return path


def selected_names(raw: str, schema: dict[str, Any]) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        fail("--parameters must select at least one Tune parameter")
    if len(names) != len(set(names)):
        fail("--parameters contains duplicates")
    unknown = [name for name in names if name not in schema["parameters"]]
    if unknown:
        fail(f"unknown tuning parameter(s): {', '.join(unknown)}")
    return names


def count_openings(path: Path, book_format: str) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if book_format == "pgn":
            count = sum(1 for line in handle if line.startswith("[Event "))
        else:
            count = sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))
    if count < 1:
        fail(f"opening book contains no {book_format.upper()} records: {path}")
    return count


def create_lab(args: argparse.Namespace) -> Path:
    lab = args.output.expanduser().absolute()
    if "\r" in str(lab) or "\n" in str(lab):
        fail("output path must contain no CR/LF characters")
    if lab.exists() or lab.is_symlink():
        fail(f"output already exists: {lab}")
    if args.iterations < 1 or args.pairs_per_iteration < 1:
        fail("iterations and pairs-per-iteration must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        fail("learning-rate must be finite and positive")
    if not (1 <= args.concurrency <= 64):
        fail("concurrency must be between 1 and 64")
    if not (1 <= args.threads <= 64) or not (1 <= args.hash_mb <= 512):
        fail("Threads must be 1..64 and Hash must be 1..512")
    if not (0 <= args.move_overhead_ms <= 5_000):
        fail("move-overhead-ms must be between 0 and 5000")
    if not (0 <= args.time_margin_ms <= 60_000):
        fail("time-margin-ms must be between 0 and 60000")
    if not args.tc.strip() or "\r" in args.tc or "\n" in args.tc:
        fail("tc must be non-empty and contain no CR/LF characters")
    lab.mkdir(parents=True)
    assets = lab / "assets"
    assets.mkdir()
    engine_suffix = ".exe" if args.engine.suffix.lower() == ".exe" else ""
    fastchess_suffix = ".exe" if args.fastchess.suffix.lower() == ".exe" else ""
    engine_record = freeze_file(args.engine, assets / f"engine{engine_suffix}", executable=True)
    fastchess_record = freeze_file(
        args.fastchess, assets / f"fastchess{fastchess_suffix}", executable=True
    )
    book_suffix = args.book.suffix.lower()
    if args.book_format is None and book_suffix not in (".epd", ".pgn"):
        fail("--book-format is required unless the book ends in .epd or .pgn")
    if not book_suffix:
        book_suffix = f".{args.book_format}"
    book_record = freeze_file(args.book, assets / f"openings{book_suffix}")
    engine = resolve_asset(lab, engine_record)
    schema = tuning_manifest(engine)
    names = selected_names(args.parameters, schema)
    evaluation: dict[str, Any]
    if args.evalfile == "embedded":
        match = EMBEDDED_EVAL_RE.match(schema["eval_file_default"])
        if not match:
            fail(
                "--evalfile embedded requires an engine whose advertised EvalFile is "
                "<embedded:sha256:size>"
            )
        digest, size = match.groups()
        evaluation = {
            "mode": "embedded",
            "sha256": digest,
            "size": int(size),
            "preflight": evaluator_preflight(engine, None),
        }
    else:
        eval_record = freeze_file(Path(args.evalfile), assets / "eval.nnue")
        frozen_eval = resolve_asset(lab, eval_record)
        evaluation = {
            "mode": "external",
            "asset": eval_record,
            "preflight": evaluator_preflight(engine, frozen_eval),
        }
    book_format = args.book_format or book_suffix.removeprefix(".")
    book = resolve_asset(lab, book_record)
    book_openings = count_openings(book, book_format)
    required_openings = args.iterations * args.pairs_per_iteration
    if required_openings > book_openings:
        fail(
            f"campaign needs {required_openings} disjoint openings but book has {book_openings}"
        )
    vector = {name: float(schema["parameters"][name]["default"]) for name in names}
    manifest = {
        "schema": SCHEMA,
        "assets": {"engine": engine_record, "fastchess": fastchess_record, "book": book_record},
        "evaluation": evaluation,
        "tuning_schema": schema,
        "parameters": names,
        "config": {
            "iterations": args.iterations,
            "pairs_per_iteration": args.pairs_per_iteration,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "tc": args.tc,
            "concurrency": args.concurrency,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "move_overhead_ms": args.move_overhead_ms,
            "book_format": book_format,
            "book_openings": book_openings,
            "time_margin_ms": args.time_margin_ms,
        },
    }
    manifest_path = lab / "manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_text(lab / "manifest.sha256", f"{sha256(manifest_path)}  manifest.json\n")
    atomic_json(
        lab / "checkpoint.json",
        {"schema": SCHEMA, "next_iteration": 0, "vector": vector, "history": []},
    )
    return lab


def load_lab(lab: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lab = lab.expanduser().resolve(strict=True)
    manifest_path = lab / "manifest.json"
    digest_path = lab / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        fail("campaign manifest or manifest checksum is missing")
    recorded_digest = digest_path.read_text(encoding="utf-8").split()[0]
    if recorded_digest != sha256(manifest_path):
        fail("campaign manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = json.loads((lab / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or checkpoint.get("schema") != SCHEMA:
        fail("unsupported or corrupt local SPSA schema")
    for record in manifest["assets"].values():
        resolve_asset(lab, record)
    if manifest["evaluation"]["mode"] == "external":
        eval_file = resolve_asset(lab, manifest["evaluation"]["asset"])
    else:
        eval_file = None
    if evaluator_preflight(
        resolve_asset(lab, manifest["assets"]["engine"]), eval_file
    ) != manifest["evaluation"]["preflight"]:
        fail("evaluation preflight identity no longer matches the campaign manifest")
    current_schema = tuning_manifest(resolve_asset(lab, manifest["assets"]["engine"]))
    if current_schema != manifest["tuning_schema"]:
        fail("frozen engine TuneManifest no longer matches the campaign manifest")
    return manifest, checkpoint


def bounded(value: float, spec: dict[str, int]) -> int:
    return max(spec["min"], min(spec["max"], int(round(value))))


def vectors_for_iteration(
    manifest: dict[str, Any], checkpoint: dict[str, Any], iteration: int
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, float],
]:
    rng = random.Random(manifest["config"]["seed"] + iteration)
    plus: dict[str, int] = {}
    minus: dict[str, int] = {}
    deltas: dict[str, int] = {}
    spans: dict[str, int] = {}
    half_radii: dict[str, float] = {}
    for name in manifest["parameters"]:
        spec = manifest["tuning_schema"]["parameters"][name]
        delta = 1 if rng.getrandbits(1) else -1
        radius = max(1, int(round(spec["step"] / ((iteration + 1) ** 0.101))))
        center = float(checkpoint["vector"][name])
        plus[name] = bounded(center + radius * delta, spec)
        minus[name] = bounded(center - radius * delta, spec)
        deltas[name] = delta
        span = plus[name] - minus[name]
        if span == 0:
            fail(f"parameter {name} has a zero perturbation span at iteration {iteration}")
        spans[name] = span
        half_radii[name] = abs(span) / 2.0
    return plus, minus, deltas, spans, half_radii


def engine_arguments(name: str, engine: Path, config: dict[str, Any], vector: dict[str, int]) -> list[str]:
    arguments = [
        "-engine",
        f"name={name}",
        f"cmd={engine}",
        f"dir={engine.parent}",
        f"option.Threads={config['threads']}",
        f"option.Hash={config['hash_mb']}",
        f"option.Move Overhead={config['move_overhead_ms']}",
    ]
    arguments.extend(f"option.{key}={value}" for key, value in sorted(vector.items()))
    return arguments


def fastchess_command(
    lab: Path,
    manifest: dict[str, Any],
    iteration_dir: Path,
    plus: dict[str, int],
    minus: dict[str, int],
    book_start: int,
) -> list[str]:
    config = manifest["config"]
    fastchess = resolve_asset(lab, manifest["assets"]["fastchess"])
    engine = resolve_asset(lab, manifest["assets"]["engine"])
    book = resolve_asset(lab, manifest["assets"]["book"])
    command = [
        str(fastchess), "-recover", "-repeat", "-games", "2", "-rounds",
        str(config["pairs_per_iteration"]), "-strict", "-ratinginterval", "1",
        "-scoreinterval", "1", "-autosaveinterval", "2", "-report", "penta=true",
        "-variant", "standard", "-concurrency", str(config["concurrency"]),
        "-openings", f"file={book}", f"format={config['book_format']}", "order=sequential",
        f"start={book_start}",
    ]
    command.extend(engine_arguments("SPSA-plus", engine, config, plus))
    if manifest["evaluation"]["mode"] == "external":
        eval_file = resolve_asset(lab, manifest["evaluation"]["asset"])
        command.append(f"option.EvalFile={eval_file}")
    command.extend(engine_arguments("SPSA-minus", engine, config, minus))
    if manifest["evaluation"]["mode"] == "external":
        eval_file = resolve_asset(lab, manifest["evaluation"]["asset"])
        command.append(f"option.EvalFile={eval_file}")
    command.extend([
        "-each", f"tc={config['tc']}", "proto=uci", f"timemargin={config['time_margin_ms']}",
        "-pgnout", f"file={iteration_dir / 'games.pgn'}", "append=false",
        "-log", f"file={iteration_dir / 'fastchess.log'}", "level=info", "engine=true", "append=false",
        "-config", f"outname={iteration_dir / 'recovery.json'}",
    ])
    return command


def score_statistics(wins: int, losses: int, draws: int) -> dict[str, Any]:
    games = wins + losses + draws
    if games == 0:
        fail("FastChess reported zero games")
    score = (wins + draws / 2.0) / games
    mean_square = (wins + draws / 4.0) / games
    score_se = math.sqrt(max(0.0, mean_square - score * score) / games)
    clipped = min(1.0 - 1e-9, max(1e-9, score))
    elo = 400.0 * math.log10(clipped / (1.0 - clipped))
    elo_se = (400.0 / math.log(10.0)) * score_se / (clipped * (1.0 - clipped))
    return {
        "wins": wins, "losses": losses, "draws": draws, "games": games,
        "score": score, "score_se": score_se, "elo": elo, "elo_se": elo_se,
        "signal": (wins - losses) / games,
    }


def parse_score(text: str) -> dict[str, Any]:
    matches = SCORE_RE.findall(text)
    if not matches:
        matches = RESULTS_RE.findall(text)
    if not matches:
        fail("FastChess output did not contain a parseable SPSA score")
    return score_statistics(*map(int, matches[-1]))


def iteration_evidence(iteration_dir: Path, expected_games: int) -> dict[str, Any]:
    pgn = iteration_dir / "games.pgn"
    log = iteration_dir / "fastchess.log"
    recovery = iteration_dir / "recovery.json"
    for path in (pgn, log, recovery):
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"iteration evidence is missing or empty: {path}")
    games: list[dict[str, str]] = []
    tags: dict[str, str] = {}
    for line in pgn.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TAG_RE.match(line)
        if not match:
            continue
        if match.group(1) == "Event" and tags:
            if "Result" in tags:
                games.append(tags)
            tags = {}
        tags[match.group(1)] = match.group(2)
    if tags and "Result" in tags:
        games.append(tags)
    if len(games) != expected_games:
        fail(f"PGN has {len(games)} complete games; expected {expected_games}")
    wins = losses = draws = 0
    plus_white = plus_black = 0
    round_pairs: dict[str, list[tuple[str, str]]] = {}
    terminations: dict[str, int] = {}
    failures: dict[str, int] = {}
    for game in games:
        result = game.get("Result")
        white, black = game.get("White"), game.get("Black")
        if result not in {"1-0", "0-1", "1/2-1/2"}:
            fail(f"PGN contains unfinished or invalid result {result!r}")
        if {white, black} != {"SPSA-plus", "SPSA-minus"}:
            fail("PGN game does not contain the exact SPSA-plus/SPSA-minus engine pair")
        round_tag = game.get("Round", "").strip()
        if not round_tag:
            fail("PGN game is missing a non-empty Round tag")
        round_pairs.setdefault(round_tag, []).append((white, black))
        plus_white += int(white == "SPSA-plus")
        plus_black += int(black == "SPSA-plus")
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0" and white == "SPSA-plus") or (
            result == "0-1" and black == "SPSA-plus"
        ):
            wins += 1
        else:
            losses += 1
        termination = game.get("Termination", "unknown").strip().lower()
        terminations[termination] = terminations.get(termination, 0) + 1
        if termination not in EXPECTED_TERMINATIONS:
            category = FAILURE_CLASSES.get(termination, "unknown_abnormal")
            failures[category] = failures.get(category, 0) + 1
    if failures:
        fail(f"PGN contains abnormal engine terminations: {failures}")
    if plus_white != plus_black:
        fail(
            f"PGN is not color reversed: SPSA-plus has {plus_white} white and "
            f"{plus_black} black games"
        )
    if len(round_pairs) != expected_games // 2:
        fail(
            f"PGN has {len(round_pairs)} opening rounds; expected {expected_games // 2}"
        )
    for round_tag, colors in round_pairs.items():
        if len(colors) != 2:
            fail(f"PGN Round {round_tag!r} has {len(colors)} games; expected exactly two")
        if sorted(colors) != [
            ("SPSA-minus", "SPSA-plus"),
            ("SPSA-plus", "SPSA-minus"),
        ]:
            fail(f"PGN Round {round_tag!r} is not an opposite-color engine pair")
    return {
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "terminations": dict(sorted(terminations.items())),
        "failures": {},
        "artifacts": {
            path.name: {"sha256": sha256(path), "size": path.stat().st_size}
            for path in (pgn, log, recovery)
        },
    }


def validate_cached_result(
    result: dict[str, Any],
    iteration: int,
    plus: dict[str, int],
    minus: dict[str, int],
    expected_games: int,
    iteration_dir: Path,
) -> dict[str, Any]:
    if result.get("iteration") != iteration or result.get("plus") != plus or result.get("minus") != minus:
        fail(f"cached result metadata does not match iteration {iteration}")
    counts = [result.get(name) for name in ("wins", "losses", "draws")]
    if any(type(value) is not int or value < 0 for value in counts):
        fail(f"cached result has invalid W/D/L counts at iteration {iteration}")
    statistics = score_statistics(counts[0], counts[1], counts[2])
    if statistics["games"] != expected_games:
        fail(
            f"cached result at iteration {iteration} has {statistics['games']} games; "
            f"expected {expected_games}"
        )
    evidence = iteration_evidence(iteration_dir, expected_games)
    if [evidence[key] for key in ("wins", "losses", "draws")] != counts:
        fail(f"cached console result disagrees with PGN at iteration {iteration}")
    return {
        **statistics,
        "iteration": iteration,
        "plus": plus,
        "minus": minus,
        "evidence": evidence,
    }


def complete_iteration(lab: Path, manifest: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    iteration = int(checkpoint["next_iteration"])
    plus, minus, deltas, spans, half_radii = vectors_for_iteration(
        manifest, checkpoint, iteration
    )
    iteration_dir = lab / "iterations" / f"{iteration:05d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    config = manifest["config"]
    required_openings = config["iterations"] * config["pairs_per_iteration"]
    first_start = 1 + manifest["config"]["seed"] % (
        config["book_openings"] - required_openings + 1
    )
    book_start = first_start + iteration * config["pairs_per_iteration"]
    command = fastchess_command(lab, manifest, iteration_dir, plus, minus, book_start)
    atomic_json(iteration_dir / "experiment.json", {
        "iteration": iteration, "plus": plus, "minus": minus, "deltas": deltas,
        "spans": spans,
        "half_radii": half_radii,
        "book_start": book_start,
        "command": command,
    })
    result_path = iteration_dir / "result.json"
    expected_games = 2 * manifest["config"]["pairs_per_iteration"]
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result = validate_cached_result(
            result, iteration, plus, minus, expected_games, iteration_dir
        )
    else:
        recovery = iteration_dir / "recovery.json"
        run_command = command
        if recovery.is_file():
            run_command = [
                str(resolve_asset(lab, manifest["assets"]["fastchess"])),
                "-config",
                f"file={recovery}",
                "stats=true",
            ]
            atomic_json(iteration_dir / "resume-command.json", run_command)
        with (iteration_dir / "console.log").open("a", encoding="utf-8") as output:
            completed = subprocess.run(
                run_command,
                text=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            fail(f"FastChess iteration {iteration} exited {completed.returncode}; rerun resume")
        console = (iteration_dir / "console.log").read_text(encoding="utf-8")
        result = parse_score(console)
        if result["games"] != expected_games:
            fail(
                f"FastChess iteration {iteration} reported {result['games']} games; "
                f"expected {expected_games}"
            )
        result.update({"iteration": iteration, "plus": plus, "minus": minus})
        evidence = iteration_evidence(iteration_dir, expected_games)
        if (result["wins"], result["losses"], result["draws"]) != (
            evidence["wins"],
            evidence["losses"],
            evidence["draws"],
        ):
            fail(f"FastChess console and PGN W/D/L disagree at iteration {iteration}")
        result["evidence"] = evidence
        atomic_json(result_path, result)

    gain = manifest["config"]["learning_rate"] / ((iteration + 1) ** 0.602)
    vector = dict(checkpoint["vector"])
    updates: dict[str, float] = {}
    for name in manifest["parameters"]:
        spec = manifest["tuning_schema"]["parameters"][name]
        gradient = result["signal"] / spans[name]
        update = gain * (spec["step"] ** 2) * gradient
        vector[name] = float(max(spec["min"], min(spec["max"], float(vector[name]) + update)))
        updates[name] = update
    history = list(checkpoint["history"])
    history.append(
        {
            "iteration": iteration,
            "result": result,
            "updates": updates,
            "spans": spans,
            "half_radii": half_radii,
        }
    )
    next_checkpoint = {"schema": SCHEMA, "next_iteration": iteration + 1, "vector": vector, "history": history}
    atomic_json(lab / "checkpoint.json", next_checkpoint)
    write_recommendation(lab, manifest, next_checkpoint)
    return next_checkpoint


def write_recommendation(lab: Path, manifest: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    vector = {
        name: bounded(float(checkpoint["vector"][name]), manifest["tuning_schema"]["parameters"][name])
        for name in manifest["parameters"]
    }
    last = checkpoint["history"][-1] if checkpoint["history"] else None
    recommendation = {
        "schema": SCHEMA,
        "completed_iterations": checkpoint["next_iteration"],
        "total_games": sum(entry["result"]["games"] for entry in checkpoint["history"]),
        "recommended_vector": vector,
        "manifest": tuning_manifest(resolve_asset(lab, manifest["assets"]["engine"]), vector),
        "last_comparison": None if last is None else last["result"],
        "parameter_resolution": None if last is None else last["half_radii"],
        "warning": "Experimental SPSA output; promote only after independent paired SPRT validation.",
    }
    atomic_json(lab / "recommended.json", recommendation)


def run_lab(lab: Path) -> None:
    lab = lab.expanduser().resolve(strict=True)
    with campaign_lock(lab):
        manifest, checkpoint = load_lab(lab)
        iterations = manifest["config"]["iterations"]
        while checkpoint["next_iteration"] < iterations:
            checkpoint = complete_iteration(lab, manifest, checkpoint)
            result = checkpoint["history"][-1]["result"]
            print(
                f"iteration {result['iteration'] + 1}/{iterations}: "
                f"{result['wins']}W/{result['draws']}D/{result['losses']}L "
                f"score {result['score']:.4f} +/- {result['score_se']:.4f}",
                flush=True,
            )
        print(f"recommended vector: {lab / 'recommended.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    for name in ("fastchess", "engine", "book", "output"):
        start.add_argument(f"--{name}", type=Path, required=True)
    start.add_argument(
        "--evalfile",
        required=True,
        help="'embedded' or an explicit network path that will be frozen",
    )
    start.add_argument("--parameters", required=True)
    start.add_argument("--iterations", type=int, default=20)
    start.add_argument("--pairs-per-iteration", type=int, default=64)
    start.add_argument("--seed", type=int, default=0x564F4C4B)
    start.add_argument("--learning-rate", type=float, default=1.0)
    start.add_argument("--tc", default="10+0.1")
    start.add_argument("--concurrency", type=int, default=1)
    start.add_argument("--threads", type=int, default=1)
    start.add_argument("--hash-mb", type=int, default=64)
    start.add_argument("--move-overhead-ms", type=int, default=10)
    start.add_argument("--time-margin-ms", type=int, default=1000)
    start.add_argument("--book-format", choices=("epd", "pgn"))
    resume = commands.add_parser("resume")
    resume.add_argument("--lab", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--lab", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if args.command == "start":
            lab = create_lab(args)
            run_lab(lab)
        elif args.command == "resume":
            run_lab(args.lab)
        else:
            lab = args.lab.expanduser().resolve(strict=True)
            with campaign_lock(lab):
                manifest, checkpoint = load_lab(lab)
                write_recommendation(lab, manifest, checkpoint)
                print((lab / "recommended.json").read_text(encoding="utf-8"), end="")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
