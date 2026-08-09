#!/usr/bin/env python3
"""Fail-closed UCI process smoke test used by CI and release packaging."""

from __future__ import annotations

import argparse
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


REQUIRED_OPTIONS = (
    "Hash",
    "Threads",
    "Move Overhead",
    "SyzygyPath",
    "SyzygyProbeLimit",
    "Syzygy50MoveRule",
    "EvalFile",
)
STARTPOS_MOVES = {
    f"{file_name}2{file_name}{rank}"
    for file_name in "abcdefgh"
    for rank in ("3", "4")
} | {"b1a3", "b1c3", "g1f3", "g1h3"}
BESTMOVE_RE = re.compile(r"^bestmove\s+(\S+)(?:\s+ponder\s+\S+)?$")


class SmokeFailure(RuntimeError):
    pass


@dataclass
class EngineProcess:
    process: subprocess.Popen[str]
    stdout_lines: queue.Queue[str | None]
    stderr_lines: queue.Queue[str | None]
    transcript: list[str]

    @classmethod
    def start(cls, engine: pathlib.Path) -> "EngineProcess":
        try:
            process = subprocess.Popen(
                [str(engine)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as error:
            raise SmokeFailure(f"failed to launch {engine}: {error}") from error

        assert process.stdout is not None
        assert process.stderr is not None
        stdout_lines: queue.Queue[str | None] = queue.Queue()
        stderr_lines: queue.Queue[str | None] = queue.Queue()
        transcript: list[str] = []

        def read_stream(
            stream: object, destination: queue.Queue[str | None], label: str
        ) -> None:
            try:
                for raw_line in stream:  # type: ignore[union-attr]
                    line = raw_line.rstrip("\r\n")
                    transcript.append(f"[{label}] {line}")
                    destination.put(line)
            finally:
                destination.put(None)

        threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_lines, "stdout"),
            daemon=True,
        ).start()
        threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_lines, "stderr"),
            daemon=True,
        ).start()
        return cls(process, stdout_lines, stderr_lines, transcript)

    def send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise SmokeFailure(
                f"engine exited with code {self.process.returncode} before '{command}'"
            )
        assert self.process.stdin is not None
        self.transcript.append(f"[stdin] {command}")
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise SmokeFailure(f"failed to send '{command}': {error}") from error

    def read_until(self, predicate: object, description: str, timeout: float) -> list[str]:
        deadline = time.monotonic() + timeout
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SmokeFailure(f"timed out waiting for {description}")
            try:
                line = self.stdout_lines.get(timeout=remaining)
            except queue.Empty as error:
                raise SmokeFailure(f"timed out waiting for {description}") from error
            if line is None:
                raise SmokeFailure(
                    f"engine closed stdout while waiting for {description}; "
                    f"exit={self.process.poll()}"
                )
            if line.lower().startswith("info string error"):
                raise SmokeFailure(f"engine reported a UCI error: {line}")
            lines.append(line)
            if predicate(line):  # type: ignore[operator]
                return lines

    def close(self, timeout: float) -> None:
        try:
            if self.process.poll() is None:
                self.send("quit")
            try:
                return_code = self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=timeout)
                raise SmokeFailure("engine did not exit promptly after 'quit'") from error
            if return_code != 0:
                raise SmokeFailure(f"engine exited with code {return_code} after 'quit'")
        except SmokeFailure:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=timeout)
            raise
        finally:
            self.close_pipes()

    def close_pipes(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def existing_file(value: str, label: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_file():
        raise SmokeFailure(f"{label} is not a file: {path}")
    return path


def existing_directory(value: str, label: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_dir():
        raise SmokeFailure(f"{label} is not a directory: {path}")
    return path


def run_smoke(
    engine: pathlib.Path,
    eval_file: str,
    timeout: float,
    depth: int,
    threads: int,
    hash_mb: int,
    small_eval_file: str | None = None,
    dual_policy: str = "off",
    dual_threshold: int = 200,
    move_overhead_ms: int = 10,
    syzygy_path: str | None = None,
    syzygy_probe_limit: int = 7,
    syzygy_50_move_rule: str = "true",
) -> list[str]:
    evaluator = "classical"
    eval_value = ""
    use_default_evaluator = eval_file == "embedded"
    if use_default_evaluator:
        evaluator = "embedded"
    elif eval_file != "classical":
        evaluator_path = existing_file(eval_file, "EvalFile")
        evaluator = str(evaluator_path)
        eval_value = evaluator
    small_evaluator = None
    if small_eval_file:
        small_evaluator = str(existing_file(small_eval_file, "SmallEvalFile"))
    tablebase_path = ""
    if syzygy_path:
        tablebase_path = str(existing_directory(syzygy_path, "SyzygyPath"))

    session = EngineProcess.start(engine)
    try:
        session.send("uci")
        handshake = session.read_until(lambda line: line == "uciok", "uciok", timeout)
        for option in REQUIRED_OPTIONS:
            prefix = f"option name {option} "
            if not any(line.startswith(prefix) for line in handshake):
                raise SmokeFailure(f"engine did not advertise required option '{option}'")
        if use_default_evaluator and not any(
            line.startswith("option name EvalFile type string default <embedded:")
            for line in handshake
        ):
            raise SmokeFailure("engine did not advertise an embedded default EvalFile")
        if small_evaluator is not None:
            for option in ("SmallEvalFile", "DualEvalPolicy", "DualEvalThreshold"):
                prefix = f"option name {option} "
                if not any(line.startswith(prefix) for line in handshake):
                    raise SmokeFailure(
                        f"engine did not advertise required dual option '{option}'"
                    )

        settings = [
            ("Threads", str(threads)),
            ("Hash", str(hash_mb)),
            ("Move Overhead", str(move_overhead_ms)),
            ("SyzygyPath", tablebase_path),
            ("SyzygyProbeLimit", str(syzygy_probe_limit)),
            ("Syzygy50MoveRule", syzygy_50_move_rule),
        ]
        if not use_default_evaluator:
            settings.append(("EvalFile", eval_value))
        for name, value in settings:
            session.send(f"setoption name {name} value {value}".rstrip())
        if small_evaluator is not None:
            session.send(f"setoption name SmallEvalFile value {small_evaluator}")
            session.send(f"setoption name DualEvalThreshold value {dual_threshold}")
            session.send(f"setoption name DualEvalPolicy value {dual_policy}")
        session.send("isready")
        session.read_until(lambda line: line == "readyok", "readyok", timeout)

        session.send("ucinewgame")
        session.send("isready")
        session.read_until(lambda line: line == "readyok", "readyok after ucinewgame", timeout)
        session.send("position startpos")
        session.send(f"go depth {depth}")
        search = session.read_until(
            lambda line: line.startswith("bestmove "), "finite-search bestmove", timeout
        )
        bestmove_line = search[-1]
        match = BESTMOVE_RE.fullmatch(bestmove_line)
        if match is None:
            raise SmokeFailure(f"malformed bestmove response: {bestmove_line}")
        if match.group(1) not in STARTPOS_MOVES:
            raise SmokeFailure(f"illegal start-position bestmove: {match.group(1)}")
        if not any(line.startswith("info ") for line in search):
            raise SmokeFailure("finite search produced no UCI info line")

        session.send("position startpos")
        session.send("go infinite")
        time.sleep(min(0.05, timeout / 4))
        session.send("stop")
        session.read_until(
            lambda line: line.startswith("bestmove "), "bestmove after stop", timeout
        )
        session.close(timeout)
    except Exception:
        if session.process.poll() is None:
            session.process.kill()
            session.process.wait(timeout=timeout)
        session.close_pipes()
        raise

    session.transcript.append(f"[smoke] evaluator={evaluator}")
    if small_evaluator is not None:
        session.transcript.append(
            f"[smoke] small_evaluator={small_evaluator} "
            f"dual_policy={dual_policy} dual_threshold={dual_threshold}"
        )
    return session.transcript


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="engine executable")
    parser.add_argument(
        "--evalfile",
        required=True,
        help="'classical', 'embedded', or a network file",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--move-overhead-ms", type=int, default=10)
    parser.add_argument("--syzygy-path")
    parser.add_argument("--syzygy-probe-limit", type=int, default=7)
    parser.add_argument(
        "--syzygy-50-move-rule", choices=("true", "false"), default="true"
    )
    parser.add_argument("--small-evalfile")
    parser.add_argument(
        "--dual-policy", choices=("off", "small-fallback"), default="off"
    )
    parser.add_argument("--dual-threshold", type=int, default=200)
    parser.add_argument("--transcript", help="optional transcript output path")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.depth <= 0 or args.threads <= 0 or args.hash_mb <= 0:
        parser.error("--depth, --threads, and --hash-mb must be positive")
    if args.depth > 127:
        parser.error("--depth must be at most 127")
    if args.threads > 64:
        parser.error("--threads must be at most 64")
    if args.hash_mb > 512:
        parser.error("--hash-mb must be at most 512")
    if not 0 <= args.move_overhead_ms <= 5000:
        parser.error("--move-overhead-ms must be between 0 and 5000")
    if not 0 <= args.syzygy_probe_limit <= 7:
        parser.error("--syzygy-probe-limit must be between 0 and 7")
    if not 0 <= args.dual_threshold <= 2000:
        parser.error("--dual-threshold must be between 0 and 2000")
    if args.dual_policy == "small-fallback" and not args.small_evalfile:
        parser.error("--dual-policy small-fallback requires --small-evalfile")
    if args.dual_policy == "small-fallback" and args.evalfile in {"classical", "embedded"}:
        parser.error("dual evaluation requires a network --evalfile")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        engine = existing_file(args.engine, "engine")
        transcript = run_smoke(
            engine,
            args.evalfile,
            args.timeout,
            args.depth,
            args.threads,
            args.hash_mb,
            small_eval_file=args.small_evalfile,
            dual_policy=args.dual_policy,
            dual_threshold=args.dual_threshold,
            move_overhead_ms=args.move_overhead_ms,
            syzygy_path=args.syzygy_path,
            syzygy_probe_limit=args.syzygy_probe_limit,
            syzygy_50_move_rule=args.syzygy_50_move_rule,
        )
        if args.transcript:
            transcript_path = pathlib.Path(args.transcript)
            transcript_path.write_text("\n".join(transcript) + "\n", encoding="utf-8")
        print(f"UCI smoke passed: {engine}")
        print(transcript[-1])
        return 0
    except SmokeFailure as error:
        print(f"UCI smoke failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
