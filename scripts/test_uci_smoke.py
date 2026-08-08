#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("uci_smoke", SCRIPT_DIR / "uci_smoke.py")
assert SPEC is not None and SPEC.loader is not None
uci_smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = uci_smoke
SPEC.loader.exec_module(uci_smoke)


FIXTURE = r'''#!/usr/bin/env python3
import sys

mode = __MODE__
for raw in sys.stdin:
    command = raw.strip()
    if command == "uci":
        for option in (
            "Hash",
            "Threads",
            "Move Overhead",
            "SyzygyPath",
            "SyzygyProbeLimit",
            "Syzygy50MoveRule",
            "EvalFile",
            "SmallEvalFile",
            "DualEvalPolicy",
            "DualEvalThreshold",
        ):
            print(f"option name {option} type string default", flush=True)
        print("uciok", flush=True)
    elif command.startswith("setoption") and mode == "reject":
        print("info string error: injected rejection", flush=True)
        mode = "ok"
    elif command == "isready":
        print("readyok", flush=True)
    elif command.startswith("go depth"):
        print("info depth 1 score cp 0 nodes 1", flush=True)
        print("bestmove " + ("e2e4" if mode != "illegal" else "a1a8"), flush=True)
    elif command == "stop":
        print("bestmove e2e4", flush=True)
    elif command == "quit":
        raise SystemExit(7 if mode == "exit-nonzero" else 0)
'''


class UciSmokeTests(unittest.TestCase):
    def fixture(self, root: pathlib.Path, mode: str) -> pathlib.Path:
        if sys.platform == "win32":
            self.skipTest("executable script fixture uses a POSIX shebang")
        path = root / f"engine-{mode}"
        source = FIXTURE.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1)
        path.write_text(source.replace("__MODE__", repr(mode)), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_accepts_compliant_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine = self.fixture(pathlib.Path(temporary), "ok")
            transcript = uci_smoke.run_smoke(engine, "classical", 2.0, 1, 1, 1)
            self.assertIn("[smoke] evaluator=classical", transcript)
            self.assertIn(
                "[stdin] setoption name SyzygyProbeLimit value 7", transcript
            )
            self.assertIn(
                "[stdin] setoption name Syzygy50MoveRule value true", transcript
            )

    def test_propagates_requested_runtime_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            engine = self.fixture(root, "ok")
            tablebases = root / "syzygy"
            tablebases.mkdir()
            transcript = uci_smoke.run_smoke(
                engine,
                "classical",
                2.0,
                1,
                1,
                1,
                move_overhead_ms=37,
                syzygy_path=str(tablebases),
                syzygy_probe_limit=5,
                syzygy_50_move_rule="false",
            )
            self.assertIn("[stdin] setoption name Move Overhead value 37", transcript)
            self.assertIn(
                f"[stdin] setoption name SyzygyPath value {tablebases.resolve()}",
                transcript,
            )
            self.assertIn(
                "[stdin] setoption name SyzygyProbeLimit value 5", transcript
            )
            self.assertIn(
                "[stdin] setoption name Syzygy50MoveRule value false", transcript
            )

    def test_rejects_nonzero_engine_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine = self.fixture(pathlib.Path(temporary), "exit-nonzero")
            with self.assertRaisesRegex(uci_smoke.SmokeFailure, "exited with code 7"):
                uci_smoke.run_smoke(engine, "classical", 2.0, 1, 1, 1)

    def test_rejects_engine_reported_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine = self.fixture(pathlib.Path(temporary), "reject")
            with self.assertRaisesRegex(uci_smoke.SmokeFailure, "injected rejection"):
                uci_smoke.run_smoke(engine, "classical", 2.0, 1, 1, 1)

    def test_configures_explicit_dual_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            engine = self.fixture(root, "ok")
            big = root / "big.nnue"
            small = root / "small.nnue"
            big.write_bytes(b"big fixture")
            small.write_bytes(b"small fixture")
            transcript = uci_smoke.run_smoke(
                engine,
                str(big),
                2.0,
                1,
                1,
                1,
                small_eval_file=str(small),
                dual_policy="small-fallback",
                dual_threshold=300,
            )
            self.assertIn(
                f"[stdin] setoption name SmallEvalFile value {small.resolve()}", transcript
            )
            self.assertIn(
                "[stdin] setoption name DualEvalThreshold value 300", transcript
            )
            self.assertIn(
                "[stdin] setoption name DualEvalPolicy value small-fallback",
                transcript,
            )

    def test_rejects_illegal_bestmove(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine = self.fixture(pathlib.Path(temporary), "illegal")
            with self.assertRaisesRegex(uci_smoke.SmokeFailure, "illegal start-position"):
                uci_smoke.run_smoke(engine, "classical", 2.0, 1, 1, 1)


class StrengthWrapperTests(unittest.TestCase):
    def test_dry_run_is_paired_hashed_and_immutable(self) -> None:
        if sys.platform == "win32":
            self.skipTest("FastChess wrapper requires Bash")
        true_binary = shutil.which("true")
        self.assertIsNotNone(true_binary)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            book = root / "openings.epd"
            book.write_text("startpos\n", encoding="utf-8")
            output = root / "run"
            command = [
                str(SCRIPT_DIR / "run-strength-sprt.sh"),
                "--fastchess",
                true_binary,
                "--baseline",
                true_binary,
                "--candidate",
                true_binary,
                "--book",
                str(book),
                "--output-dir",
                str(output),
                "--evalfile",
                "classical",
                "--rounds",
                "2",
                "--move-overhead-ms",
                "37",
                "--syzygy-path",
                str(root),
                "--syzygy-probe-limit",
                "5",
                "--syzygy-50-move-rule",
                "false",
                "--dry-run",
            ]
            first = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            manifest = (output / "manifest.txt").read_text(encoding="utf-8")
            frozen_command = (output / "command.sh").read_text(encoding="utf-8")
            self.assertIn("baseline_sha256=", manifest)
            self.assertIn("candidate_sha256=", manifest)
            self.assertIn("book_sha256=", manifest)
            self.assertIn("-repeat", frozen_command)
            self.assertIn("-games 2", frozen_command)
            self.assertIn("model=normalized", frozen_command)
            self.assertIn("option.EvalFile=", frozen_command)
            self.assertIn("option.Move\\ Overhead=37", frozen_command)
            self.assertIn("option.SyzygyProbeLimit=5", frozen_command)
            self.assertIn("option.Syzygy50MoveRule=false", frozen_command)
            self.assertIn("syzygy_probe_limit=5", manifest)
            self.assertIn("syzygy_50_move_rule=false", manifest)
            self.assertEqual((output / "manifest.txt").stat().st_mode & stat.S_IWUSR, 0)

            second = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

    def test_dual_dry_run_freezes_small_network_options_and_hash(self) -> None:
        if sys.platform == "win32":
            self.skipTest("FastChess wrapper requires Bash")
        true_binary = shutil.which("true")
        self.assertIsNotNone(true_binary)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            book = root / "openings.epd"
            book.write_text("startpos\n", encoding="utf-8")
            network = root / "network.nnue"
            small = root / "small.nnue"
            network.write_bytes(b"big fixture")
            small.write_bytes(b"small fixture")
            output = root / "dual-run"
            command = [
                str(SCRIPT_DIR / "run-strength-sprt.sh"),
                "--fastchess",
                true_binary,
                "--baseline",
                true_binary,
                "--candidate",
                true_binary,
                "--book",
                str(book),
                "--output-dir",
                str(output),
                "--baseline-evalfile",
                "classical",
                "--candidate-evalfile",
                str(network),
                "--candidate-small-evalfile",
                str(small),
                "--candidate-dual-policy",
                "small-fallback",
                "--candidate-dual-threshold",
                "300",
                "--rounds",
                "2",
                "--dry-run",
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = (output / "manifest.txt").read_text(encoding="utf-8")
            frozen_command = (output / "command.sh").read_text(encoding="utf-8")
            self.assertIn("candidate_small_evalfile_sha256=", manifest)
            self.assertIn("candidate_dual_policy=small-fallback", manifest)
            self.assertIn("candidate_dual_threshold=300", manifest)
            self.assertIn("option.SmallEvalFile=", frozen_command)
            self.assertIn("option.DualEvalPolicy=small-fallback", frozen_command)
            self.assertIn("option.DualEvalThreshold=300", frozen_command)


if __name__ == "__main__":
    unittest.main()
