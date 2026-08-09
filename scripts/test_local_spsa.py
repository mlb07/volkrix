#!/usr/bin/env python3

import json
import importlib.util
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("local_spsa.py")
SPEC = importlib.util.spec_from_file_location("volkrix_local_spsa", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
local_spsa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_spsa)


def executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class LocalSpsaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.engine = executable(
            self.root / "engine.exe",
            """#!/usr/bin/env python3
import sys
values = {"TuneA": 10, "TuneB": 20}
print("option name EvalFile type string default <embedded:" + "a" * 64 + ":1234>")
for line in sys.stdin:
    fields = line.strip().split()
    if line.strip() == "uci":
        print("uciok")
    elif line.strip() == "isready":
        print("readyok")
    elif line.startswith("go "):
        print("bestmove e2e4")
    elif line.startswith("setoption name TuneA value "):
        values["TuneA"] = int(fields[-1])
    elif line.startswith("setoption name TuneB value "):
        values["TuneB"] = int(fields[-1])
    elif line.startswith("setoption name TuneManifest"):
        print("info string tuning manifest version 1 checksum fake")
        print(f"info string tuning parameter TuneA value {values['TuneA']} default 10 min 0 max 30 step 4")
        print(f"info string tuning parameter TuneB value {values['TuneB']} default 20 min 10 max 40 step 5")
        print("info string tuning manifest end")
    elif line.strip() == "quit":
        break
""",
        )
        self.fastchess = executable(
            self.root / "fastchess.exe",
            r"""#!/usr/bin/env python3
import pathlib, sys
rounds = int(sys.argv[sys.argv.index('-rounds') + 1])
pgn = pathlib.Path(sys.argv[sys.argv.index('-pgnout') + 1].split('=', 1)[1])
log = pathlib.Path(sys.argv[sys.argv.index('-log') + 1].split('=', 1)[1])
recovery = pathlib.Path(sys.argv[sys.argv.index('-config') + 1].split('=', 1)[1])
games = []
for opening in range(1, rounds + 1):
    games.append(f'[Event "test"]\n[Round "{opening}"]\n[White "SPSA-plus"]\n[Black "SPSA-minus"]\n[Result "1-0"]\n[Termination "normal"]\n\n1-0\n')
    games.append(f'[Event "test"]\n[Round "{opening}"]\n[White "SPSA-minus"]\n[Black "SPSA-plus"]\n[Result "1/2-1/2"]\n[Termination "normal"]\n\n1/2-1/2\n')
pgn.write_text('\n'.join(games))
log.write_text('clean engine log\n')
recovery.write_text('{}')
# Exactly two games per round: deterministic plus advantage.
print(f"Score of SPSA-plus vs SPSA-minus: {rounds} - 0 - {rounds}")
""",
        )
        self.book = self.root / "book.epd"
        self.book.write_text(
            "".join(f"8/8/8/8/8/8/K6k/8 w - - ; id {index}\n" for index in range(8)),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def command(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_campaign_is_bounded_seeded_resumable_and_hash_frozen(self) -> None:
        lab = self.root / "lab"
        completed = self.command(
            "start",
            "--fastchess", str(self.fastchess),
            "--engine", str(self.engine),
            "--book", str(self.book),
            "--evalfile", "embedded",
            "--output", str(lab),
            "--parameters", "TuneA,TuneB",
            "--iterations", "2",
            "--pairs-per-iteration", "2",
            "--seed", "7",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        checkpoint = json.loads((lab / "checkpoint.json").read_text(encoding="utf-8"))
        recommendation = json.loads((lab / "recommended.json").read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["next_iteration"], 2)
        self.assertTrue((lab / "assets" / "engine.exe").is_file())
        self.assertTrue((lab / "assets" / "fastchess.exe").is_file())
        self.assertEqual(recommendation["total_games"], 8)
        self.assertGreaterEqual(recommendation["recommended_vector"]["TuneA"], 0)
        self.assertLessEqual(recommendation["recommended_vector"]["TuneA"], 30)
        self.assertGreaterEqual(recommendation["recommended_vector"]["TuneB"], 10)
        self.assertLessEqual(recommendation["recommended_vector"]["TuneB"], 40)
        first_experiment = json.loads(
            (lab / "iterations" / "00000" / "experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(first_experiment["plus"], {"TuneA": 6, "TuneB": 25})
        second_experiment = json.loads(
            (lab / "iterations" / "00001" / "experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(second_experiment["book_start"] - first_experiment["book_start"], 2)
        self.assertIn(f"start={first_experiment['book_start']}", first_experiment["command"])
        self.assertIn("score_se", recommendation["last_comparison"])

        resumed = self.command("resume", "--lab", str(lab))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(
            checkpoint,
            json.loads((lab / "checkpoint.json").read_text(encoding="utf-8")),
        )

        stale = json.loads(
            (lab / "iterations" / "00000" / "result.json").read_text(encoding="utf-8")
        )
        stale["plus"]["TuneA"] += 1
        (lab / "iterations" / "00000" / "result.json").write_text(
            json.dumps(stale), encoding="utf-8"
        )
        (lab / "checkpoint.json").write_text(
            json.dumps(
                {
                    "schema": "volkrix-local-spsa-v1",
                    "next_iteration": 0,
                    "vector": {"TuneA": 10.0, "TuneB": 20.0},
                    "history": [],
                }
            ),
            encoding="utf-8",
        )
        stale_rejected = self.command("resume", "--lab", str(lab))
        self.assertEqual(stale_rejected.returncode, 2)
        self.assertIn("cached result metadata does not match", stale_rejected.stderr)

        (lab / "assets" / "openings.epd").write_text("tampered", encoding="utf-8")
        rejected = self.command("resume", "--lab", str(lab))
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("frozen asset changed", rejected.stderr)

    def test_invalid_parameter_is_rejected_before_any_games(self) -> None:
        completed = self.command(
            "start",
            "--fastchess", str(self.fastchess),
            "--engine", str(self.engine),
            "--book", str(self.book),
            "--evalfile", "embedded",
            "--output", str(self.root / "invalid"),
            "--parameters", "TuneMissing",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unknown tuning parameter", completed.stderr)

    def test_boundary_perturbation_uses_exact_signed_odd_span(self) -> None:
        manifest = {
            "config": {"seed": 7},
            "parameters": ["TuneA"],
            "tuning_schema": {
                "parameters": {"TuneA": {"default": 10, "min": 0, "max": 30, "step": 4}}
            },
        }
        checkpoint = {"vector": {"TuneA": 29.0}}
        _, _, _, spans, half_radii = local_spsa.vectors_for_iteration(
            manifest, checkpoint, 0
        )
        self.assertEqual(abs(spans["TuneA"]), 5)
        self.assertEqual(half_radii["TuneA"], 2.5)

    def test_invalid_runtime_numbers_are_rejected(self) -> None:
        invalid = [
            ("--learning-rate", "nan", "learning-rate"),
            ("--learning-rate", "-1", "learning-rate"),
            ("--concurrency", "65", "concurrency"),
            ("--move-overhead-ms", "5001", "move-overhead"),
            ("--time-margin-ms", "60001", "time-margin"),
            ("--tc", "0.2+0.02\nmalformed", "contain no CR/LF"),
        ]
        for index, (flag, value, message) in enumerate(invalid):
            with self.subTest(flag=flag, value=value):
                completed = self.command(
                    "start", "--fastchess", str(self.fastchess), "--engine", str(self.engine),
                    "--book", str(self.book), "--evalfile", "embedded",
                    "--output", str(self.root / f"invalid-number-{index}"),
                    "--parameters", "TuneA", "--iterations", "1", "--pairs-per-iteration", "1",
                    flag, value,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(message, completed.stderr)

    def test_unknown_book_suffix_requires_explicit_format(self) -> None:
        unknown = self.root / "book.data"
        unknown.write_text(self.book.read_text(encoding="utf-8"), encoding="utf-8")
        completed = self.command(
            "start", "--fastchess", str(self.fastchess), "--engine", str(self.engine),
            "--book", str(unknown), "--evalfile", "embedded",
            "--output", str(self.root / "unknown-book"), "--parameters", "TuneA",
            "--iterations", "1", "--pairs-per-iteration", "1",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--book-format is required", completed.stderr)

    def test_active_campaign_lock_rejects_concurrent_resume(self) -> None:
        lab = self.root / "locked"
        started = self.command(
            "start", "--fastchess", str(self.fastchess), "--engine", str(self.engine),
            "--book", str(self.book), "--evalfile", "embedded", "--output", str(lab),
            "--parameters", "TuneA", "--iterations", "1", "--pairs-per-iteration", "1",
        )
        self.assertEqual(started.returncode, 0, started.stderr)
        import os, socket
        (lab / ".campaign.lock").write_text(
            json.dumps({"pid": os.getpid(), "host": socket.gethostname()}), encoding="utf-8"
        )
        rejected = self.command("resume", "--lab", str(lab))
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("campaign is locked", rejected.stderr)

    def test_interrupted_iteration_resumes_from_fastchess_recovery(self) -> None:
        crash_fastchess = executable(
            self.root / "crash-fastchess",
            r"""#!/usr/bin/env python3
import json, pathlib, sys
config_index = sys.argv.index('-config')
if sys.argv[config_index + 1].startswith('file='):
    recovery = pathlib.Path(sys.argv[config_index + 1].split('=', 1)[1])
    state = json.loads(recovery.read_text())
    pathlib.Path(state['pgn']).write_text('[Event "test"]\n[Round "1"]\n[White "SPSA-plus"]\n[Black "SPSA-minus"]\n[Result "1-0"]\n[Termination "normal"]\n\n1-0\n\n[Event "test"]\n[Round "1"]\n[White "SPSA-minus"]\n[Black "SPSA-plus"]\n[Result "1/2-1/2"]\n[Termination "normal"]\n\n1/2-1/2\n')
    pathlib.Path(state['log']).write_text('recovered cleanly\n')
    print('Score of SPSA-plus vs SPSA-minus: 1 - 0 - 1')
    raise SystemExit(0)
outname = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('outname='))
pgn = sys.argv[sys.argv.index('-pgnout') + 1].split('=', 1)[1]
log = sys.argv[sys.argv.index('-log') + 1].split('=', 1)[1]
pathlib.Path(outname).write_text(json.dumps({'pgn': pgn, 'log': log}))
raise SystemExit(7)
""",
        )
        lab = self.root / "recovery"
        failed = self.command(
            "start", "--fastchess", str(crash_fastchess), "--engine", str(self.engine),
            "--book", str(self.book), "--evalfile", "embedded", "--output", str(lab),
            "--parameters", "TuneA", "--iterations", "1", "--pairs-per-iteration", "1",
        )
        self.assertEqual(failed.returncode, 2)
        self.assertFalse((lab / ".campaign.lock").exists())
        resumed = self.command("resume", "--lab", str(lab))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        resume_command = json.loads(
            (lab / "iterations" / "00000" / "resume-command.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            resume_command,
            [
                str((lab / "assets" / "fastchess").resolve()),
                "-config",
                f"file={(lab / 'iterations' / '00000' / 'recovery.json').resolve()}",
                "stats=true",
            ],
        )

    def test_embedded_mode_rejects_engine_without_embedded_identity(self) -> None:
        plain = executable(
            self.root / "plain-engine",
            """#!/usr/bin/env python3
import sys
print('option name EvalFile type string default')
for line in sys.stdin:
    if line.startswith('setoption name TuneManifest'):
        print('info string tuning manifest version 1 checksum fake')
        print('info string tuning parameter TuneA value 10 default 10 min 0 max 30 step 4')
        print('info string tuning manifest end')
    elif line.strip() == 'quit':
        break
""",
        )
        completed = self.command(
            "start", "--fastchess", str(self.fastchess), "--engine", str(plain),
            "--book", str(self.book), "--evalfile", "embedded",
            "--output", str(self.root / "plain"), "--parameters", "TuneA",
            "--iterations", "1", "--pairs-per-iteration", "1",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires an engine whose advertised EvalFile", completed.stderr)

    def test_evaluation_preflight_rejects_engine_without_bestmove(self) -> None:
        broken = executable(
            self.root / "broken-engine",
            """#!/usr/bin/env python3
import sys
print('option name EvalFile type string default <embedded:' + 'b' * 64 + ':1234>')
for line in sys.stdin:
    if line.strip() == 'uci': print('uciok')
    elif line.strip() == 'isready': print('readyok')
    elif line.startswith('setoption name TuneManifest'):
        print('info string tuning manifest version 1 checksum fake')
        print('info string tuning parameter TuneA value 10 default 10 min 0 max 30 step 4')
        print('info string tuning manifest end')
""",
        )
        completed = self.command(
            "start", "--fastchess", str(self.fastchess), "--engine", str(broken),
            "--book", str(self.book), "--evalfile", "embedded",
            "--output", str(self.root / "broken"), "--parameters", "TuneA",
            "--iterations", "1", "--pairs-per-iteration", "1",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("evaluation preflight failed", completed.stderr)

    def test_external_evaluator_is_frozen_preflighted_and_passed_to_both_engines(self) -> None:
        network = self.root / "network.nnue"
        network.write_bytes(b"deterministic-test-network")
        lab = self.root / "external-eval"
        completed = self.command(
            "start", "--fastchess", str(self.fastchess), "--engine", str(self.engine),
            "--book", str(self.book), "--evalfile", str(network), "--output", str(lab),
            "--parameters", "TuneA", "--iterations", "1", "--pairs-per-iteration", "1",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = json.loads((lab / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["evaluation"]["mode"], "external")
        frozen = lab / manifest["evaluation"]["asset"]["path"]
        self.assertEqual(frozen.read_bytes(), network.read_bytes())
        experiment = json.loads(
            (lab / "iterations" / "00000" / "experiment.json").read_text(encoding="utf-8")
        )
        option = f"option.EvalFile={frozen.resolve()}"
        self.assertEqual(experiment["command"].count(option), 2)

    def test_abnormal_pgn_termination_rejects_iteration(self) -> None:
        abnormal = executable(
            self.root / "abnormal-fastchess",
            self.fastchess.read_text(encoding="utf-8").replace(
                'Termination "normal"', 'Termination "time forfeit"'
            ),
        )
        completed = self.command(
            "start", "--fastchess", str(abnormal), "--engine", str(self.engine),
            "--book", str(self.book), "--evalfile", "embedded",
            "--output", str(self.root / "abnormal"), "--parameters", "TuneA",
            "--iterations", "1", "--pairs-per-iteration", "1",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("abnormal engine terminations", completed.stderr)

    def test_pgn_requires_two_opposite_color_games_per_exact_round(self) -> None:
        def game(round_tag: str, white: str, black: str) -> str:
            return (
                f'[Event "test"]\n[Round "{round_tag}"]\n[White "{white}"]\n'
                f'[Black "{black}"]\n[Result "1/2-1/2"]\n'
                '[Termination "normal"]\n\n1/2-1/2\n'
            )

        cases = {
            "aggregate-only": [
                game("1", "SPSA-plus", "SPSA-minus"),
                game("1", "SPSA-plus", "SPSA-minus"),
                game("2", "SPSA-minus", "SPSA-plus"),
                game("2", "SPSA-minus", "SPSA-plus"),
            ],
            "duplicate-missing": [
                game("1", "SPSA-plus", "SPSA-minus"),
                game("1", "SPSA-minus", "SPSA-plus"),
                game("1", "SPSA-plus", "SPSA-minus"),
                game("2", "SPSA-minus", "SPSA-plus"),
            ],
        }
        for name, games in cases.items():
            with self.subTest(name=name):
                iteration = self.root / name
                iteration.mkdir()
                (iteration / "games.pgn").write_text("\n".join(games), encoding="utf-8")
                (iteration / "fastchess.log").write_text("log", encoding="utf-8")
                (iteration / "recovery.json").write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Round|opening rounds"):
                    local_spsa.iteration_evidence(iteration, 4)


if __name__ == "__main__":
    unittest.main()
