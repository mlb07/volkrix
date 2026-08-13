#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import pathlib
import stat
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("strength_lab.py")
SPEC = importlib.util.spec_from_file_location("strength_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class StrengthLabTests(unittest.TestCase):
    UCI_ENGINE = b"""#!/usr/bin/env python3
import sys
for command in sys.stdin:
    command = command.strip()
    if command == 'uci':
        print('id name Fixture')
        print('uciok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('go '):
        print('bestmove e2e4', flush=True)
    elif command == 'quit':
        break
"""

    def make_file(self, root: pathlib.Path, name: str, data: bytes, executable: bool = False) -> pathlib.Path:
        path = root / name
        path.write_bytes(data)
        if executable:
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def config(self, root: pathlib.Path) -> pathlib.Path:
        fastchess = self.make_file(root, "fastchess", b"#!/bin/sh\nexit 0\n", True)
        candidate = self.make_file(root, "volkrix", self.UCI_ENGINE, True)
        opponent = self.make_file(root, "opponent", self.UCI_ENGINE, True)
        book = self.make_file(root, "book.epd", b"startpos\n")
        config = {
            "schema": lab.SCHEMA,
            "fastchess": {"path": str(fastchess)},
            "candidate": {"name": "Volkrix", "path": str(candidate), "options": {"EvalFile": "classical"}},
            "openings": {"path": str(book), "format": "epd"},
            "profiles": [
                {"name": "STC", "tc": "10+0.1", "pairs": 2, "concurrency": 1},
                {"name": "LTC", "tc": "60+0.6", "pairs": 1, "concurrency": 1},
            ],
            "opponents": [{"name": "Rival", "path": str(opponent), "options": {}}],
        }
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_prepare_freezes_matrix_commands_and_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "lab"
            lab.prepare(self.config(root), output)
            manifest = lab.load_lab(output)
            self.assertEqual(manifest["input_storage"], "self-contained-copy-v1")
            self.assertTrue(
                pathlib.Path(manifest["openings"]["path"]).is_relative_to(
                    (output / "inputs").resolve()
                )
            )
            self.assertEqual(
                pathlib.Path(manifest["openings"]["path"]).stat().st_mode & 0o222,
                0,
            )
            self.assertNotEqual(
                pathlib.Path(manifest["candidate"]["path"]).stat().st_mode & 0o111,
                0,
            )
            self.assertEqual([job["id"] for job in manifest["jobs"]], ["STC__Rival", "LTC__Rival"])
            self.assertEqual(len(manifest["preflights"]), 2)
            self.assertEqual(len(manifest["preflights"][0]["contexts"]), 2)
            job, _ = lab.verify_job(output, manifest["jobs"][0])
            command = job["command"]
            self.assertIn("-repeat", command)
            self.assertIn("2", command)
            self.assertIn("order=sequential", command)
            self.assertIn("-autosaveinterval", command)
            self.assertIn("movecount=3", command)
            self.assertNotIn("movecount=3 score=400", command)
            self.assertNotIn("fi=true", command)
            lab.verify_inputs(manifest)
            frozen_openings = pathlib.Path(manifest["openings"]["path"])
            frozen_openings.chmod(0o644)
            frozen_openings.write_bytes(b"changed")
            with self.assertRaisesRegex(lab.LabError, "frozen input changed"):
                lab.verify_inputs(manifest)

    def test_prepared_lab_survives_deletion_of_every_source_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "lab"
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            network = self.make_file(root, "network.nnue", b"frozen-network")
            config["candidate"]["options"]["EvalFile"] = str(network)
            config["assets"] = [{"name": "network", "path": str(network)}]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            source_paths = [
                pathlib.Path(config["fastchess"]["path"]),
                pathlib.Path(config["candidate"]["path"]),
                pathlib.Path(config["openings"]["path"]),
                pathlib.Path(config["opponents"][0]["path"]),
                network,
            ]

            lab.prepare(config_path, output)
            manifest = lab.load_lab(output)
            for path in source_paths:
                path.unlink()

            lab.verify_inputs(manifest)
            lab.verify_preflights(output, manifest)
            job, _ = lab.verify_job(output, manifest["jobs"][0])
            for source_path in source_paths:
                self.assertNotIn(str(source_path), job["command"])

    def test_relative_evalfile_is_frozen_as_config_absolute_not_engine_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config_dir = root / "configs"
            engine_dir = root / "engines"
            config_dir.mkdir()
            engine_dir.mkdir()
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            candidate = self.make_file(engine_dir, "volkrix", self.UCI_ENGINE, True)
            network = self.make_file(config_dir, "network.nnue", b"frozen-network")
            config["candidate"]["path"] = str(candidate)
            config["candidate"]["options"]["EvalFile"] = "network.nnue"
            config["profiles"][0]["options"] = {"SmallEvalFile": "network.nnue"}
            config["assets"] = [{"name": "network", "path": "network.nnue"}]
            relocated = config_dir / "config.json"
            relocated.write_text(json.dumps(config), encoding="utf-8")

            resolved = lab.resolve_config(relocated)
            self.assertEqual(
                resolved["candidate"]["options"]["EvalFile"], str(network.resolve())
            )
            self.assertEqual(
                resolved["profiles"][0]["options"]["SmallEvalFile"],
                str(network.resolve()),
            )
            command = lab.job_command(
                resolved,
                resolved["profiles"][0],
                resolved["opponents"][0],
                (root / "job").resolve(),
            )
            self.assertIn(f"option.EvalFile={network.resolve()}", command)
            self.assertIn(f"option.SmallEvalFile={network.resolve()}", command)

    def test_prepare_preflights_effective_options_and_freezes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "lab"
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["profiles"][0]["options"] = {"Hash": 16}
            config["profiles"][1]["options"] = {"Hash": 32}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            lab.prepare(config_path, output)
            manifest = lab.load_lab(output)
            self.assertEqual(len(manifest["preflights"]), 4)
            self.assertTrue(
                all("setoption name Hash value" in item["stdin"] for item in manifest["preflights"])
            )
            lab.verify_preflights(output, manifest)
            evidence = output / manifest["preflights"][0]["log"]
            evidence.chmod(0o644)
            evidence.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "preflight evidence changed"):
                lab.verify_lab(output)

    def test_empty_option_is_preflighted_but_omitted_from_fastchess_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "lab"
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["candidate"]["options"]["SyzygyPath"] = ""
            config["opponents"][0]["options"]["SyzygyPath"] = ""
            config_path.write_text(json.dumps(config), encoding="utf-8")

            lab.prepare(config_path, output)
            manifest = lab.load_lab(output)
            job, _ = lab.verify_job(output, manifest["jobs"][0])
            self.assertNotIn("option.SyzygyPath=", job["command"])
            self.assertEqual(
                [record["options"]["SyzygyPath"] for record in manifest["preflights"]],
                ["", ""],
            )
            self.assertTrue(
                all(
                    "setoption name SyzygyPath value \n" in record["stdin"]
                    for record in manifest["preflights"]
                )
            )

    def test_arbitrary_empty_option_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["candidate"]["options"]["EvalFile"] = ""
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(lab.LabError, "only the verified default-empty SyzygyPath"):
                lab.prepare(config_path, root / "lab")

    def test_prepare_rejects_engine_that_reports_option_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            broken = self.make_file(
                root,
                "broken",
                b"#!/bin/sh\nprintf 'uciok\\ninfo string error invalid option\\nreadyok\\nbestmove e2e4\\n'\n",
                True,
            )
            config["candidate"]["path"] = str(broken)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "option/error diagnostic"):
                lab.prepare(config_path, root / "lab")

    def test_prepare_preflight_timeout_kills_engine_and_closes_pipes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            hanging = self.make_file(
                root,
                "hanging",
                b"""#!/usr/bin/env python3
import sys, time
for command in sys.stdin:
    if command.strip() == 'uci':
        print('uciok', flush=True)
    elif command.strip() == 'isready':
        time.sleep(30)
""",
                True,
            )
            config["candidate"]["path"] = str(hanging)
            config["preflight_timeout_seconds"] = 0.1
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "timed out"):
                lab.prepare(config_path, root / "lab")

    def test_nonfinite_numbers_and_carriage_returns_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["opponents"][0]["rating"] = float("nan")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "finite number"):
                lab.resolve_config(config_path)

            config["opponents"][0]["rating"] = 10**1000
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "finite number"):
                lab.resolve_config(config_path)

            config["opponents"][0].pop("rating")
            config["candidate"]["options"]["Threads"] = float("inf")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "must be finite"):
                lab.resolve_config(config_path)

            config["candidate"]["options"]["Threads"] = 1
            config["profiles"][0]["tc"] = "10+0.1\r-injected"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "time control"):
                lab.resolve_config(config_path)

            config["profiles"][0]["tc"] = "10+0.1"
            config["candidate"]["options"]["Bad\rName"] = 1
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "invalid UCI option name"):
                lab.resolve_config(config_path)

            config["candidate"]["options"].pop("Bad\rName")
            config["adjudication"] = {"resign": "movecount=3\rscore=400"}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "single-line string"):
                lab.resolve_config(config_path)

    def test_pgn_summary_classifies_engine_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pgn = pathlib.Path(temporary) / "games.pgn"
            pgn.write_text(
                '\n'.join(
                    [
                        '[Event "g1"]', '[Round "1"]', '[White "Volkrix"]', '[Black "Rival"]',
                        '[Result "1-0"]', '[Termination "normal"]', '', '1-0', '',
                        '[Event "g2"]', '[Round "1"]', '[White "Rival"]', '[Black "Volkrix"]',
                        '[Result "1-0"]', '[Termination "illegal move"]', '', '1-0', '',
                    ]
                ),
                encoding="utf-8",
            )
            summary = lab.summarize_games(pgn, "Volkrix", 2)
            self.assertEqual(summary["candidate_wins"], 1)
            self.assertEqual(summary["candidate_losses"], 1)
            self.assertEqual(summary["failures"]["illegal_move"], 1)
            self.assertEqual(summary["pentanomial"], [0, 0, 1, 0, 0])
            self.assertEqual(summary["elo_difference"], 0.0)

    def test_pentanomial_groups_interleaved_completion_by_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pgn = pathlib.Path(temporary) / "interleaved.pgn"
            games = [
                ("1a", "1", "Volkrix", "Rival", "1-0"),
                ("2a", "2", "Volkrix", "Rival", "0-1"),
                ("1b", "1", "Rival", "Volkrix", "0-1"),
                ("2b", "2", "Rival", "Volkrix", "1-0"),
            ]
            pgn.write_text(
                "\n\n".join(
                    f'[Event "{event}"]\n[Round "{round_tag}"]\n[White "{white}"]\n'
                    f'[Black "{black}"]\n[Result "{result}"]\n[Termination "normal"]\n\n{result}'
                    for event, round_tag, white, black, result in games
                )
                + "\n",
                encoding="utf-8",
            )
            summary = lab.summarize_games(pgn, "Volkrix", 4)
            self.assertEqual(summary["pentanomial"], [1, 0, 0, 0, 1])
            self.assertEqual(summary["score_percent"], 50.0)

    def test_pentanomial_rejects_duplicate_or_non_reversed_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pgn = pathlib.Path(temporary) / "malformed.pgn"
            pgn.write_text(
                '\n\n'.join(
                    [
                        '[Event "a"]\n[Round "7"]\n[White "Volkrix"]\n[Black "Rival"]\n[Result "1-0"]\n[Termination "normal"]\n\n1-0',
                        '[Event "b"]\n[Round "7"]\n[White "Volkrix"]\n[Black "Rival"]\n[Result "1-0"]\n[Termination "normal"]\n\n1-0',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(lab.LabError, "not a color-reversed"):
                lab.summarize_games(pgn, "Volkrix", 2)

    def test_duplicate_names_and_bad_checksums_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["opponents"][0]["name"] = "Volkrix"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "duplicate engine name"):
                lab.resolve_config(config_path)

            config["opponents"][0]["name"] = "Rival"
            config["candidate"]["sha256"] = "0" * 64
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "checksum mismatch"):
                lab.resolve_config(config_path)

    def test_completed_artifacts_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "lab"
            lab.prepare(self.config(root), output)
            manifest = lab.load_lab(output)
            _, job_dir = lab.verify_job(output, manifest["jobs"][0])
            (job_dir / "summary.json").write_text("{}\n", encoding="utf-8")
            (job_dir / "games.pgn").write_text("frozen games\n", encoding="utf-8")
            hashes = {
                name: lab.sha256_file(job_dir / name)
                for name in ("summary.json", "games.pgn")
            }
            lab.write_new(
                job_dir / "completed.json",
                lab.canonical_json({"artifacts": hashes}),
            )
            lab.verify_lab(output)
            (job_dir / "summary.json").write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "completed artifact changed"):
                lab.verify_lab(output)

    def test_summary_without_completion_marker_stays_pending_and_tamper_blocks_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "lab"
            lab.prepare(self.config(root), output)
            manifest = lab.load_lab(output)
            _, job_dir = lab.verify_job(output, manifest["jobs"][0])
            (job_dir / "summary.json").write_text('{"games":2}\n', encoding="utf-8")
            report = lab.aggregate(output, manifest)
            self.assertEqual(report["jobs"][manifest["jobs"][0]["id"]]["status"], "pending")

            (job_dir / "games.pgn").write_text("games\n", encoding="utf-8")
            hashes = {
                name: lab.sha256_file(job_dir / name)
                for name in ("summary.json", "games.pgn")
            }
            lab.write_new(
                job_dir / "completed.json", lab.canonical_json({"artifacts": hashes})
            )
            (job_dir / "summary.json").write_text('{"games":999}\n', encoding="utf-8")
            with self.assertRaisesRegex(lab.LabError, "completed artifact changed"):
                lab.aggregate(output, manifest)
            with self.assertRaisesRegex(lab.LabError, "completed artifact changed"):
                lab.run_lab(output, {manifest["jobs"][0]["id"]}, False)

    def test_interrupted_job_resumes_from_fastchess_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config_path = self.config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            fake = root / "fastchess"
            fake.write_text(
                """#!/usr/bin/env python3
import json, pathlib, sys
cwd = pathlib.Path.cwd()
job = json.loads((cwd / 'job.json').read_text())
resume = any(arg.startswith('file=') and arg.endswith('recovery.json') for arg in sys.argv)
recovery = cwd / 'recovery.json'
if not resume:
    recovery.write_text('{}')
    raise SystemExit(7)
games = []
for number in range(job['expected_games']):
    white, black = (job['candidate'], job['opponent']) if number % 2 == 0 else (job['opponent'], job['candidate'])
    games.append(f'[Event "g{number}"]\\n[Round "{number // 2 + 1}"]\\n[White "{white}"]\\n[Black "{black}"]\\n[Result "1/2-1/2"]\\n[Termination "normal"]\\n\\n1/2-1/2\\n')
(cwd / 'games.pgn').write_text('\\n'.join(games))
(cwd / 'fastchess.log').write_text('resumed\\n')
""",
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            config["fastchess"] = {"path": str(fake)}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "lab"
            lab.prepare(config_path, output)
            selected = {"STC__Rival"}
            with self.assertRaisesRegex(lab.LabError, "rerun the same command to resume"):
                lab.run_lab(output, selected, False)
            self.assertTrue((output / "jobs/STC__Rival/recovery.json").is_file())
            lab.run_lab(output, selected, False)
            self.assertTrue((output / "jobs/STC__Rival/completed.json").is_file())
            summary = json.loads((output / "jobs/STC__Rival/summary.json").read_text())
            self.assertEqual(summary["games"], 4)
            self.assertEqual(summary["draws"], 4)
            lab.verify_lab(output)


if __name__ == "__main__":
    unittest.main()
