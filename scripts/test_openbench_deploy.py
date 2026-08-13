#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).with_name("openbench_deploy.py")
SPEC = importlib.util.spec_from_file_location("openbench_deploy", MODULE_PATH)
assert SPEC and SPEC.loader
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


class OpenBenchDeploymentTests(unittest.TestCase):
    def fake_upstream(self, root: pathlib.Path) -> pathlib.Path:
        upstream = root / "OpenBench"
        for directory in ("OpenBench", "OpenSite", "Config", "Books", "Engines", "Client"):
            (upstream / directory).mkdir(parents=True, exist_ok=True)
        (upstream / "OpenBench/__init__.py").write_text("", encoding="utf-8")
        (upstream / "OpenSite/__init__.py").write_text("", encoding="utf-8")
        (upstream / "OpenSite/settings.py").write_text(
            "import pathlib\nPROJECT_PATH = str(pathlib.Path(__file__).resolve().parents[1])\n",
            encoding="utf-8",
        )
        (upstream / "OpenBench/config.py").write_text(
            """OPENBENCH_STATIC_VERSION = 'v17'
def verify_engine_basics(value): assert value['private'] is False
def verify_engine_build(name, value): assert value['build']['path'] == 'openbench'
def verify_engine_test_preset(value): assert isinstance(value, dict)
def verify_engine_tune_preset(value): assert isinstance(value, dict)
def verify_engine_datagen_preset(value): assert isinstance(value, dict)
""",
            encoding="utf-8",
        )
        for relative in ("OpenBench/utils.py", "OpenBench/workloads/view_workload.py"):
            path = upstream / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "import datetime\nfrom django.utils import timezone\n"
                "def recent():\n"
                "    target = datetime.datetime.utcnow()\n"
                "    target = target.replace(tzinfo=timezone.utc)\n"
                "    return target\n",
                encoding="utf-8",
            )
        (upstream / "OpenBench/apps.py").write_text(
            "import atexit\nimport pathlib\n"
            "def ready():\n"
            "        # Attempt to spawn the PGN Watcher, globally once\n\n"
            "        from OpenBench.pgn_watcher import PGNWatcher\n",
            encoding="utf-8",
        )
        (upstream / "OpenBench/pgn_watcher.py").write_text(
            "class PGNWatcher:\n"
            "    def run(self):\n\n"
            "        # Loop until we are shutdown by the atexit.register()\n"
            "        while not self.stop_event.is_set():\n"
            "            pass\n",
            encoding="utf-8",
        )
        (upstream / "Client/utils.py").write_text(
            "import argparse\nimport hashlib\nimport os\nimport platform\nimport requests\n"
            "import shutil\nimport subprocess\n\n"
            "IS_WINDOWS = False\nIS_LINUX = True\n\n"
            "def kill_process_by_name(process_name):\n\n"
            "    process_name = os.path.basename(process_name)\n\n"
            "    if IS_LINUX:\n"
            "        subprocess.run(['pkill', '-KILL', '-f', process_name])\n\n"
            "    if IS_WINDOWS:\n"
            "        subprocess.run(['taskkill', '/f', '/im', process_name])\n",
            encoding="utf-8",
        )
        config = {
            "client_version": 49,
            "client_repo_url": "https://example.invalid/OpenBench",
            "client_repo_ref": "master",
            "fastchess_min_version": "1.8.1",
            "fastchess_repo_url": "https://example.invalid/fastchess",
            "fastchess_repo_ref": "master",
            "use_cross_approval": False,
            "require_login_to_view": False,
            "require_manual_registration": False,
            "balance_engine_throughputs": False,
            "books": [],
            "engines": [],
        }
        (upstream / "Config/config.json").write_text(json.dumps(config), encoding="utf-8")
        (upstream / "Books/UHO_Lichess_4852_v1.epd.json").write_text(
            json.dumps({"sha": "a" * 64, "source": "https://example.invalid/book.zip"}),
            encoding="utf-8",
        )
        (upstream / "manage.py").write_text("# fixture\n", encoding="utf-8")
        lock = deploy.strict_json(deploy.UPSTREAM_LOCK)
        (upstream / "requirements.txt").write_text(
            f"Django=={lock['upstream_django_pin']}\nrequests\nscipy\n", encoding="utf-8"
        )
        return upstream

    def prepare_args(self, upstream: pathlib.Path, output: pathlib.Path) -> argparse.Namespace:
        lock = deploy.strict_json(deploy.UPSTREAM_LOCK)
        return argparse.Namespace(
            openbench_root=upstream,
            output=output,
            nps=765432,
            book="UHO_Lichess_4852_v1.epd",
            client_repo_url=lock["openbench"]["repository"],
            client_ref=lock["openbench"]["commit"],
            fastchess_ref=lock["fastchess"]["commit"],
        )

    def test_current_engine_configuration_has_complete_deployment_presets(self) -> None:
        self.assertFalse(
            any(deploy.DEPLOYMENT_ASSETS.rglob("*.pyc")),
            "deployment templates must not contain generated Python bytecode",
        )
        config = deploy.strict_json(deploy.ENGINE_CONFIG)
        lock = deploy.strict_json(deploy.UPSTREAM_LOCK)
        self.assertEqual(deploy.validate_engine_config(config), [])
        deploy_requirements = (deploy.DEPLOYMENT_ASSETS / "requirements.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"gunicorn=={lock['deployment_gunicorn']}", deploy_requirements)
        self.assertIn(f"mysqlclient=={lock['deployment_mysqlclient']}", deploy_requirements)
        broken = json.loads(json.dumps(config))
        broken["tune_presets"].pop("LTC")
        self.assertIn("tune_presets.LTC is required", deploy.validate_engine_config(broken))

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "duplicate.json"
            path.write_text('{"nps":1,"nps":2}', encoding="utf-8")
            with self.assertRaisesRegex(deploy.DeployError, "duplicate JSON key"):
                deploy.strict_json(path)

    def test_audit_emits_versioned_complete_top_level_schema(self) -> None:
        args = argparse.Namespace(openbench_root=None, fastchess_root=None, network=None)

        def fake_version(tool: str) -> tuple[str, tuple[int, ...]]:
            return f"/{tool}: {tool} 99.0", (99, 0)

        with (
            mock.patch.object(deploy, "command_version", side_effect=fake_version),
            mock.patch.object(
                deploy,
                "core_layout",
                return_value={"logical": 4, "physical": 4, "performance": None, "efficiency": None},
            ),
            mock.patch.object(deploy, "machine_memory_bytes", return_value=8 * 1024**3),
            mock.patch.object(deploy.platform, "system", return_value="Linux"),
            mock.patch.object(deploy.platform, "machine", return_value="x86_64"),
        ):
            report = deploy.audit(args)

        self.assertEqual(report["schema"], "volkrix-openbench-deployment-audit-v1")
        self.assertEqual(
            set(report),
            {"schema", "ready", "blockers", "warnings", "facts", "external_remaining"},
        )

    def test_prepare_copies_pinned_instance_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            upstream = self.fake_upstream(root)
            output = root / "prepared"
            lock = deploy.strict_json(deploy.UPSTREAM_LOCK)

            def fake_git(_root: pathlib.Path, *arguments: str) -> str:
                if arguments == ("rev-parse", "HEAD"):
                    return lock["openbench"]["commit"]
                if arguments == ("status", "--porcelain"):
                    return ""
                raise AssertionError(arguments)

            with mock.patch.object(deploy, "git", side_effect=fake_git):
                manifest = deploy.prepare(self.prepare_args(upstream, output))

            config = deploy.strict_json(output / "Config/config.json")
            engine = deploy.strict_json(output / "Engines/Volkrix.json")
            self.assertEqual(config["client_repo_ref"], lock["openbench"]["commit"])
            self.assertEqual(config["fastchess_repo_ref"], lock["fastchess"]["commit"])
            self.assertEqual(config["engines"], ["Volkrix"])
            self.assertEqual(config["books"], ["UHO_Lichess_4852_v1.epd"])
            self.assertTrue(config["require_login_to_view"])
            self.assertTrue(config["require_manual_registration"])
            self.assertEqual(engine["nps"], 765432)
            self.assertIn(
                f"Django=={lock['deployment_django']}",
                (output / "requirements.txt").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                f"Django=={lock['upstream_django_pin']}",
                (output / "requirements.txt").read_text(encoding="utf-8"),
            )
            requirements = (output / "requirements.txt").read_text(encoding="utf-8")
            self.assertIn(f"requests=={lock['deployment_requests']}", requirements)
            self.assertIn(f"scipy=={lock['deployment_scipy']}", requirements)
            self.assertEqual(manifest["schema"], "volkrix-openbench-deployment-v1")
            self.assertEqual(
                manifest["compatibility_patches"],
                [
                    "OpenBench/utils.py",
                    "OpenBench/workloads/view_workload.py",
                    "OpenBench/apps.py",
                    "OpenBench/pgn_watcher.py",
                    "Client/utils.py",
                ],
            )
            for relative in manifest["compatibility_patches"][:2]:
                contents = (output / relative).read_text(encoding="utf-8")
                self.assertIn("target = timezone.now()", contents)
                self.assertNotIn("timezone.utc", contents)
            self.assertIn(
                "OPENBENCH_DISABLE_WATCHER",
                (output / "OpenBench/apps.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "self.stop_event.wait(timeout=1)",
                (output / "OpenBench/pgn_watcher.py").read_text(encoding="utf-8"),
            )
            client_utils = (output / "Client/utils.py").read_text(encoding="utf-8")
            self.assertIn("psutil.Process().children(recursive=True)", client_utils)
            self.assertNotIn("subprocess.run(['pkill'", client_utils)
            self.assertNotIn("subprocess.run(['taskkill'", client_utils)
            self.assertTrue((output / "OpenSite/volkrix_settings.py").is_file())
            self.assertFalse(any(output.rglob("credentials.*")))
            self.assertIn("openbench.env.example", manifest["files"])
            for name, digest in manifest["files"].items():
                self.assertEqual(deploy.sha256_file(output / name), digest)

    def test_prepare_rejects_unpinned_or_unsafe_inputs_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            upstream = self.fake_upstream(root)
            output = root / "prepared"
            args = self.prepare_args(upstream, output)
            args.book = "../escape.epd"
            with self.assertRaisesRegex(deploy.DeployError, "plain EPD/PGN"):
                deploy.prepare(args)
            self.assertFalse(output.exists())

            args = self.prepare_args(upstream, output)
            args.nps = 0
            with self.assertRaisesRegex(deploy.DeployError, "positive integer"):
                deploy.prepare(args)
            self.assertFalse(output.exists())

            args = self.prepare_args(upstream, output)
            with mock.patch.object(deploy, "git", return_value="0" * 40):
                with self.assertRaisesRegex(deploy.DeployError, "differs from audited"):
                    deploy.prepare(args)
            self.assertFalse(output.exists())

            (upstream / "requirements.txt").write_text("Django==0.0\n", encoding="utf-8")

            def fake_git(_root: pathlib.Path, *arguments: str) -> str:
                if arguments == ("rev-parse", "HEAD"):
                    return deploy.strict_json(deploy.UPSTREAM_LOCK)["openbench"]["commit"]
                if arguments == ("status", "--porcelain"):
                    return ""
                raise AssertionError(arguments)

            with mock.patch.object(deploy, "git", side_effect=fake_git):
                with self.assertRaisesRegex(deploy.DeployError, "re-audit dependency"):
                    deploy.prepare(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
