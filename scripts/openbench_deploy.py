#!/usr/bin/env python3
"""Audit and prepare a pinned Volkrix OpenBench server/worker deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE_CONFIG = ROOT / "openbench" / "Volkrix.json.example"
UPSTREAM_LOCK = ROOT / "openbench" / "upstream-lock.json"
DEPLOYMENT_ASSETS = ROOT / "openbench" / "deployment"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_BOOK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DeployError(RuntimeError):
    pass


def strict_json(path: pathlib.Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DeployError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError) as error:
        raise DeployError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise DeployError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: pathlib.Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise DeployError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def git(root: pathlib.Path, *arguments: str) -> str:
    return run(["git", "-C", str(root), *arguments])


def version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+(?:\.\d+)+)", text)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def validate_engine_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    expect(config.get("private") is False, "Volkrix must be configured as a public engine")
    expect(isinstance(config.get("nps"), int) and config.get("nps", 0) > 0, "nps must be a positive integer")
    expect(isinstance(config.get("source"), str) and config["source"].startswith("https://github.com/"), "source must be a public GitHub repository")
    build = config.get("build")
    expect(isinstance(build, dict), "build must be an object")
    if isinstance(build, dict):
        expect(build.get("path") == "openbench", "build.path must be openbench")
        compilers = build.get("compilers")
        expect(isinstance(compilers, list) and "cargo>=1.85.0" in compilers, "build.compilers must require cargo>=1.85.0")
        systems = build.get("systems")
        expect(isinstance(systems, list) and set(systems) == {"Linux", "Windows", "Darwin"}, "build.systems must cover Linux, Windows, and Darwin")
        expect(build.get("cpuflags") == [], "portable public builds must not claim x86-only CPU flags")
    for preset_group in ("test_presets", "tune_presets", "datagen_presets"):
        value = config.get(preset_group)
        expect(isinstance(value, dict) and isinstance(value.get("default"), dict), f"{preset_group}.default is required")
    tests = config.get("test_presets", {})
    tunes = config.get("tune_presets", {})
    for name in ("STC", "LTC"):
        expect(isinstance(tests.get(name), dict), f"test_presets.{name} is required")
        expect(isinstance(tunes.get(name), dict), f"tune_presets.{name} is required")
    return errors


def official_schema_check(openbench_root: pathlib.Path, engine_config: pathlib.Path) -> None:
    script = """
import json, pathlib, sys
root, engine = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(root))
from OpenBench import config as schema
value = json.loads(engine.read_text(encoding='utf-8'))
schema.verify_engine_basics(value)
schema.verify_engine_build('Volkrix', value)
for preset in value['test_presets'].values(): schema.verify_engine_test_preset(preset)
for preset in value['tune_presets'].values(): schema.verify_engine_tune_preset(preset)
for preset in value['datagen_presets'].values(): schema.verify_engine_datagen_preset(preset)
print(schema.OPENBENCH_STATIC_VERSION)
"""
    observed = run([sys.executable, "-c", script, str(openbench_root), str(engine_config)])
    locked = strict_json(UPSTREAM_LOCK)["openbench"]["config_schema"]
    if observed != locked:
        raise DeployError(f"official schema is {observed}, but audited lock expects {locked}")


def command_version(name: str) -> tuple[str | None, tuple[int, ...]]:
    path = shutil.which(name)
    if path is None:
        return None, ()
    completed = subprocess.run(
        [path, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    first = completed.stdout.splitlines()[0] if completed.stdout else "version unavailable"
    return f"{path}: {first}", version_tuple(first)


def machine_memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        try:
            return int(run(["sysctl", "-n", "hw.memsize"]))
        except DeployError:
            return None
    memory = pathlib.Path("/proc/meminfo")
    if memory.is_file():
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", memory.read_text(), re.MULTILINE)
        if match:
            return int(match.group(1)) * 1024
    return None


def core_layout() -> dict[str, int | None]:
    logical = os.cpu_count()
    layout: dict[str, int | None] = {
        "logical": logical,
        "physical": logical,
        "performance": None,
        "efficiency": None,
    }
    if platform.system() != "Darwin":
        return layout
    try:
        performance_name = run(["sysctl", "-n", "hw.perflevel0.name"])
        if performance_name == "Performance":
            layout["performance"] = int(run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]))
            layout["efficiency"] = int(run(["sysctl", "-n", "hw.perflevel1.physicalcpu"]))
            layout["physical"] = int(layout["performance"] or 0) + int(layout["efficiency"] or 0)
    except (DeployError, ValueError):
        pass
    return layout


def audit(args: argparse.Namespace) -> dict[str, Any]:
    lock = strict_json(UPSTREAM_LOCK)
    config = strict_json(ENGINE_CONFIG)
    blockers = validate_engine_config(config)
    warnings: list[str] = []
    cores = core_layout()
    facts: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cores": cores,
        "memory_bytes": machine_memory_bytes(),
        "openbench_lock": lock,
        "tools": {},
    }

    if sys.version_info < (3, 9):
        blockers.append("Python 3.9 or newer is required by the current OpenBench server")
    try:
        import requests  # type: ignore[import-not-found]

        facts["python_requests"] = getattr(requests, "__version__", "installed")
    except ImportError:
        blockers.append("the OpenBench worker requires the Python requests package")
        facts["python_requests"] = None
    python311, _ = command_version("python3.11")
    facts["validated_server_python"] = python311
    if python311 is None:
        warnings.append("Python 3.11 is not installed; use uv/venv or another supported isolated server runtime")

    required_tools = ["git", "make", "cargo"]
    for tool in [*required_tools, "clang++", "g++", "docker"]:
        display, version = command_version(tool)
        facts["tools"][tool] = display
        if tool in required_tools and display is None:
            blockers.append(f"required worker tool {tool} is missing")
        if tool == "cargo" and display is not None and version < (1, 85, 0):
            blockers.append(f"Cargo {version} is older than required 1.85.0")
    if facts["tools"]["clang++"] is None and facts["tools"]["g++"] is None:
        blockers.append("a C++ compiler is required to build FastChess")
    if facts["tools"]["docker"] is None:
        warnings.append("Docker is absent; this is not a worker blocker and upstream has no official container deployment")

    current_system = platform.system()
    if current_system not in config.get("build", {}).get("systems", []):
        blockers.append(f"engine configuration does not support {current_system}")

    if args.openbench_root:
        upstream = args.openbench_root.resolve(strict=True)
        head = git(upstream, "rev-parse", "HEAD")
        facts["openbench_head"] = head
        if head != lock["openbench"]["commit"]:
            blockers.append(f"OpenBench checkout {head} differs from audited {lock['openbench']['commit']}")
        try:
            official_schema_check(upstream, ENGINE_CONFIG)
        except DeployError as error:
            blockers.append(str(error))
        upstream_config = strict_json(upstream / "Config" / "config.json")
        if upstream_config.get("client_version") != lock["openbench"]["client_version"]:
            blockers.append("upstream client_version differs from the audited lock")
        if upstream_config.get("fastchess_min_version") != lock["fastchess"]["minimum_version"]:
            blockers.append("upstream FastChess minimum differs from the audited lock")
        requirement_lines = (upstream / "requirements.txt").read_text(encoding="utf-8").splitlines()
        expected_django = f"Django=={lock['upstream_django_pin']}"
        if requirement_lines.count(expected_django) != 1:
            blockers.append("upstream Django requirement differs from the audited dependency lock")
        else:
            warnings.append(
                f"upstream pins unsupported Django {lock['upstream_django_pin']}; prepared instances replace it with supported {lock['deployment_django']} LTS"
            )
        for dependency in ("requests", "scipy"):
            if requirement_lines.count(dependency) != 1:
                blockers.append(f"upstream {dependency} requirement differs from the audited dependency shape")

    if args.fastchess_root:
        fastchess = args.fastchess_root.resolve(strict=True)
        head = git(fastchess, "rev-parse", "HEAD")
        facts["fastchess_head"] = head
        if head != lock["fastchess"]["commit"]:
            blockers.append(f"FastChess checkout {head} differs from audited {lock['fastchess']['commit']}")

    if args.network:
        network = args.network.resolve(strict=True)
        digest = sha256_file(network)
        facts["network"] = {"path": str(network), "sha256": digest, "size": network.stat().st_size}
        expected = "c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7"
        if digest != expected:
            blockers.append(f"production network digest is {digest}, expected {expected}")

    if config.get("nps") == 1_000_000:
        warnings.append("engine nps is still the example value; calibrate the finalized embedded binary on the reference worker")

    worker_threads = cores["performance"] or cores["physical"] or cores["logical"] or 1
    if cores["performance"] and cores["efficiency"]:
        warnings.append(
            "heterogeneous Apple cores detected; start the reference worker on performance-core count only and verify concurrent bench stability"
        )
    facts["recommended_worker_command"] = (
        "OPENBENCH_USERNAME=... OPENBENCH_PASSWORD=... OPENBENCH_SERVER=https://... "
        f"python3 Client/client.py --no-client-downloads -T {worker_threads} -N 1 --focus Volkrix"
    )
    external = [
        "merge and push the finalized Volkrix commit to the public source repository",
        "choose a stable server host/domain, production MySQL database, TLS termination, backups, and monitoring",
        "generate server secrets and create/approve OpenBench administrator, user, and worker accounts",
        "upload the verified production NNUE through the OpenBench network administration UI and select it for workloads",
        "calibrate nps with at least five concurrent bench sets on the designated reference worker",
        "review and retain the opening-book source and license record",
        "supply worker credentials at runtime without committing them",
    ]
    return {"ready": not blockers, "blockers": blockers, "warnings": warnings, "facts": facts, "external_remaining": external}


def copy_instance(source: pathlib.Path, output: pathlib.Path) -> None:
    ignored = shutil.ignore_patterns(".git", "db.sqlite3", "Media", "__pycache__", "*.pyc")
    shutil.copytree(source, output, ignore=ignored)


def harden_requirements(path: pathlib.Path, lock: dict[str, Any]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    replacements = {
        f"Django=={lock['upstream_django_pin']}": f"Django=={lock['deployment_django']}",
        "requests": f"requests=={lock['deployment_requests']}",
        "scipy": f"scipy=={lock['deployment_scipy']}",
    }
    for expected in replacements:
        if lines.count(expected) != 1:
            raise DeployError(
                f"expected exactly one {expected} in pinned upstream requirements; "
                "re-audit dependency compatibility"
            )
    path.write_text(
        "\n".join(replacements.get(line, line) for line in lines) + "\n",
        encoding="utf-8",
    )


TIMEZONE_OBSOLETE = (
    "    target = datetime.datetime.utcnow()\n"
    "    target = target.replace(tzinfo=timezone.utc)\n"
)
TIMEZONE_FILES = (
    pathlib.Path("OpenBench/utils.py"),
    pathlib.Path("OpenBench/workloads/view_workload.py"),
)
WATCHER_PATCHES = {
    pathlib.Path("OpenBench/apps.py"): (
        "import atexit\nimport pathlib\n",
        "import atexit\nimport os\nimport pathlib\n",
        "        # Attempt to spawn the PGN Watcher, globally once\n\n"
        "        from OpenBench.pgn_watcher import PGNWatcher\n",
        "        # Management commands must not start a background database watcher.\n"
        "        if os.environ.get('OPENBENCH_DISABLE_WATCHER', '').lower() in {'1', 'true', 'yes'}:\n"
        "            return\n\n"
        "        # Attempt to spawn the PGN Watcher, globally once\n\n"
        "        from OpenBench.pgn_watcher import PGNWatcher\n",
    ),
    pathlib.Path("OpenBench/pgn_watcher.py"): (
        "    def run(self):\n\n"
        "        # Loop until we are shutdown by the atexit.register()\n"
        "        while not self.stop_event.is_set():\n",
        "    def run(self):\n\n"
        "        # AppConfig.ready() launches this thread before Django marks the app\n"
        "        # registry ready. Delay the first query to avoid initialization races.\n"
        "        if self.stop_event.wait(timeout=1):\n"
        "            return\n\n"
        "        # Loop until we are shutdown by the atexit.register()\n"
        "        while not self.stop_event.is_set():\n",
    ),
}


def validate_django_timezone_compatibility(instance: pathlib.Path) -> None:
    for relative in TIMEZONE_FILES:
        contents = (instance / relative).read_text(encoding="utf-8")
        if contents.count(TIMEZONE_OBSOLETE) != 1:
            raise DeployError(
                f"expected exactly one audited timezone compatibility site in {relative}; "
                "re-audit the pinned OpenBench revision"
            )


def validate_watcher_compatibility(instance: pathlib.Path) -> None:
    for relative, patches in WATCHER_PATCHES.items():
        contents = (instance / relative).read_text(encoding="utf-8")
        for obsolete in patches[::2]:
            if contents.count(obsolete) != 1:
                raise DeployError(
                    f"expected exactly one audited watcher compatibility site in {relative}; "
                    "re-audit the pinned OpenBench revision"
                )


def harden_django_timezone_compatibility(instance: pathlib.Path) -> list[pathlib.Path]:
    """Replace the two audited uses of an API removed before Django 5.2."""
    validate_django_timezone_compatibility(instance)
    changed: list[pathlib.Path] = []
    for relative in TIMEZONE_FILES:
        path = instance / relative
        contents = path.read_text(encoding="utf-8")
        path.write_text(
            contents.replace(TIMEZONE_OBSOLETE, "    target = timezone.now()\n"),
            encoding="utf-8",
        )
        changed.append(relative)
    return changed


def harden_watcher_compatibility(instance: pathlib.Path) -> list[pathlib.Path]:
    """Keep management commands watcher-free and delay the runtime's first query."""
    validate_watcher_compatibility(instance)
    for relative, patches in WATCHER_PATCHES.items():
        path = instance / relative
        contents = path.read_text(encoding="utf-8")
        for obsolete, replacement in zip(patches[::2], patches[1::2]):
            contents = contents.replace(obsolete, replacement)
        path.write_text(contents, encoding="utf-8")
    return list(WATCHER_PATCHES)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = args.openbench_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise DeployError(f"output already exists: {output}")
    if isinstance(args.nps, bool) or not isinstance(args.nps, int) or args.nps <= 0:
        raise DeployError("--nps must be a positive integer")
    if not SAFE_BOOK_RE.fullmatch(args.book):
        raise DeployError("--book must be a plain EPD/PGN configuration name")
    lock = strict_json(UPSTREAM_LOCK)
    head = git(source, "rev-parse", "HEAD")
    if head != lock["openbench"]["commit"]:
        raise DeployError(f"OpenBench checkout {head} differs from audited {lock['openbench']['commit']}")
    if git(source, "status", "--porcelain"):
        raise DeployError("OpenBench source checkout is not clean")
    if not COMMIT_RE.fullmatch(args.client_ref) or not COMMIT_RE.fullmatch(args.fastchess_ref):
        raise DeployError("client and FastChess refs must be full 40-character commits")
    if args.client_ref != lock["openbench"]["commit"]:
        raise DeployError("client ref differs from the audited OpenBench commit")
    if args.fastchess_ref != lock["fastchess"]["commit"]:
        raise DeployError("FastChess ref differs from the audited commit")
    if not args.client_repo_url.startswith("https://github.com/") or any(
        character in args.client_repo_url for character in "\r\n"
    ):
        raise DeployError("client repository must be an HTTPS GitHub URL")
    official_schema_check(source, ENGINE_CONFIG)

    source_requirements = source / "requirements.txt"
    requirements = source_requirements.read_text(encoding="utf-8").splitlines()
    for expected in (f"Django=={lock['upstream_django_pin']}", "requests", "scipy"):
        if requirements.count(expected) != 1:
            raise DeployError(
                f"expected exactly one {expected} in pinned upstream requirements; "
                "re-audit dependency compatibility"
            )
    validate_django_timezone_compatibility(source)
    validate_watcher_compatibility(source)

    source_book_config = source / "Books" / f"{args.book}.json"
    if not source_book_config.is_file():
        raise DeployError(f"pinned OpenBench checkout has no configuration for {args.book}")

    copy_instance(source, output)
    harden_requirements(output / "requirements.txt", lock)
    compatibility_files = harden_django_timezone_compatibility(output)
    compatibility_files.extend(harden_watcher_compatibility(output))
    config_path = output / "Config" / "config.json"
    config = strict_json(config_path)
    config.update(
        {
            "client_repo_url": args.client_repo_url,
            "client_repo_ref": args.client_ref,
            "fastchess_repo_url": lock["fastchess"]["repository"],
            "fastchess_repo_ref": args.fastchess_ref,
            "use_cross_approval": False,
            "require_login_to_view": True,
            "require_manual_registration": True,
            "balance_engine_throughputs": False,
            "books": [args.book],
            "engines": ["Volkrix"],
        }
    )
    book_config = output / "Books" / f"{args.book}.json"
    if not book_config.is_file():
        raise DeployError(f"copied instance lost configuration for {args.book}")
    engine = strict_json(ENGINE_CONFIG)
    engine["nps"] = args.nps
    engine_path = output / "Engines" / "Volkrix.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    engine_path.write_text(json.dumps(engine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(DEPLOYMENT_ASSETS / "volkrix_settings.py", output / "OpenSite" / "volkrix_settings.py")
    shutil.copy2(DEPLOYMENT_ASSETS / "requirements.txt", output / "requirements-deploy.txt")
    shutil.copy2(DEPLOYMENT_ASSETS / "openbench.env.example", output / "openbench.env.example")
    shutil.copy2(DEPLOYMENT_ASSETS / "DEPLOYMENT.md", output / "DEPLOYMENT.md")

    manifest = {
        "schema": "volkrix-openbench-deployment-v1",
        "openbench_repository": lock["openbench"]["repository"],
        "openbench_commit": head,
        "client_repository": args.client_repo_url,
        "client_commit": args.client_ref,
        "fastchess_repository": lock["fastchess"]["repository"],
        "fastchess_commit": args.fastchess_ref,
        "book": args.book,
        "nps": args.nps,
        "compatibility_patches": [str(path) for path in compatibility_files],
        "files": {
            "Config/config.json": sha256_file(config_path),
            "Engines/Volkrix.json": sha256_file(engine_path),
            f"Books/{args.book}.json": sha256_file(book_config),
            "OpenSite/volkrix_settings.py": sha256_file(output / "OpenSite" / "volkrix_settings.py"),
            "openbench.env.example": sha256_file(output / "openbench.env.example"),
            "requirements-deploy.txt": sha256_file(output / "requirements-deploy.txt"),
            "requirements.txt": sha256_file(output / "requirements.txt"),
            "DEPLOYMENT.md": sha256_file(output / "DEPLOYMENT.md"),
            **{str(path): sha256_file(output / path) for path in compatibility_files},
        },
    }
    manifest_path = output / "VOLKRIX-DEPLOYMENT.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--openbench-root", type=pathlib.Path)
    audit_parser.add_argument("--fastchess-root", type=pathlib.Path)
    audit_parser.add_argument("--network", type=pathlib.Path)
    audit_parser.add_argument("--json", action="store_true")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--openbench-root", type=pathlib.Path, required=True)
    prepare_parser.add_argument("--output", type=pathlib.Path, required=True)
    prepare_parser.add_argument("--nps", type=int, required=True)
    prepare_parser.add_argument("--book", default="UHO_Lichess_4852_v1.epd")
    lock = strict_json(UPSTREAM_LOCK)
    prepare_parser.add_argument("--client-repo-url", default=lock["openbench"]["repository"])
    prepare_parser.add_argument("--client-ref", default=lock["openbench"]["commit"])
    prepare_parser.add_argument("--fastchess-ref", default=lock["fastchess"]["commit"])
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "audit":
            report = audit(args)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print("OpenBench deployment readiness:", "READY" if report["ready"] else "BLOCKED")
                for label in ("blockers", "warnings", "external_remaining"):
                    print(f"\n{label.replace('_', ' ').title()}:")
                    for item in report[label]:
                        print(f"- {item}")
                print("\nFacts:")
                print(json.dumps(report["facts"], indent=2, sort_keys=True))
            return 0 if report["ready"] else 2
        if args.nps <= 0:
            raise DeployError("--nps must be positive")
        manifest = prepare(args)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (DeployError, OSError) as error:
        print(f"OpenBench deployment error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
