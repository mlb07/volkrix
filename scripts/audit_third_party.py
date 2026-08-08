#!/usr/bin/env python3
"""Fail-closed audit of locked Rust licenses and vendored artifact provenance."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent
APPROVED_LICENSES = {
    "Apache-2.0 OR MIT",
    "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
    "BSD-2-Clause OR Apache-2.0 OR MIT",
    "MIT",
    "MIT OR Apache-2.0",
    "Unlicense OR MIT",
    "(MIT OR Apache-2.0) AND Unicode-3.0",
    "MIT OR Apache-2.0 OR LGPL-2.1-or-later",
}
OFFLINE_COPYLEFT_EXCEPTIONS = {
    ("montyformat", "0.9.2", "AGPL-3.0"),
    ("sfbinpack", "0.6.2", "GPL-3.0"),
}
BULLET_REVISION = "feab6443fc523c9d349427bca2d5bb3c04369420"
BULLET_SOURCE_PREFIX = (
    f"git+https://github.com/jw1912/bullet?rev={BULLET_REVISION}#{BULLET_REVISION}"
)
VENDOR_TREE_SHA256 = {
    "fathom": "a422b0e87d14f05dfd70bf4bc1d98d0a56a644ec80946d74f3a11df233712065",
    "nnue-rs": "65a9734068ea169a4a892d03c8997f0a28ee1833f8719292aa75b8360a349355",
}
FATHOM_REVISION = "c9c6fef0dddc05d2e242c183acf5833149ab676d"
NNUE_RS_REVISION = "64ba58a93224e18ae48c19a6c58f34026f237730"
NETWORK_SHA256 = (
    "c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7",
    "37f18f62d772f3107e1d6aaca3898c130c3c86f2ab63e6555fbbca20635a899d",
)


class AuditFailure(RuntimeError):
    pass


def cargo_metadata(root: pathlib.Path = ROOT) -> dict[str, Any]:
    result = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AuditFailure(f"cargo metadata --locked failed:\n{result.stderr.rstrip()}")
    return json.loads(result.stdout)


def production_dependency_ids(metadata: dict[str, Any]) -> set[str]:
    packages = metadata["packages"]
    roots = [
        package
        for package in packages
        if package["name"] == "volkrix"
        and package["source"] is None
        and pathlib.Path(package["manifest_path"]).resolve() == ROOT / "Cargo.toml"
    ]
    if len(roots) != 1:
        raise AuditFailure("could not identify exactly one Volkrix engine package")
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    pending = [roots[0]["id"]]
    visited: set[str] = set()
    while pending:
        package_id = pending.pop()
        if package_id in visited:
            continue
        visited.add(package_id)
        for dependency in nodes[package_id]["deps"]:
            kinds = dependency["dep_kinds"]
            if any(kind["kind"] in (None, "build") for kind in kinds):
                pending.append(dependency["pkg"])
    return visited


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    packages = metadata["packages"]
    production = production_dependency_ids(metadata)
    reviewed: list[str] = []
    expected_git_packages = {"acyclib", "bullet_lib"}
    seen_git_packages: set[str] = set()

    for package in packages:
        name = package["name"]
        version = package["version"]
        source = package["source"]
        license_expression = package.get("license")
        key = (name, version, license_expression)

        if source is None:
            if name not in {"volkrix", "volkrix-nnue"}:
                raise AuditFailure(f"unexpected unregistered local package: {name} {version}")
            continue
        if not license_expression:
            raise AuditFailure(f"dependency has no SPDX license expression: {name} {version}")
        if license_expression not in APPROVED_LICENSES:
            if key not in OFFLINE_COPYLEFT_EXCEPTIONS:
                raise AuditFailure(
                    f"unreviewed license {license_expression!r}: {name} {version}"
                )
            if package["id"] in production:
                raise AuditFailure(
                    f"offline-only copyleft dependency reached engine graph: {name} {version}"
                )

        if source.startswith("git+"):
            seen_git_packages.add(name)
            if name not in expected_git_packages or source != BULLET_SOURCE_PREFIX:
                raise AuditFailure(f"unreviewed or unpinned git source: {name} {source}")
        elif not source.startswith("registry+"):
            raise AuditFailure(f"unsupported dependency source: {name} {source}")
        reviewed.append(f"{name} {version} [{license_expression}]")

    if seen_git_packages != expected_git_packages:
        raise AuditFailure(
            "locked Bullet packages changed: "
            f"expected {sorted(expected_git_packages)}, got {sorted(seen_git_packages)}"
        )
    return sorted(reviewed)


def tree_sha256(directory: pathlib.Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise AuditFailure(f"vendored tree is empty: {directory}")
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def require_contains(path: pathlib.Path, values: tuple[str, ...]) -> None:
    if not path.is_file():
        raise AuditFailure(f"required provenance file is missing: {path}")
    text = path.read_text(encoding="utf-8")
    for value in values:
        if value not in text:
            raise AuditFailure(f"{path.relative_to(ROOT)} does not record {value!r}")


def validate_provenance(root: pathlib.Path = ROOT) -> list[str]:
    for name, expected in VENDOR_TREE_SHA256.items():
        actual = tree_sha256(root / "vendor" / name)
        if actual != expected:
            raise AuditFailure(
                f"vendor/{name} tree hash changed: expected {expected}, got {actual}"
            )

    require_contains(
        root / "vendor" / "fathom" / "UPSTREAM.md",
        (FATHOM_REVISION, "https://github.com/jdart1/Fathom"),
    )
    require_contains(
        root / "vendor" / "nnue-rs" / "VOLKRIX_FORK.md",
        (NNUE_RS_REVISION, "https://github.com/hedgeg0d/nnue-rs"),
    )
    require_contains(
        root / "vendor" / "nnue-rs" / "Cargo.toml.upstream",
        ('name = "nnue-rs"', 'version = "0.4.0"', 'license = "MIT"'),
    )
    if (root / "vendor" / "nnue-rs" / "Cargo.toml").exists():
        raise AuditFailure("vendor/nnue-rs/Cargo.toml would break Volkrix source packaging")

    notice_values = (
        FATHOM_REVISION,
        NNUE_RS_REVISION,
        BULLET_REVISION,
        "montyformat 0.9.2",
        "sfbinpack 0.6.2",
        *NETWORK_SHA256,
    )
    require_contains(root / "THIRD_PARTY_NOTICES.md", notice_values)
    require_contains(root / "scripts" / "fetch-stockfish18-net.sh", NETWORK_SHA256)
    require_contains(
        root / "Cargo.lock",
        (BULLET_REVISION, 'name = "montyformat"', 'name = "sfbinpack"'),
    )
    return [
        f"vendor/{name} sha256={digest}"
        for name, digest in sorted(VENDOR_TREE_SHA256.items())
    ]


def audit(root: pathlib.Path = ROOT) -> list[str]:
    if root.resolve() != ROOT:
        raise AuditFailure("alternate roots are not supported by the provenance audit")
    dependencies = validate_metadata(cargo_metadata(root))
    provenance = validate_provenance(root)
    return [
        f"locked external dependencies reviewed: {len(dependencies)}",
        *provenance,
    ]


def main() -> int:
    try:
        lines = audit()
    except (AuditFailure, OSError, json.JSONDecodeError) as error:
        print(f"third-party audit failed: {error}", file=sys.stderr)
        return 1
    print("third-party license and provenance audit passed")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
