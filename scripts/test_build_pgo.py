#!/usr/bin/env python3
"""Regression tests for PGO source provenance."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_pgo", ROOT / "scripts" / "build_pgo.py")
assert SPEC is not None and SPEC.loader is not None
build_pgo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_pgo)


def git(root: pathlib.Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class SourceProvenanceTests(unittest.TestCase):
    def test_dirty_identity_covers_tracked_and_untracked_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            git(root, "init")
            git(root, "config", "user.email", "test@example.invalid")
            git(root, "config", "user.name", "Volkrix Test")
            (root / "Cargo.lock").write_bytes(b"lock-v1\n")
            (root / "tracked.bin").write_bytes(b"tracked-v1\x00")
            git(root, "add", "Cargo.lock", "tracked.bin")
            git(root, "commit", "-m", "fixture")

            clean = build_pgo.source_provenance(root)
            self.assertFalse(clean["source_dirty"])
            self.assertEqual(clean["source_id"], clean["source_commit"])
            self.assertEqual(clean["untracked_paths"], [])

            (root / "tracked.bin").write_bytes(b"tracked-v2\x00\xff")
            tracked_dirty = build_pgo.source_provenance(root)
            self.assertTrue(tracked_dirty["source_dirty"])
            self.assertIn("-dirty-", tracked_dirty["source_id"])
            self.assertNotEqual(
                clean["tracked_diff_sha256"], tracked_dirty["tracked_diff_sha256"]
            )
            self.assertNotEqual(
                clean["source_tree_sha256"], tracked_dirty["source_tree_sha256"]
            )

            (root / "untracked.bin").write_bytes(b"untracked\x00\xff")
            with_untracked = build_pgo.source_provenance(root)
            self.assertEqual(with_untracked["untracked_paths"], ["untracked.bin"])
            self.assertNotEqual(
                tracked_dirty["untracked_sha256"], with_untracked["untracked_sha256"]
            )
            self.assertNotEqual(
                tracked_dirty["source_tree_sha256"], with_untracked["source_tree_sha256"]
            )

            (root / "Cargo.lock").write_bytes(b"lock-v2\n")
            lock_changed = build_pgo.source_provenance(root)
            self.assertNotEqual(
                with_untracked["cargo_lock_sha256"], lock_changed["cargo_lock_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
