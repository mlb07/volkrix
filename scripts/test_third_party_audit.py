#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "audit_third_party", SCRIPT_DIR / "audit_third_party.py"
)
assert SPEC is not None and SPEC.loader is not None
audit_third_party = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_third_party
SPEC.loader.exec_module(audit_third_party)


class ThirdPartyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = audit_third_party.cargo_metadata()

    def test_repository_license_and_provenance_audit_passes(self) -> None:
        self.assertTrue(audit_third_party.audit())

    def test_unknown_license_fails_closed(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        dependency = next(
            package for package in metadata["packages"] if package["name"] == "cc"
        )
        dependency["license"] = "UNKNOWN"
        with self.assertRaisesRegex(audit_third_party.AuditFailure, "unreviewed license"):
            audit_third_party.validate_metadata(metadata)

    def test_unpinned_git_dependency_fails_closed(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        dependency = next(
            package
            for package in metadata["packages"]
            if package["name"] == "bullet_lib"
        )
        dependency["source"] = "git+https://github.com/jw1912/bullet#floating"
        with self.assertRaisesRegex(audit_third_party.AuditFailure, "unpinned git"):
            audit_third_party.validate_metadata(metadata)


if __name__ == "__main__":
    unittest.main()
