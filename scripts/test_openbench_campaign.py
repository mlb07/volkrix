#!/usr/bin/env python3

from __future__ import annotations

import bz2
import importlib.util
import io
import json
import pathlib
import tarfile
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("openbench_campaign.py")
SPEC = importlib.util.spec_from_file_location("openbench_campaign", MODULE_PATH)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


BOOK = "7a7f6470615a69c6cf23d565417701d38732876f480af90d67b42abade35644a"
NETWORK = "c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7"
DEV = "1" * 40
BASE = "2" * 40


def side(commit: str) -> dict[str, object]:
    return {"commit": commit, "bench": 187557, "network_sha256": NETWORK, "options": "Threads=1 Hash=16"}


def deployment_audit() -> dict[str, object]:
    lock = campaign.strict_json(campaign.UPSTREAM_LOCK)
    return {
        "schema": "volkrix-openbench-deployment-audit-v1",
        "ready": True,
        "blockers": [],
        "warnings": [],
        "external_remaining": [],
        "facts": {
            "platform": "Linux",
            "machine": "x86_64",
            "cores": {"logical": 4, "physical": 4, "performance": None, "efficiency": None},
            "memory_bytes": 8 * 1024**3,
            "openbench_lock": lock,
            "tools": {"git": "git 2", "make": "make 4", "cargo": "cargo 1.85", "clang++": "clang 18", "g++": None, "docker": None},
            "python_requests": "2.34.2",
            "validated_server_python": "Python 3.11",
            "openbench_head": lock["openbench"]["commit"],
            "fastchess_head": lock["fastchess"]["commit"],
            "network": {"path": "/srv/nn-c288c895ea92.nnue", "sha256": NETWORK, "size": 108_919_594},
            "recommended_worker_command": "python3 Client/client.py -T 4 -N 1 --focus Volkrix",
        },
    }


def specimen(kind: str = "stc", *, book: str = BOOK) -> dict[str, object]:
    lock = campaign.strict_json(campaign.UPSTREAM_LOCK)
    result: dict[str, object] = {
        "schema": "volkrix-openbench-workload-v1",
        "kind": kind,
        "campaign": "fixture",
        "engine": "Volkrix",
        "source_repo": "https://github.com/mlb07/volkrix",
        "dev": side(DEV),
        "base": side(BASE),
        "test": {
            "mode": "SPRT",
            "book": {"name": "UHO_Lichess_4852_v1.epd", "sha256": book},
            "time_control": "10.0+0.1",
            "upload_pgns": "COMPACT",
            "workload_size": 32,
            "scale": {"method": "BOTH", "nps": 1_000_000},
            "adjudication": {"syzygy_wdl": "DISABLED", "syzygy_adj": "DISABLED", "win": "None", "draw": "None"},
            "bounds": [0.0, 3.0],
            "confidence": [0.05, 0.05],
        },
        "server": {
            "openbench_commit": lock["openbench"]["commit"],
            "fastchess_commit": lock["fastchess"]["commit"],
            "client_version": lock["openbench"]["client_version"],
            "network_name": "nn-c288c895ea92.nnue",
            "reference_nps": 1_000_000,
        },
        "worker": {"threads": 4, "sockets": 1, "focus": "Volkrix", "min_bench_samples": 5, "max_bench_cv": 0.03, "max_nps_deviation": 0.03},
    }
    if kind == "no-change":
        result["base"] = side(DEV)
        result["test"]["mode"] = "GAMES"  # type: ignore[index]
        result["test"]["max_games"] = 2000  # type: ignore[index]
        result["test"].pop("bounds")  # type: ignore[union-attr]
        result["test"].pop("confidence")  # type: ignore[union-attr]
        result["policy"] = {"max_abs_elo": 10.0, "require_ci_contains_zero": True}
    elif kind == "ltc":
        result["test"]["time_control"] = "60.0+0.6"  # type: ignore[index]
        result["test"]["workload_size"] = 8  # type: ignore[index]
        result["dev"]["options"] = "Threads=1 Hash=64"  # type: ignore[index]
        result["base"]["options"] = "Threads=1 Hash=64"  # type: ignore[index]
    elif kind == "spsa":
        result.pop("base")
        result["test"]["mode"] = "SPSA"  # type: ignore[index]
        result["test"]["workload_size"] = 8  # type: ignore[index]
        result["test"].pop("bounds")  # type: ignore[union-attr]
        result["test"].pop("confidence")  # type: ignore[union-attr]
        result["spsa"] = {"reporting": "BATCHED", "distribution": "SINGLE", "alpha": 0.602, "gamma": 0.101,
                          "a_ratio": 0.1, "iterations": 1000, "pairs_per": 8,
                          "inputs": [{"name": "TuneSeeMargin", "type": "int", "start": 70, "min": 20, "max": 140, "c_end": 1, "r_end": 0.002}]}
    return result


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def pgn_archive(games: int = 1) -> bytes:
    game = b'[Event "fixture"]\n[Result "1/2-1/2"]\n\n1. e4 e5 1/2-1/2\n\n'
    payload = bz2.compress(game * games)
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        member = tarfile.TarInfo("7.1.0.pgn.bz2")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


class FakeClient:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses

    def request(self, endpoint: str, *, binary: bool = False) -> object:
        return self.responses[endpoint]


def workload_info(
    spec: dict[str, object],
    digest: str,
    kind: str = "stc",
    penta: list[int] | None = None,
) -> dict[str, object]:
    test = spec["test"]
    dev, base = spec["dev"], spec.get("base")
    penta = penta or [10, 20, 40, 25, 15]
    info: dict[str, object] = {
        "id": 7, "info": f"volkrix-campaign-sha256={digest}", "book_name": test["book"]["name"], "test_mode": test["mode"],  # type: ignore[index]
        "dev_engine": "Volkrix", "base_engine": "Volkrix",
        "dev_repo": spec["source_repo"], "dev": {"sha": dev["commit"], "bench": dev["bench"]}, "dev_options": dev["options"],  # type: ignore[index]
        "dev_network": NETWORK[:8].upper(), "dev_time_control": campaign.normalize_time_control(test["time_control"]),  # type: ignore[index]
        "base_repo": spec["source_repo"], "base": {"sha": base["commit"], "bench": base["bench"]}, "base_options": base["options"],  # type: ignore[index]
        "base_network": NETWORK[:8].upper(), "base_time_control": campaign.normalize_time_control(test["time_control"]),  # type: ignore[index]
        "scale_method": test["scale"]["method"], "scale_nps": test["scale"]["nps"], "upload_pgns": test["upload_pgns"],  # type: ignore[index]
        "priority": 0, "throughput": 1,
        "workload_size": test["workload_size"], "syzygy_wdl": test["adjudication"]["syzygy_wdl"], "syzygy_adj": test["adjudication"]["syzygy_adj"],  # type: ignore[index]
        "win_adj": test["adjudication"]["win"], "draw_adj": test["adjudication"]["draw"], "finished": True, "error": False, "deleted": False,  # type: ignore[index]
        "passed": True, "failed": False, "penta": penta, "games": sum(penta) * 2,
    }
    if kind == "no-change":
        info["max_games"] = test["max_games"]  # type: ignore[index]
    else:
        info.update({"elolower": 0.0, "eloupper": 3.0, "beta": 0.05, "alpha": 0.05})
    return info


def export_fixture(root: pathlib.Path, spec: dict[str, object], kind: str) -> pathlib.Path:
    digest = campaign.digest_value(spec)
    penta = [100, 200, 500, 200, 100] if kind == "no-change" else None
    info = workload_info(spec, digest, kind, penta)
    values = info["penta"]
    row = {
        "games": info["games"],
        "LL": values[0],
        "LD": values[1],
        "DD": values[2],
        "DW": values[3],
        "WW": values[4],
        "crashes": 0,
        "timeloss": 0,
        "active": False,
    }
    responses = {
        "api/workload/7/info/": {"info": info},
        "api/workload/7/results/": {"results": [row]},
        "api/workload/7/summary/": {"summary": {}},
        "api/pgns/7/": pgn_archive(info["games"]),
    }
    preflight = {
        "schema": "volkrix-openbench-preflight-v1",
        "ready": True,
        "workload_sha256": digest,
    }
    output = root / kind
    campaign.export_result(spec, digest, 7, FakeClient(responses), output, preflight)
    return output


class CampaignTests(unittest.TestCase):
    def test_freeze_is_canonical_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, locked = root / "source.json", root / "locked.json"
            write_json(source, specimen())
            result = campaign.freeze(source, locked)
            spec, digest = campaign.load_lock(locked)
            self.assertEqual(result["sha256"], digest)
            self.assertEqual(campaign.form_payload(spec, digest)["dev_branch"], DEV)
            self.assertEqual(campaign.form_payload(spec, digest)["dev_network"], NETWORK[:8].upper())
            self.assertEqual(campaign.form_payload(spec, digest)["base_network"], NETWORK[:8].upper())
            value = json.loads(locked.read_text())
            value["spec"]["dev"]["bench"] += 1
            write_json(locked, value)
            with self.assertRaisesRegex(campaign.CampaignError, "digest mismatch"):
                campaign.load_lock(locked)

    def test_no_change_rejects_any_nonidentical_input(self) -> None:
        value = specimen("no-change")
        campaign.validate_spec(value)
        value["base"]["options"] = "Threads=1 Hash=32"  # type: ignore[index]
        with self.assertRaisesRegex(campaign.CampaignError, "identical"):
            campaign.validate_spec(value)

    def test_workloads_require_the_exact_production_network(self) -> None:
        value = specimen()
        value["dev"]["network_sha256"] = "0" * 64  # type: ignore[index]
        with self.assertRaisesRegex(campaign.CampaignError, "frozen production network"):
            campaign.validate_spec(value)

        value = specimen()
        value["base"]["network_sha256"] = "0" * 64  # type: ignore[index]
        with self.assertRaisesRegex(campaign.CampaignError, "frozen production network"):
            campaign.validate_spec(value)

    def test_time_controls_reject_zero_and_nonfinite_values(self) -> None:
        for value in ("N=0", "D=0", "MT=0", "inf+0.1", "1.0+inf"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(campaign.CampaignError, "invalid time control"):
                    campaign.normalize_time_control(value)

    def test_worker_record_consumes_full_deployment_audit_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec = specimen()
            audit = deployment_audit()
            audit_path, output = root / "audit.json", root / "worker.json"
            write_json(audit_path, audit)
            campaign.create_worker_record(spec, audit_path, "990000,1000000,1010000,1005000,995000", output)
            self.assertTrue(output.is_file())
            output.unlink()
            with self.assertRaisesRegex(campaign.CampaignError, "coefficient"):
                campaign.create_worker_record(spec, audit_path, "500000,1000000,1500000,1000000,1000000", output)
            self.assertFalse(output.exists())

            write_json(audit_path, {"ready": True, "facts": audit["facts"]})
            with self.assertRaisesRegex(campaign.CampaignError, "deployment audit"):
                campaign.create_worker_record(
                    spec,
                    audit_path,
                    "1000000,1000000,1000000,1000000,1000000",
                    output,
                )
            self.assertFalse(output.exists())

    def test_server_preflight_verifies_all_pins_assets_and_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec = specimen()
            lock = campaign.strict_json(campaign.UPSTREAM_LOCK)
            audit = deployment_audit()
            audit_path, worker_path = root / "audit.json", root / "worker.json"
            write_json(audit_path, audit)
            campaign.create_worker_record(spec, audit_path, "1000000,1000000,1000000,1000000,1000000", worker_path)
            engine = {"source": spec["source_repo"], "nps": 1_000_000, "build": {"path": "openbench", "compilers": ["cargo>=1.85.0"], "cpuflags": [], "systems": ["Linux", "Windows", "Darwin"]}}
            responses = {
                "clientVersionRef/": {"client_version": lock["openbench"]["client_version"], "client_repo_url": lock["openbench"]["repository"], "client_repo_ref": lock["openbench"]["commit"]},
                "clientMatchRunnerVersionRef/": {"fastchess_min_version": lock["fastchess"]["minimum_version"], "fastchess_repo_url": lock["fastchess"]["repository"], "fastchess_repo_ref": lock["fastchess"]["commit"]},
                "api/config/": {"engines": ["Volkrix"], "books": {"UHO_Lichess_4852_v1.epd": {"sha": BOOK.upper(), "source": "official"}}},
                "api/config/Volkrix/": engine,
                "api/networks/Volkrix/": {"default": {"name": "nn-c288c895ea92.nnue", "sha256": NETWORK[:8].upper()}, "networks": []},
            }
            report = campaign.preflight(spec, "a" * 64, FakeClient(responses), worker_path)
            self.assertTrue(report["ready"])
            responses["clientVersionRef/"]["client_repo_ref"] = "0" * 40  # type: ignore[index]
            with self.assertRaisesRegex(campaign.CampaignError, "client pin"):
                campaign.preflight(spec, "a" * 64, FakeClient(responses), worker_path)
            responses["clientVersionRef/"]["client_repo_ref"] = lock["openbench"]["commit"]  # type: ignore[index]
            responses["api/networks/Volkrix/"]["default"]["sha256"] = NETWORK[:8]  # type: ignore[index]
            with self.assertRaisesRegex(campaign.CampaignError, "default network"):
                campaign.preflight(spec, "a" * 64, FakeClient(responses), worker_path)

    def test_spsa_exports_are_numeric_ordered_complete_and_bounded(self) -> None:
        spec = specimen("spsa")
        campaign.validate_spec(spec)
        campaign.validate_spsa_input_export(
            "TuneSeeMargin, int, 70.0, 20.0, 140.0, 1.0, 0.002", spec
        )
        campaign.validate_spsa_outputs("TuneSeeMargin, 72", spec)
        with self.assertRaisesRegex(campaign.CampaignError, "out of bounds"):
            campaign.validate_spsa_outputs("TuneSeeMargin, 141", spec)

    def test_export_reconciles_worker_results_and_fails_on_faults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec = specimen()
            digest = campaign.digest_value(spec)
            info = workload_info(spec, digest)
            row = {"games": info["games"], "LL": 10, "LD": 20, "DD": 40, "DW": 25, "WW": 15, "crashes": 0, "timeloss": 0, "active": False}
            responses = {"api/workload/7/info/": {"info": info}, "api/workload/7/results/": {"results": [row]}, "api/workload/7/summary/": {"summary": {}}, "api/pgns/7/": pgn_archive(info["games"])}
            preflight = {"schema": "volkrix-openbench-preflight-v1", "ready": True, "workload_sha256": digest}
            manifest = campaign.export_result(spec, digest, 7, FakeClient(responses), root / "good", preflight)
            self.assertTrue(manifest["eligible"])
            row["crashes"] = 1
            with self.assertRaisesRegex(campaign.CampaignError, "crashes"):
                campaign.export_result(spec, digest, 7, FakeClient(responses), root / "bad", preflight)

    def test_export_rejects_negative_worker_counters_before_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec = specimen()
            digest = campaign.digest_value(spec)
            info = workload_info(spec, digest)
            rows = [
                {"games": 110, "LL": 10, "LD": 20, "DD": 40, "DW": 25, "WW": -40, "crashes": -1, "timeloss": 0, "active": False},
                {"games": 110, "LL": 0, "LD": 0, "DD": 0, "DW": 0, "WW": 55, "crashes": 1, "timeloss": 0, "active": False},
            ]
            responses = {
                "api/workload/7/info/": {"info": info},
                "api/workload/7/results/": {"results": rows},
                "api/workload/7/summary/": {"summary": {}},
            }
            preflight = {"schema": "volkrix-openbench-preflight-v1", "ready": True, "workload_sha256": digest}
            with self.assertRaisesRegex(campaign.CampaignError, "invalid WW"):
                campaign.export_result(spec, digest, 7, FakeClient(responses), root / "negative", preflight)

            responses["api/workload/7/results/"]["results"] = [{
                "games": info["games"],
                "LL": 10,
                "LD": 20,
                "DD": 40,
                "DW": 25,
                "WW": 15,
                "crashes": False,
                "timeloss": 0,
                "active": False,
            }]
            with self.assertRaisesRegex(campaign.CampaignError, "invalid crashes"):
                campaign.export_result(spec, digest, 7, FakeClient(responses), root / "boolean", preflight)

    def test_pgn_archive_records_legitimate_inflight_surplus_but_rejects_omissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec = specimen()
            digest = campaign.digest_value(spec)
            info = workload_info(spec, digest, penta=[8, 21, 55, 14, 3])
            row = {
                "games": 202,
                "LL": 8,
                "LD": 21,
                "DD": 55,
                "DW": 14,
                "WW": 3,
                "crashes": 0,
                "timeloss": 0,
                "active": False,
            }
            preflight = {
                "schema": "volkrix-openbench-preflight-v1",
                "ready": True,
                "workload_sha256": digest,
            }
            responses = {
                "api/workload/7/info/": {"info": info},
                "api/workload/7/results/": {"results": [row]},
                "api/workload/7/summary/": {"summary": {}},
                "api/pgns/7/": pgn_archive(208),
            }
            result = campaign.export_result(
                spec, digest, 7, FakeClient(responses), root / "surplus", preflight
            )
            self.assertEqual(result["pgn_archive"]["accepted_games"], 202)
            self.assertEqual(result["pgn_archive"]["games"], 208)
            self.assertEqual(result["pgn_archive"]["surplus_games"], 6)
            self.assertEqual(campaign.load_result(root / "surplus"), result)

            responses["api/pgns/7/"] = pgn_archive(200)
            with self.assertRaisesRegex(campaign.CampaignError, "omits accepted"):
                campaign.export_result(
                    spec, digest, 7, FakeClient(responses), root / "missing", preflight
                )

    def test_workload_network_prefix_is_exact_and_fail_closed(self) -> None:
        spec = specimen()
        digest = campaign.digest_value(spec)
        for field in ("dev_network", "base_network"):
            for invalid in ("", NETWORK[:7].upper(), "0" * 8, NETWORK[:8]):
                info = workload_info(spec, digest)
                info[field] = invalid
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(campaign.CampaignError, f"{field[:-8]} network mismatch"):
                        campaign.expected_info(spec, digest, info)

    def test_promotion_requires_passes_matching_chain_and_heldout_book(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            specs = {name: specimen(name, book="f" * 64 if name == "ltc" else BOOK) for name in ("no-change", "stc", "ltc")}
            specs["no-change"]["dev"] = side(BASE)
            specs["no-change"]["base"] = side(BASE)
            paths = {name: export_fixture(root, specs[name], name) for name in specs}
            decision = campaign.promotion(paths["no-change"], paths["stc"], paths["ltc"], root / "decision.json")
            self.assertTrue(decision["promote"])

            same_root = root / "same-book"
            same_specs = {name: specimen(name) for name in ("no-change", "stc", "ltc")}
            same_specs["no-change"]["dev"] = side(BASE)
            same_specs["no-change"]["base"] = side(BASE)
            same_paths = {
                name: export_fixture(same_root, same_specs[name], name)
                for name in same_specs
            }
            rejected = campaign.promotion(
                same_paths["no-change"],
                same_paths["stc"],
                same_paths["ltc"],
                root / "heldout-rejected.json",
            )
            self.assertFalse(rejected["promote"])
            self.assertIn("held-out", rejected["reasons"][0])

            manifest = json.loads((paths["ltc"] / "result.json").read_text())
            manifest["book_sha256"] = BOOK
            (paths["ltc"] / "result.json").write_bytes(campaign.canonical(manifest))
            with self.assertRaisesRegex(campaign.CampaignError, "verified raw evidence"):
                campaign.promotion(paths["no-change"], paths["stc"], paths["ltc"], root / "rejected.json")

    def test_load_result_rejects_manifest_and_artifact_tampering(self) -> None:
        for field, replacement in (
            ("eligible", False),
            ("kind", "ltc"),
            ("dev_commit", "3" * 40),
            ("network_sha256", "0" * 64),
            ("book_sha256", "f" * 64),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                path = export_fixture(pathlib.Path(temporary), specimen(), "stc")
                manifest = json.loads((path / "result.json").read_text())
                manifest[field] = replacement
                (path / "result.json").write_bytes(campaign.canonical(manifest))
                with self.assertRaisesRegex(campaign.CampaignError, "verified raw evidence"):
                    campaign.load_result(path)

        with tempfile.TemporaryDirectory() as temporary:
            path = export_fixture(pathlib.Path(temporary), specimen(), "stc")
            manifest = json.loads((path / "result.json").read_text())
            manifest["artifacts"].pop("info.json")
            (path / "result.json").write_bytes(campaign.canonical(manifest))
            with self.assertRaisesRegex(campaign.CampaignError, "artifact set"):
                campaign.load_result(path)

        with tempfile.TemporaryDirectory() as temporary:
            path = export_fixture(pathlib.Path(temporary), specimen(), "stc")
            (path / "info.json").write_bytes(b"{}\n")
            with self.assertRaisesRegex(campaign.CampaignError, "artifact digest mismatch"):
                campaign.load_result(path)


if __name__ == "__main__":
    unittest.main()
