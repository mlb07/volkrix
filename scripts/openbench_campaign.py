#!/usr/bin/env python3
"""Freeze, preflight, export, and decide Volkrix OpenBench campaigns."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import math
import os
import pathlib
import statistics
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM_LOCK = ROOT / "openbench" / "upstream-lock.json"
HEX40 = set("0123456789abcdef")
KINDS = {"no-change", "stc", "ltc", "spsa"}
PRODUCTION_NETWORK_SHA256 = "c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7"


class CampaignError(RuntimeError):
    pass


def strict_json(path: pathlib.Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CampaignError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{path} must contain a JSON object")
    return value


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def exact_hex(value: Any, length: int, label: str) -> str:
    require(isinstance(value, str) and len(value) == length, f"{label} must be {length} lowercase hex characters")
    require(all(character in HEX40 for character in value), f"{label} must be {length} lowercase hex characters")
    return value


def number(value: Any, label: str, *, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    if positive:
        require(result > 0, f"{label} must be positive")
    return result


def object_field(parent: dict[str, Any], name: str) -> dict[str, Any]:
    value = parent.get(name)
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def only_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    require(not unknown, f"unknown {label} fields: {unknown}")


def validate_side(side: dict[str, Any], label: str) -> None:
    only_keys(side, {"commit", "bench", "network_sha256", "options"}, label)
    exact_hex(side.get("commit"), 40, f"{label}.commit")
    exact_hex(side.get("network_sha256"), 64, f"{label}.network_sha256")
    require(
        side["network_sha256"] == PRODUCTION_NETWORK_SHA256,
        f"{label}.network_sha256 must identify the frozen production network",
    )
    require(isinstance(side.get("bench"), int) and side["bench"] > 0, f"{label}.bench must be a positive integer")
    require(isinstance(side.get("options"), str) and "Threads=" in side["options"] and "Hash=" in side["options"], f"{label}.options must set Threads and Hash")


def validate_spec(spec: dict[str, Any]) -> None:
    only_keys(spec, {"schema", "kind", "campaign", "engine", "source_repo", "dev", "base", "test", "policy", "spsa", "server", "worker"}, "workload")
    require(spec.get("schema") == "volkrix-openbench-workload-v1", "unsupported workload schema")
    require(spec.get("kind") in KINDS, f"kind must be one of {sorted(KINDS)}")
    require(isinstance(spec.get("campaign"), str) and spec["campaign"].strip(), "campaign must be non-empty")
    require(spec.get("engine") == "Volkrix", "engine must be Volkrix")
    require(spec.get("source_repo") == "https://github.com/mlb07/volkrix", "unexpected source repository")
    dev = object_field(spec, "dev")
    validate_side(dev, "dev")
    kind = spec["kind"]
    base = spec.get("base")
    if kind != "spsa":
        require(isinstance(base, dict), "base must be an object")
        validate_side(base, "base")
    else:
        require("base" not in spec, "SPSA must not define a base side")
    test = object_field(spec, "test")
    only_keys(test, {"mode", "book", "time_control", "upload_pgns", "workload_size", "scale", "adjudication", "max_games", "bounds", "confidence"}, "test")
    require(test.get("mode") == ("SPSA" if kind == "spsa" else ("GAMES" if kind == "no-change" else "SPRT")), "test.mode does not match kind")
    book = object_field(test, "book")
    only_keys(book, {"name", "sha256"}, "book")
    require(isinstance(book.get("name"), str) and book["name"].endswith((".epd", ".pgn")), "test.book.name must be an EPD or PGN configuration name")
    exact_hex(book.get("sha256"), 64, "test.book.sha256")
    require(isinstance(test.get("time_control"), str) and test["time_control"].strip(), "test.time_control is required")
    normalize_time_control(test["time_control"])
    require(test.get("upload_pgns") in {"COMPACT", "VERBOSE"}, "PGN upload must be COMPACT or VERBOSE")
    require(isinstance(test.get("workload_size"), int) and test["workload_size"] > 0, "test.workload_size must be positive")
    scale = object_field(test, "scale")
    only_keys(scale, {"method", "nps"}, "scale")
    require(scale.get("method") in {"DEV", "BASE", "BOTH"}, "test.scale.method is invalid")
    require(isinstance(scale.get("nps"), int) and scale["nps"] > 0, "test.scale.nps must be positive")
    adjudication = object_field(test, "adjudication")
    only_keys(adjudication, {"syzygy_wdl", "syzygy_adj", "win", "draw"}, "adjudication")
    require(adjudication.get("syzygy_wdl") in {"OPTIONAL", "DISABLED", "3-MAN", "4-MAN", "5-MAN", "6-MAN", "7-MAN"}, "invalid syzygy_wdl")
    require(adjudication.get("syzygy_adj") in {"OPTIONAL", "DISABLED", "3-MAN", "4-MAN", "5-MAN", "6-MAN", "7-MAN"}, "invalid syzygy_adj")
    for name in ("win", "draw"):
        require(isinstance(adjudication.get(name), str), f"adjudication.{name} must be text")
    if kind == "no-change":
        require("spsa" not in spec, "no-change must not define SPSA settings")
        require(dev == base, "a no-change workload must use identical dev and base inputs")
        require(isinstance(test.get("max_games"), int) and test["max_games"] >= 1000 and test["max_games"] % 2 == 0, "no-change requires at least 1000 paired games")
        policy = object_field(spec, "policy")
        only_keys(policy, {"max_abs_elo", "require_ci_contains_zero"}, "policy")
        number(policy.get("max_abs_elo"), "policy.max_abs_elo", positive=True)
        require(policy.get("require_ci_contains_zero") is True, "no-change must require a confidence interval containing zero")
    elif kind in {"stc", "ltc"}:
        require("policy" not in spec and "spsa" not in spec, "SPRT workloads must not define no-change/SPSA settings")
        bounds = test.get("bounds")
        confidence = test.get("confidence")
        require(isinstance(bounds, list) and len(bounds) == 2 and number(bounds[0], "bounds[0]") < number(bounds[1], "bounds[1]"), "SPRT bounds must be increasing")
        require(isinstance(confidence, list) and len(confidence) == 2 and all(0 < number(x, "confidence") < 1 for x in confidence), "SPRT confidence values must be in (0,1)")
    else:
        require("policy" not in spec, "SPSA must not define no-change policy")
        spsa = object_field(spec, "spsa")
        only_keys(spsa, {"reporting", "distribution", "alpha", "gamma", "a_ratio", "iterations", "pairs_per", "inputs"}, "SPSA")
        require(spsa.get("reporting") in {"BATCHED", "BULK"}, "invalid SPSA reporting mode")
        require(spsa.get("distribution") in {"SINGLE", "MULTIPLE"}, "invalid SPSA distribution mode")
        for name in ("alpha", "gamma", "a_ratio"):
            number(spsa.get(name), f"spsa.{name}", positive=True)
        for name in ("iterations", "pairs_per"):
            require(isinstance(spsa.get(name), int) and spsa[name] > 0, f"spsa.{name} must be positive")
        inputs = spsa.get("inputs")
        require(isinstance(inputs, list) and inputs, "spsa.inputs must be non-empty")
        names: set[str] = set()
        for index, item in enumerate(inputs):
            require(isinstance(item, dict), f"spsa.inputs[{index}] must be an object")
            only_keys(item, {"name", "type", "start", "min", "max", "c_end", "r_end"}, f"SPSA input {index}")
            require(isinstance(item.get("name"), str) and item["name"] not in names, "SPSA names must be unique")
            names.add(item["name"])
            require(item.get("type") in {"int", "float"}, "SPSA type must be int or float")
            values = [number(item.get(key), f"SPSA {item.get('name')} {key}") for key in ("start", "min", "max", "c_end", "r_end")]
            require(values[1] <= values[0] <= values[2] and values[3] > 0 and values[4] > 0, "invalid SPSA parameter bounds/rates")
        require(test["workload_size"] == spsa["pairs_per"], "SPSA workload_size must equal pairs_per")
    server = object_field(spec, "server")
    only_keys(server, {"openbench_commit", "fastchess_commit", "client_version", "network_name", "reference_nps"}, "server")
    lock = strict_json(UPSTREAM_LOCK)
    require(server.get("openbench_commit") == lock["openbench"]["commit"], "server OpenBench commit differs from lock")
    require(server.get("fastchess_commit") == lock["fastchess"]["commit"], "server FastChess commit differs from lock")
    require(server.get("client_version") == lock["openbench"]["client_version"], "server client version differs from lock")
    require(server.get("network_name") == "nn-c288c895ea92.nnue", "unexpected production network name")
    require(server.get("reference_nps") == test["scale"]["nps"], "reference NPS and scale NPS must match")
    worker = object_field(spec, "worker")
    only_keys(worker, {"threads", "sockets", "focus", "min_bench_samples", "max_bench_cv", "max_nps_deviation"}, "worker")
    require(isinstance(worker.get("threads"), int) and worker["threads"] > 0, "worker.threads must be positive")
    require(isinstance(worker.get("sockets"), int) and worker["sockets"] > 0, "worker.sockets must be positive")
    require(worker.get("focus") == "Volkrix", "worker focus must be Volkrix")
    require(isinstance(worker.get("min_bench_samples"), int) and worker["min_bench_samples"] >= 5, "at least five bench samples are required")
    number(worker.get("max_bench_cv"), "worker.max_bench_cv", positive=True)
    number(worker.get("max_nps_deviation"), "worker.max_nps_deviation", positive=True)


def normalize_time_control(value: str) -> str:
    if "=" in value:
        mode, amount = value.upper().split("=", 1)
        aliases = {"N": "N", "NODES": "N", "D": "D", "DEPTH": "D", "MT": "MT", "MOVETIME": "MT"}
        require(
            mode in aliases and amount.isdigit() and int(amount) > 0,
            f"invalid time control: {value}",
        )
        return f"{aliases[mode]}={int(amount)}"
    try:
        prefix, clock = value.split("/", 1) if "/" in value else (None, value)
        base, increment = clock.split("+", 1) if "+" in clock else (clock, "0")
        require(prefix is None or (prefix.isdigit() and int(prefix) > 0), f"invalid time control: {value}")
        base_value, increment_value = float(base), float(increment)
        require(
            math.isfinite(base_value)
            and math.isfinite(increment_value)
            and base_value > 0
            and increment_value >= 0,
            f"invalid time control: {value}",
        )
        result = f"{base_value:.1f}+{increment_value:.2f}"
        return f"{int(prefix)}/{result}" if prefix is not None else result
    except (ValueError, TypeError) as error:
        raise CampaignError(f"invalid time control: {value}") from error


def freeze(source: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    require(not output.exists() and not output.is_symlink(), f"output already exists: {output}")
    spec = strict_json(source)
    validate_spec(spec)
    lock = {"schema": "volkrix-openbench-workload-lock-v1", "sha256": digest_value(spec), "spec": spec}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(lock))
    return lock


def load_lock(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    lock = strict_json(path)
    require(lock.get("schema") == "volkrix-openbench-workload-lock-v1", "unsupported lock schema")
    spec = lock.get("spec")
    require(isinstance(spec, dict), "lock spec is missing")
    validate_spec(spec)
    observed = digest_value(spec)
    exact_hex(lock.get("sha256"), 64, "lock sha256")
    require(lock["sha256"] == observed, "workload lock digest mismatch")
    return spec, observed


def spsa_text(spec: dict[str, Any]) -> str:
    return "\n".join(", ".join(str(item[key]) for key in ("name", "type", "start", "min", "max", "c_end", "r_end")) for item in spec["spsa"]["inputs"])


def form_payload(spec: dict[str, Any], lock_digest: str) -> dict[str, str]:
    test, dev = spec["test"], spec["dev"]
    adj, scale = test["adjudication"], test["scale"]
    payload = {
        "dev_engine": "Volkrix", "dev_repo": spec["source_repo"], "dev_branch": dev["commit"],
        "dev_bench": str(dev["bench"]), "dev_network": dev["network_sha256"][:8].upper(), "dev_options": dev["options"],
        "dev_time_control": test["time_control"], "book_name": test["book"]["name"], "upload_pgns": test["upload_pgns"],
        "priority": "0", "throughput": "1", "syzygy_wdl": adj["syzygy_wdl"], "syzygy_adj": adj["syzygy_adj"],
        "win_adj": adj["win"], "draw_adj": adj["draw"], "scale_method": scale["method"], "scale_nps": str(scale["nps"]),
        "info": f"volkrix-campaign-sha256={lock_digest}",
    }
    if spec["kind"] == "spsa":
        cfg = spec["spsa"]
        payload.update({"spsa_reporting_type": cfg["reporting"], "spsa_distribution_type": cfg["distribution"],
                        "spsa_alpha": str(cfg["alpha"]), "spsa_gamma": str(cfg["gamma"]), "spsa_A_ratio": str(cfg["a_ratio"]),
                        "spsa_iterations": str(cfg["iterations"]), "spsa_pairs_per": str(cfg["pairs_per"]), "spsa_inputs": spsa_text(spec)})
    else:
        base = spec["base"]
        payload.update({"base_engine": "Volkrix", "base_repo": spec["source_repo"], "base_branch": base["commit"],
                        "base_bench": str(base["bench"]), "base_network": base["network_sha256"][:8].upper(), "base_options": base["options"],
                        "base_time_control": test["time_control"], "workload_size": str(test["workload_size"]), "test_mode": test["mode"],
                        "test_bounds": "[%s, %s]" % tuple(test.get("bounds", [0, 0])),
                        "test_confidence": "[%s, %s]" % tuple(test.get("confidence", [0.05, 0.05])),
                        "test_max_games": str(test.get("max_games", 0))})
    return payload


class ApiClient:
    def __init__(self, base_url: str, username: str, password: str, timeout: float = 15.0):
        parsed = urllib.parse.urlparse(base_url)
        require(parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}), "server URL must use HTTPS (HTTP is allowed only for localhost)")
        require(not parsed.query and not parsed.fragment and parsed.hostname, "server URL is invalid")
        self.base_url = base_url.rstrip("/") + "/"
        self.origin = (parsed.scheme, parsed.hostname, parsed.port)
        self.credentials = urllib.parse.urlencode({"username": username, "password": password}).encode()
        self.timeout = timeout
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
                return None
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, endpoint: str, *, binary: bool = False) -> Any:
        url = urllib.parse.urljoin(self.base_url, endpoint.lstrip("/"))
        request = urllib.request.Request(url, data=self.credentials, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                final = urllib.parse.urlparse(response.geturl())
                require((final.scheme, final.hostname, final.port) == self.origin, "server changed origin")
                body = response.read()
                if binary:
                    require(response.headers.get_content_type() != "application/json", f"server returned an error document for {endpoint}")
                    return body
                content_type = response.headers.get_content_type()
                if content_type == "text/plain":
                    return body.decode("utf-8")
                value = json.loads(body)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise CampaignError(f"request failed for {endpoint}: {error}") from error
        require(isinstance(value, dict), f"server returned a non-object JSON response for {endpoint}")
        require(not value.get("error"), f"server rejected {endpoint}: {value.get('error')}")
        return value


def validate_worker_record(path: pathlib.Path, spec: dict[str, Any]) -> dict[str, Any]:
    record = strict_json(path)
    require(record.get("schema") == "volkrix-openbench-worker-record-v1", "unsupported worker record")
    audit = record.get("deployment_audit")
    require(isinstance(audit, dict), "deployment audit is absent")
    only_keys(
        audit,
        {"schema", "ready", "blockers", "warnings", "facts", "external_remaining"},
        "deployment audit",
    )
    require(
        audit.get("schema") == "volkrix-openbench-deployment-audit-v1",
        "unsupported deployment audit schema",
    )
    require(audit.get("ready") is True, "deployment audit is not ready")
    require(audit.get("blockers") == [], "deployment audit contains blockers")
    for name in ("warnings", "external_remaining"):
        values = audit.get(name)
        require(
            isinstance(values, list) and all(isinstance(value, str) for value in values),
            f"deployment audit {name} is malformed",
        )
    require(record.get("deployment_audit_canonical_sha256") == digest_value(audit), "embedded deployment audit digest mismatch")
    facts = audit.get("facts")
    require(isinstance(facts, dict), "deployment audit facts are missing")
    only_keys(
        facts,
        {
            "platform", "machine", "cores", "memory_bytes", "openbench_lock", "tools",
            "python_requests", "validated_server_python", "openbench_head", "fastchess_head",
            "network", "recommended_worker_command",
        },
        "deployment audit facts",
    )
    lock = strict_json(UPSTREAM_LOCK)
    require(facts.get("openbench_lock") == lock, "worker embedded upstream lock differs from repository lock")
    require(facts.get("platform") in {"Linux", "Windows", "Darwin"}, "worker platform fact is invalid")
    require(isinstance(facts.get("machine"), str) and facts["machine"], "worker machine fact is missing")
    require(isinstance(facts.get("memory_bytes"), int) and not isinstance(facts["memory_bytes"], bool) and facts["memory_bytes"] > 0, "worker memory fact is invalid")
    cores = facts.get("cores")
    require(isinstance(cores, dict) and set(cores) == {"logical", "physical", "performance", "efficiency"}, "worker core facts are malformed")
    for key, value in cores.items():
        require(value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0), f"worker {key} core fact is invalid")
    require(isinstance(cores["logical"], int) and cores["logical"] > 0, "worker logical core fact is invalid")
    tools = facts.get("tools")
    require(isinstance(tools, dict) and set(tools) == {"git", "make", "cargo", "clang++", "g++", "docker"}, "worker tool facts are malformed")
    require(all(isinstance(tools[name], str) and tools[name] for name in ("git", "make", "cargo")), "worker required tool facts are missing")
    require(any(isinstance(tools[name], str) and tools[name] for name in ("clang++", "g++")), "worker compiler fact is missing")
    require(isinstance(facts.get("python_requests"), str) and facts["python_requests"], "worker requests fact is missing")
    require(facts.get("validated_server_python") is None or isinstance(facts["validated_server_python"], str), "worker server Python fact is malformed")
    require(isinstance(facts.get("recommended_worker_command"), str) and facts["recommended_worker_command"], "worker command fact is missing")
    require(facts.get("openbench_head") == lock["openbench"]["commit"], "worker OpenBench checkout is not pinned")
    require(facts.get("fastchess_head") == lock["fastchess"]["commit"], "worker FastChess checkout is not pinned")
    network = facts.get("network")
    require(
        isinstance(network, dict)
        and set(network) == {"path", "sha256", "size"}
        and isinstance(network.get("path"), str)
        and bool(network["path"])
        and network.get("sha256") == spec["dev"]["network_sha256"]
        and isinstance(network.get("size"), int)
        and network["size"] > 0,
        "worker production network mismatch",
    )
    run = record.get("worker_run")
    require(isinstance(run, dict), "worker_run is missing")
    policy = spec["worker"]
    for key in ("threads", "sockets", "focus"):
        require(run.get(key) == policy[key], f"worker {key} differs from workload lock")
    samples = record.get("bench_nps")
    require(isinstance(samples, list) and len(samples) >= policy["min_bench_samples"], "insufficient worker bench samples")
    samples = [number(value, "bench sample", positive=True) for value in samples]
    median = statistics.median(samples)
    cv = statistics.pstdev(samples) / statistics.fmean(samples) if len(samples) > 1 else 0.0
    deviation = abs(median - spec["server"]["reference_nps"]) / spec["server"]["reference_nps"]
    require(cv <= policy["max_bench_cv"], f"worker bench coefficient of variation {cv:.4f} exceeds policy")
    require(deviation <= policy["max_nps_deviation"], f"worker median NPS deviation {deviation:.4f} exceeds policy")
    return {"sha256": sha256_file(path), "samples": len(samples), "median_nps": median, "coefficient_of_variation": cv, "reference_deviation": deviation}


def create_worker_record(spec: dict[str, Any], audit_path: pathlib.Path, samples_text: str, output: pathlib.Path) -> dict[str, Any]:
    require(not output.exists() and not output.is_symlink(), f"output already exists: {output}")
    audit = strict_json(audit_path)
    try:
        samples = [float(value.strip()) for value in samples_text.split(",") if value.strip()]
    except ValueError as error:
        raise CampaignError("--bench-nps must be a comma-separated list of numbers") from error
    record = {"schema": "volkrix-openbench-worker-record-v1", "deployment_audit_sha256": sha256_file(audit_path),
              "deployment_audit_canonical_sha256": digest_value(audit),
              "deployment_audit": audit, "worker_run": {"threads": spec["worker"]["threads"], "sockets": spec["worker"]["sockets"], "focus": spec["worker"]["focus"]},
              "bench_nps": samples}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(record))
    try:
        validate_worker_record(output, spec)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return record


def preflight(spec: dict[str, Any], lock_digest: str, client: ApiClient, worker_record: pathlib.Path) -> dict[str, Any]:
    upstream = strict_json(UPSTREAM_LOCK)
    client_ref = client.request("clientVersionRef/")
    runner_ref = client.request("clientMatchRunnerVersionRef/")
    config = client.request("api/config/")
    engine = client.request("api/config/Volkrix/")
    networks = client.request("api/networks/Volkrix/")
    require(client_ref == {"client_version": upstream["openbench"]["client_version"], "client_repo_url": upstream["openbench"]["repository"], "client_repo_ref": upstream["openbench"]["commit"]}, "server client pin mismatch")
    require(runner_ref == {"fastchess_min_version": upstream["fastchess"]["minimum_version"], "fastchess_repo_url": upstream["fastchess"]["repository"], "fastchess_repo_ref": upstream["fastchess"]["commit"]}, "server FastChess pin mismatch")
    require(config.get("engines") == ["Volkrix"], "server must expose exactly the Volkrix engine")
    book = config.get("books", {}).get(spec["test"]["book"]["name"])
    require(isinstance(book, dict) and str(book.get("sha", "")).lower() == spec["test"]["book"]["sha256"], "server opening book digest mismatch")
    require(engine.get("source") == spec["source_repo"] and engine.get("nps") == spec["server"]["reference_nps"], "server engine source or NPS mismatch")
    require(engine.get("build") == {"path": "openbench", "compilers": ["cargo>=1.85.0"], "cpuflags": [], "systems": ["Linux", "Windows", "Darwin"]}, "server engine build contract mismatch")
    default = networks.get("default")
    prefix = default.get("sha256") if isinstance(default, dict) else None
    require(
        isinstance(default, dict)
        and default.get("name") == spec["server"]["network_name"]
        and prefix == spec["dev"]["network_sha256"][:8].upper(),
        "server default network mismatch",
    )
    worker = validate_worker_record(worker_record, spec)
    return {"schema": "volkrix-openbench-preflight-v1", "ready": True, "workload_sha256": lock_digest, "worker": worker,
            "server": {"client": client_ref, "fastchess": runner_ref, "book": book, "engine_config_sha256": digest_value(engine), "default_network": default}}


def expected_info(spec: dict[str, Any], lock_digest: str, info: dict[str, Any]) -> None:
    test, dev = spec["test"], spec["dev"]
    observed_dev = info.get("dev")
    require(isinstance(observed_dev, dict), "dev source metadata is malformed")
    if spec["kind"] == "spsa":
        require(info.get("info") == f"volkrix-campaign-sha256={lock_digest}", "SPSA workload is not bound to this lock digest")
    require(info.get("book_name") == test["book"]["name"] and info.get("test_mode") == test["mode"], "workload book or mode mismatch")
    require(info.get("dev_engine") == "Volkrix", "dev engine mismatch")
    require(info.get("dev_repo") == spec["source_repo"] and observed_dev.get("sha") == dev["commit"], "dev source mismatch")
    require(observed_dev.get("bench") == dev["bench"] and info.get("dev_options") == dev["options"], "dev bench/options mismatch")
    require(
        info.get("dev_network") == dev["network_sha256"][:8].upper(),
        "dev network mismatch",
    )
    require(str(info.get("dev_time_control")) == normalize_time_control(test["time_control"]), "dev time control mismatch")
    require(info.get("scale_method") == test["scale"]["method"] and info.get("scale_nps") == test["scale"]["nps"], "workload scaling mismatch")
    require(info.get("upload_pgns") == test["upload_pgns"] and info.get("workload_size") == test["workload_size"], "PGN/workload size mismatch")
    require(info.get("priority") == 0 and info.get("throughput") == 1, "priority/throughput mismatch")
    adj = test["adjudication"]
    require((info.get("syzygy_wdl"), info.get("syzygy_adj"), info.get("win_adj"), info.get("draw_adj")) == (adj["syzygy_wdl"], adj["syzygy_adj"], adj["win"], adj["draw"]), "adjudication mismatch")
    if spec["kind"] != "spsa":
        base = spec["base"]
        observed_base = info.get("base")
        require(isinstance(observed_base, dict), "base source metadata is malformed")
        require(info.get("base_engine") == "Volkrix", "base engine mismatch")
        require(info.get("base_repo") == spec["source_repo"] and observed_base.get("sha") == base["commit"], "base source mismatch")
        require(observed_base.get("bench") == base["bench"] and info.get("base_options") == base["options"], "base bench/options mismatch")
        require(
            info.get("base_network") == base["network_sha256"][:8].upper(),
            "base network mismatch",
        )
        require(str(info.get("base_time_control")) == normalize_time_control(test["time_control"]), "base time control mismatch")
        if spec["kind"] == "no-change":
            require(info.get("max_games") == test["max_games"], "fixed-game limit mismatch")
            require(isinstance(info.get("games"), int) and info["games"] >= test["max_games"], "fixed-game workload ended too early")
        else:
            require([info.get("elolower"), info.get("eloupper")] == [float(x) for x in test["bounds"]], "SPRT bounds mismatch")
            require([info.get("beta"), info.get("alpha")] == [float(x) for x in test["confidence"]], "SPRT confidence mismatch")
    else:
        expected_games = 2 * spec["spsa"]["pairs_per"] * spec["spsa"]["iterations"]
        require(isinstance(info.get("games"), int) and info["games"] >= expected_games, "SPSA workload ended too early")


def logistic(score: float) -> float:
    score = min(max(score, 0.001), 0.999)
    return -400 * math.log10(1.0 / score - 1.0)


def statistics_from_penta(penta: list[int]) -> dict[str, Any]:
    pairs = sum(penta)
    require(pairs >= 2, "at least two completed pairs are required")
    samples = [index / 4 for index, count in enumerate(penta) for _ in range(count)]
    mean = statistics.fmean(samples)
    error = 1.96 * statistics.stdev(samples) / math.sqrt(pairs)
    return {"pairs": pairs, "games": pairs * 2, "elo": logistic(mean), "elo95_low": logistic(mean - error), "elo95_high": logistic(mean + error)}


def validate_spsa_input_export(text: str, spec: dict[str, Any]) -> None:
    lines = [line for line in text.splitlines() if line.strip()]
    expected = spec["spsa"]["inputs"]
    require(len(lines) == len(expected), "server SPSA input count differs from lock")
    for line, item in zip(lines, expected):
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 7 and fields[:2] == [item["name"], item["type"]], "server SPSA name/type differs from lock")
        try:
            observed = [float(value) for value in fields[2:]]
        except ValueError as error:
            raise CampaignError("server returned a malformed SPSA input") from error
        wanted = [float(item[key]) for key in ("start", "min", "max", "c_end", "r_end")]
        require(observed == wanted, f"server SPSA values differ for {item['name']}")


def validate_spsa_outputs(text: str, spec: dict[str, Any]) -> None:
    expected = {item["name"]: item for item in spec["spsa"]["inputs"]}
    observed: dict[str, float] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 2 and fields[0] in expected and fields[0] not in observed, "server returned malformed/duplicate SPSA output")
        try:
            observed[fields[0]] = float(fields[1])
        except ValueError as error:
            raise CampaignError("server returned non-numeric SPSA output") from error
    require(set(observed) == set(expected), "server SPSA output parameters differ from lock")
    for name, value in observed.items():
        require(float(expected[name]["min"]) <= value <= float(expected[name]["max"]), f"SPSA output {name} is out of bounds")


def validate_pgn_archive(data: bytes) -> dict[str, int]:
    names: set[str] = set()
    compressed_bytes = 0
    games = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            members = archive.getmembers()
            require(bool(members), "PGN archive is empty")
            for member in members:
                require(member.isfile(), "PGN archive contains a non-file member")
                require(pathlib.PurePosixPath(member.name).name == member.name and member.name.endswith(".pgn.bz2"), "PGN archive contains an unsafe/unexpected member")
                require(member.name not in names and member.size > 0, "PGN archive contains a duplicate/empty member")
                names.add(member.name)
                source = archive.extractfile(member)
                require(source is not None, "PGN archive member cannot be read")
                contents = source.read()
                compressed_bytes += len(contents)
                require(contents.startswith(b"BZh"), "PGN archive member is not bzip2 data")
                sample = bz2.decompress(contents)
                result_count = sample.count(b"[Result \"")
                require(result_count > 0 and sample.count(b"[Event \"") == result_count, "PGN archive member does not contain complete PGN headers")
                games += result_count
    except (tarfile.TarError, OSError, EOFError) as error:
        raise CampaignError(f"invalid PGN archive: {error}") from error
    return {"members": len(names), "compressed_bytes": compressed_bytes, "games": games}


def reconcile_pgn_archive(archive: dict[str, int], accepted_games: int) -> dict[str, int]:
    """Bind archived play to accepted counters without misclassifying in-flight surplus.

    The pinned client uploads its completed PGN after reporting. Other workers may already have
    played games when the server closes a workload, so OpenBench can archive more games than it
    accepted into statistics. The public result API does not expose enough assignment geometry to
    derive a sound upper bound for that surplus. Missing accepted games remain a hard failure.
    """
    require(archive["games"] >= accepted_games, "PGN archive omits accepted workload games")
    return {
        **archive,
        "accepted_games": accepted_games,
        "surplus_games": archive["games"] - accepted_games,
    }


def validate_completed_result(
    spec: dict[str, Any],
    lock_digest: str,
    workload_id: int,
    info_response: dict[str, Any],
    results_response: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    info = info_response.get("info")
    results = results_response.get("results")
    require(isinstance(info, dict) and isinstance(results, list), "malformed workload API response")
    expected_info(spec, lock_digest, info)
    require(info.get("id") == workload_id and info.get("finished") is True, "workload is not finished")
    require(info.get("error") is False and info.get("deleted") is False, "workload is errored or deleted")
    penta = info.get("penta")
    require(
        isinstance(penta, list)
        and len(penta) == 5
        and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in penta),
        "invalid aggregate pentanomial",
    )
    require(sum(penta) * 2 == info.get("games"), "aggregate games and pentanomial disagree")
    counter_keys = ("games", "LL", "LD", "DD", "DW", "WW", "crashes", "timeloss")
    for index, row in enumerate(results):
        require(isinstance(row, dict), f"per-worker result {index} is not an object")
        for key in counter_keys:
            value = row.get(key)
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"per-worker result {index} has invalid {key}",
            )
        require(
            row["games"] == 2 * sum(row[key] for key in ("LL", "LD", "DD", "DW", "WW")),
            f"per-worker result {index} games and pentanomial disagree",
        )
        require(row.get("active") is False, "a worker still reports this workload active")
    sums = {key: sum(row[key] for row in results) for key in counter_keys}
    require(sums["games"] == info["games"] and [sums[key] for key in ("LL", "LD", "DD", "DW", "WW")] == penta, "per-worker and aggregate results disagree")
    require(sums["crashes"] == 0 and sums["timeloss"] == 0, "workload contains crashes or time losses")
    stats = statistics_from_penta(penta)
    if spec["kind"] == "no-change":
        policy = spec["policy"]
        eligible = abs(stats["elo"]) <= policy["max_abs_elo"] and stats["elo95_low"] <= 0 <= stats["elo95_high"]
    elif spec["kind"] in {"stc", "ltc"}:
        eligible = info.get("passed") is True and info.get("failed") is False
    else:
        eligible = True
    return stats, eligible


def result_artifact_names(kind: str) -> set[str]:
    names = {
        "workload-lock.json",
        "preflight.json",
        "info.json",
        "results.json",
        "summary.json",
        "games.pgn.tar",
    }
    if kind == "spsa":
        names.update({"spsa-inputs.txt", "spsa-outputs.txt", "spsa-digest.txt"})
    return names


def export_result(spec: dict[str, Any], lock_digest: str, workload_id: int, client: ApiClient, output: pathlib.Path, preflight_report: dict[str, Any]) -> dict[str, Any]:
    require(not output.exists() and not output.is_symlink(), f"output already exists: {output}")
    require(digest_value(spec) == lock_digest, "workload lock digest does not match specification")
    info_response = client.request(f"api/workload/{workload_id}/info/")
    results_response = client.request(f"api/workload/{workload_id}/results/")
    summary_response = client.request(f"api/workload/{workload_id}/summary/")
    require(isinstance(info_response, dict) and isinstance(results_response, dict), "malformed workload API response")
    require(isinstance(summary_response, dict), "malformed workload summary response")
    stats, eligible = validate_completed_result(
        spec, lock_digest, workload_id, info_response, results_response
    )
    require(preflight_report.get("ready") is True and preflight_report.get("workload_sha256") == lock_digest, "fresh preflight is not bound to workload lock")
    workload_lock = {
        "schema": "volkrix-openbench-workload-lock-v1",
        "sha256": lock_digest,
        "spec": spec,
    }
    artifacts: dict[str, bytes] = {
        "workload-lock.json": canonical(workload_lock),
        "preflight.json": canonical(preflight_report),
        "info.json": canonical(info_response),
        "results.json": canonical(results_response),
        "summary.json": canonical(summary_response),
    }
    if spec["kind"] == "spsa":
        for query in ("inputs", "outputs", "digest"):
            artifacts[f"spsa-{query}.txt"] = client.request(f"api/spsa/{workload_id}/{query}/").encode()
        validate_spsa_input_export(artifacts["spsa-inputs.txt"].decode(), spec)
        validate_spsa_outputs(artifacts["spsa-outputs.txt"].decode(), spec)
        require(bool(artifacts["spsa-digest.txt"].strip()), "server SPSA digest is empty")
    artifacts["games.pgn.tar"] = client.request(f"api/pgns/{workload_id}/", binary=True)
    require(artifacts["games.pgn.tar"], "empty PGN archive")
    pgn_archive = reconcile_pgn_archive(
        validate_pgn_archive(artifacts["games.pgn.tar"]),
        info_response["info"]["games"],
    )
    output.mkdir(parents=True)
    for name, data in artifacts.items():
        (output / name).write_bytes(data)
    manifest = {"schema": "volkrix-openbench-result-v1", "workload_id": workload_id, "workload_sha256": lock_digest,
                "kind": spec["kind"], "eligible": eligible, "stats": stats, "dev_commit": spec["dev"]["commit"],
                "base_commit": spec.get("base", {}).get("commit"), "network_sha256": spec["dev"]["network_sha256"],
                "book_sha256": spec["test"]["book"]["sha256"], "pgn_archive": pgn_archive,
                "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}}
    (output / "result.json").write_bytes(canonical(manifest))
    return manifest


def load_result(directory: pathlib.Path) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "result path must be a real directory")
    manifest = strict_json(directory / "result.json")
    require(manifest.get("schema") == "volkrix-openbench-result-v1", "unsupported result manifest")
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "result artifact manifest is missing")
    lock_path = directory / "workload-lock.json"
    require(lock_path.is_file() and not lock_path.is_symlink(), "workload lock must be a real file")
    lock_spec, lock_digest = load_lock(lock_path)
    expected_names = result_artifact_names(lock_spec["kind"])
    require(set(artifacts) == expected_names, "result artifact set is incomplete or unexpected")
    for name, digest in artifacts.items():
        require(isinstance(name, str) and pathlib.PurePosixPath(name).name == name, "unsafe result artifact name")
        require((directory / name).is_file() and not (directory / name).is_symlink(), f"result artifact must be a real file: {name}")
        exact_hex(digest, 64, f"artifact {name}")
        require(sha256_file(directory / name) == digest, f"result artifact digest mismatch: {name}")
    preflight_report = strict_json(directory / "preflight.json")
    require(
        preflight_report.get("schema") == "volkrix-openbench-preflight-v1"
        and preflight_report.get("ready") is True
        and preflight_report.get("workload_sha256") == lock_digest,
        "stored preflight is not ready or bound to the workload lock",
    )
    info_response = strict_json(directory / "info.json")
    results_response = strict_json(directory / "results.json")
    strict_json(directory / "summary.json")
    workload_id = manifest.get("workload_id")
    require(isinstance(workload_id, int) and not isinstance(workload_id, bool) and workload_id > 0, "invalid result workload ID")
    stats, eligible = validate_completed_result(
        lock_spec, lock_digest, workload_id, info_response, results_response
    )
    if lock_spec["kind"] == "spsa":
        inputs = (directory / "spsa-inputs.txt").read_text(encoding="utf-8")
        outputs = (directory / "spsa-outputs.txt").read_text(encoding="utf-8")
        digest = (directory / "spsa-digest.txt").read_text(encoding="utf-8")
        validate_spsa_input_export(inputs, lock_spec)
        validate_spsa_outputs(outputs, lock_spec)
        require(bool(digest.strip()), "server SPSA digest is empty")
    pgn_archive = reconcile_pgn_archive(
        validate_pgn_archive((directory / "games.pgn.tar").read_bytes()),
        info_response["info"]["games"],
    )
    expected_manifest = {
        "schema": "volkrix-openbench-result-v1",
        "workload_id": workload_id,
        "workload_sha256": lock_digest,
        "kind": lock_spec["kind"],
        "eligible": eligible,
        "stats": stats,
        "dev_commit": lock_spec["dev"]["commit"],
        "base_commit": lock_spec.get("base", {}).get("commit"),
        "network_sha256": lock_spec["dev"]["network_sha256"],
        "book_sha256": lock_spec["test"]["book"]["sha256"],
        "pgn_archive": pgn_archive,
        "artifacts": {name: sha256_file(directory / name) for name in artifacts},
    }
    require(manifest == expected_manifest, "result manifest differs from verified raw evidence")
    return manifest


def promotion(no_change: pathlib.Path, stc: pathlib.Path, ltc: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    require(not output.exists() and not output.is_symlink(), f"output already exists: {output}")
    inputs = {"no-change": load_result(no_change), "stc": load_result(stc), "ltc": load_result(ltc)}
    reasons: list[str] = []
    for kind, result in inputs.items():
        if result.get("kind") != kind:
            reasons.append(f"{kind} input has kind {result.get('kind')}")
        if result.get("eligible") is not True:
            reasons.append(f"{kind} did not pass its frozen policy")
    if inputs["stc"].get("dev_commit") != inputs["ltc"].get("dev_commit"):
        reasons.append("STC and LTC candidate commits differ")
    if inputs["stc"].get("base_commit") != inputs["ltc"].get("base_commit"):
        reasons.append("STC and LTC base commits differ")
    if inputs["no-change"].get("dev_commit") != inputs["stc"].get("base_commit"):
        reasons.append("no-change control does not match the STC/LTC base commit")
    if inputs["stc"].get("network_sha256") != inputs["ltc"].get("network_sha256"):
        reasons.append("STC and LTC networks differ")
    if inputs["no-change"].get("network_sha256") != inputs["stc"].get("network_sha256"):
        reasons.append("no-change and promotion tests use different networks")
    if inputs["stc"].get("book_sha256") == inputs["ltc"].get("book_sha256"):
        reasons.append("LTC did not use a held-out opening book")
    decision = {"schema": "volkrix-openbench-promotion-v1", "promote": not reasons, "candidate_commit": inputs["ltc"].get("dev_commit"),
                "base_commit": inputs["ltc"].get("base_commit"), "reasons": reasons,
                "inputs": {kind: {"result_sha256": sha256_file(path / "result.json"), "workload_sha256": inputs[kind].get("workload_sha256"), "workload_id": inputs[kind].get("workload_id")} for kind, path in (("no-change", no_change), ("stc", stc), ("ltc", ltc))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(decision))
    return decision


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--spec", type=pathlib.Path, required=True)
    freeze_parser.add_argument("--output", type=pathlib.Path, required=True)
    form_parser = commands.add_parser("form")
    form_parser.add_argument("--lock", type=pathlib.Path, required=True)
    form_parser.add_argument("--output", type=pathlib.Path, required=True)
    worker_parser = commands.add_parser("worker-record")
    worker_parser.add_argument("--lock", type=pathlib.Path, required=True)
    worker_parser.add_argument("--deployment-audit", type=pathlib.Path, required=True)
    worker_parser.add_argument("--bench-nps", required=True)
    worker_parser.add_argument("--output", type=pathlib.Path, required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--lock", type=pathlib.Path, required=True)
    preflight_parser.add_argument("--server", required=True)
    preflight_parser.add_argument("--worker-record", type=pathlib.Path, required=True)
    preflight_parser.add_argument("--output", type=pathlib.Path, required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--lock", type=pathlib.Path, required=True)
    export_parser.add_argument("--server", required=True)
    export_parser.add_argument("--worker-record", type=pathlib.Path, required=True)
    export_parser.add_argument("--workload-id", type=int, required=True)
    export_parser.add_argument("--output", type=pathlib.Path, required=True)
    promote_parser = commands.add_parser("promote")
    promote_parser.add_argument("--no-change", type=pathlib.Path, required=True)
    promote_parser.add_argument("--stc", type=pathlib.Path, required=True)
    promote_parser.add_argument("--ltc", type=pathlib.Path, required=True)
    promote_parser.add_argument("--output", type=pathlib.Path, required=True)
    return result


def credentials() -> tuple[str, str]:
    username, password = os.environ.get("OPENBENCH_USERNAME"), os.environ.get("OPENBENCH_PASSWORD")
    require(bool(username and password), "OPENBENCH_USERNAME and OPENBENCH_PASSWORD are required")
    return str(username), str(password)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "freeze":
            report = freeze(args.spec, args.output)
        elif args.command == "form":
            spec, digest = load_lock(args.lock)
            report = {"endpoint": "/tune/new/" if spec["kind"] == "spsa" else "/test/new/", "workload_sha256": digest, "fields": form_payload(spec, digest)}
            require(not args.output.exists() and not args.output.is_symlink(), f"output already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical(report))
        elif args.command == "worker-record":
            spec, _ = load_lock(args.lock)
            report = create_worker_record(spec, args.deployment_audit, args.bench_nps, args.output)
        elif args.command in {"preflight", "export"}:
            spec, digest = load_lock(args.lock)
            username, password = credentials()
            client = ApiClient(args.server, username, password)
            if args.command == "preflight":
                report = preflight(spec, digest, client, args.worker_record)
                require(not args.output.exists() and not args.output.is_symlink(), f"output already exists: {args.output}")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(canonical(report))
            else:
                require(args.workload_id > 0, "workload ID must be positive")
                fresh_preflight = preflight(spec, digest, client, args.worker_record)
                report = export_result(spec, digest, args.workload_id, client, args.output, fresh_preflight)
        else:
            report = promotion(args.no_change, args.stc, args.ltc, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("ready", report.get("eligible", report.get("promote", True))) else 3
    except (CampaignError, OSError) as error:
        print(f"OpenBench campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
