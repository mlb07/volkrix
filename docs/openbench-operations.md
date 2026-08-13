# Audited distributed campaign operations

`scripts/openbench_campaign.py` is the repository-side control plane for a
deployed Volkrix OpenBench instance. It does not create users, upload assets,
start workers, create workloads, approve tests, or modify a server. It freezes
the operator's intent before a workload exists, checks the server and reference
worker, exports completed evidence through read-only endpoints, and makes a
deterministic promotion decision.

## Audited upstream boundary

The interface was audited at the exact OpenBench commit recorded in
`openbench/upstream-lock.json`:

- The pinned URL table exposes authenticated client/FastChess version records,
  configuration, network, PGN, SPSA, and workload result endpoints
  ([official source](https://github.com/AndyGrant/OpenBench/blob/9906bad18e044c4b455539317b8cf037393a7218/OpenBench/urls.py#L60-L83)).
- Workload creation is an authenticated browser form, not an API
  ([official source](https://github.com/AndyGrant/OpenBench/blob/9906bad18e044c4b455539317b8cf037393a7218/OpenBench/workloads/create_workload.py#L46-L107)).
- The workload APIs expose aggregate info, per-worker results, and grouped
  summaries; completed PGN archives have an additional active-worker and
  processing check
  ([official source](https://github.com/AndyGrant/OpenBench/blob/9906bad18e044c4b455539317b8cf037393a7218/OpenBench/views.py#L945-L1030)).
- Per-worker results include pentanomial counts, crash/time-loss counters, and
  active state
  ([official source](https://github.com/AndyGrant/OpenBench/blob/9906bad18e044c4b455539317b8cf037393a7218/OpenBench/workloads/view_workload.py#L65-L96)).

There is no read-only machine-audit endpoint and no supported workload-creation
API at this revision. The tool therefore consumes `openbench_deploy.py audit`
output for a local reference worker and emits exact form fields for deliberate
browser submission. It never imitates the worker registration endpoint, which
would create server state.

The SPSA API exposes original parameters, current output values, and a digest,
but not every stored methodology field. The frozen lock and its digest-bearing
`info` field preserve the intended alpha, gamma, A-ratio, reporting, and
distribution settings; the export independently verifies all parameter inputs.
Until upstream exposes the remaining fields, an approver must compare the
created tune page to the frozen form artifact before approval. This limitation
is explicit rather than silently claiming stronger verification than the API
provides.

At this revision ordinary test creation replaces the submitted `info` value
with the development commit message, while tune creation preserves it
([official source](https://github.com/AndyGrant/OpenBench/blob/9906bad18e044c4b455539317b8cf037393a7218/OpenBench/workloads/create_workload.py#L109-L203)).
Consequently no-change/STC/LTC exports bind the lock by comparing every exposed
workload field, not by pretending the digest tag survived. SPSA uses both the
field comparison and its preserved digest tag.

## Freeze four workload types

Copy `openbench/campaign.example.json` outside the repository and replace every
placeholder. Never use a branch name: use full public 40-character commits and
their exact `bench` results. The example is deliberately invalid until those
values are supplied.

Create one specification per workload:

| Kind | Required changes from the example |
| --- | --- |
| `no-change` | identical `dev` and `base`; mode `GAMES`; even `max_games` of at least 1000; add `policy` with `max_abs_elo` and `require_ci_contains_zero: true`; remove SPRT bounds/confidence |
| `stc` | example shape; normal iteration book and STC time/options |
| `ltc` | LTC time/options/workload size and a separately sourced, held-out book |
| `spsa` | remove `base`; mode `SPSA`; remove SPRT fields; add the SPSA methodology and non-empty bounded input list |

Freeze each JSON file into a canonical, digest-bound lock. The destination must
not already exist:

```bash
python3 scripts/openbench_campaign.py freeze \
  --spec /absolute/path/stc.json \
  --output /absolute/path/stc.lock.json
python3 scripts/openbench_campaign.py form \
  --lock /absolute/path/stc.lock.json \
  --output /absolute/path/stc.form.json
```

The form artifact names the official `/test/new/` or `/tune/new/` endpoint and
every field to enter. Its `info` value contains the frozen SHA-256; the pinned
server preserves that tag for SPSA but not ordinary tests, as described above.
Do not alter the lock after workload creation; create a new lock instead.
This schema is intentionally specific to the current Stockfish 18 production
network: every dev/base side must use the full
`c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7`
SHA-256. Testing a replacement network requires a reviewed schema/policy change.
The frozen full digest stays lowercase; generated form fields and server API
records use OpenBench's exact uppercase eight-character identity, `C288C895`.

## Reference-worker and server preflight

First run the existing deployment audit with all three immutable inputs. It
must report `ready: true`:

```bash
python3 scripts/openbench_deploy.py audit \
  --openbench-root /absolute/path/to/pinned/OpenBench \
  --fastchess-root /absolute/path/to/pinned/fastchess \
  --network /absolute/path/to/nn-c288c895ea92.nnue \
  --json > /absolute/path/deployment-audit.json
```

Run at least five simultaneous-reference bench sets using the exact thread and
socket policy in the workload lock, then bind their observed NPS values to the
audit. The worker record fails if the pinned revisions/network are missing, the
versioned deployment audit is not ready or lacks its complete lock, machine,
tool, checkout, and network fact set, the coefficient of variation exceeds policy, or
the median differs too far from reference NPS:

```bash
python3 scripts/openbench_campaign.py worker-record \
  --lock /absolute/path/stc.lock.json \
  --deployment-audit /absolute/path/deployment-audit.json \
  --bench-nps 1001200,998700,1000400,1000900,999800 \
  --output /absolute/path/reference-worker.json
```

Preflight performs only authenticated reads. Credentials remain in the process
environment, HTTP is rejected except for localhost, redirects are rejected,
and the output must be new:

```bash
export OPENBENCH_USERNAME=REPLACE
export OPENBENCH_PASSWORD=REPLACE
python3 scripts/openbench_campaign.py preflight \
  --lock /absolute/path/stc.lock.json \
  --server https://openbench.example.com \
  --worker-record /absolute/path/reference-worker.json \
  --output /absolute/path/stc.preflight.json
```

The check binds the server's client and FastChess refs, engine/build contract,
reference NPS, exact book digest, default network name/prefix, and the worker
record to the workload digest. A missing field is a hard failure.

## Export, verify, and promote

After an approved workload finishes and all workers detach, export it. Frozen
specifications require `COMPACT` or `VERBOSE` uploads, and export always retains
the completed PGN archive:

```bash
python3 scripts/openbench_campaign.py export \
  --lock /absolute/path/stc.lock.json \
  --server https://openbench.example.com \
  --worker-record /absolute/path/reference-worker.json \
  --workload-id 123 \
  --output /absolute/path/results/stc
```

Export first repeats the complete server/worker preflight and records that fresh
snapshot. It then refuses active, unfinished, deleted, or errored workloads; mismatched
commits, benches, networks, options, controls, scaling, book, adjudication,
bounds, confidence, or lock tag; any crash/time loss; and any disagreement
between aggregate and per-worker games/pentanomial counts. It validates the tar
and bzip2 structure of every PGN member and requires the archive to contain at
least every accepted game. The pinned client can archive already-played games
after another worker closes the workload; those games were not accepted into
the server counters. Export records `accepted_games` and `surplus_games`
explicitly, never includes the surplus in statistics, and rejects an archive
that omits accepted play. The public pinned API does not expose enough current
assignment geometry to derive a sound upper bound for that legitimate surplus.

The exported evidence includes the canonical workload lock and an exact set of
raw artifacts, all hashed in `result.json`. Promotion reloads the lock and raw
info/results/PGNs, recomputes identity, statistics, and eligibility, and rejects
manifest edits, omissions, substitutions, symlinks, or unexpected artifacts.
A no-change result is eligible
only if it meets its predeclared absolute-Elo threshold and its pair-aware
approximate 95% interval contains zero. STC/LTC eligibility requires OpenBench's
SPRT pass state.

Create the final decision from three independently verified result directories:

```bash
python3 scripts/openbench_campaign.py promote \
  --no-change /absolute/path/results/no-change \
  --stc /absolute/path/results/stc \
  --ltc /absolute/path/results/ltc \
  --output /absolute/path/promotion.json
```

Promotion is denied unless all policies pass, no-change tested the exact base,
STC and LTC use identical candidate/base commits and network, and LTC uses a
different held-out book digest. A rejected decision is still written for the
audit trail and the command exits nonzero.

## Local verification

```bash
python3 scripts/test_openbench_campaign.py -v
```
