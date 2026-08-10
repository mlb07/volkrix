# Search parameter tuning

Volkrix has an explicitly isolated UCI parameter surface for OpenBench SPSA
experiments. It does **not** change the normal release configuration and it does
not make untested values production defaults.

## Build and inspect

Build a tuning binary with:

```sh
cargo build --release --locked --features spsa-tuning
```

The tuning binary advertises every parameter as a bounded UCI `spin` option
whose name starts with `Tune`. The exact live vector, bounds, recommended
initial steps, schema version, and deterministic checksum are available without
source parsing:

```text
setoption name TuneManifest
```

Record this output with every experiment. The checksum covers parameters in a
fixed order and changes when any live value changes. Changing a tuning option
also clears the transposition table so scores and bounds from the previous
vector cannot contaminate the next search.

## Parameter groups

The schema in [`src/search/parameters.rs`](../src/search/parameters.rs) is the
single source of truth for names, defaults, hard bounds, and suggested starting
steps. It currently covers:

- aspiration-window width;
- the logarithmic LMR divisor;
- null-move base/depth/evaluation reduction, static margin, and verification
  depth;
- reverse-futility and ordinary futility margins;
- late-move, SEE, and history-pruning thresholds;
- history bonus scaling;
- ProbCut margin and static-evaluation gate;
- increment allocation, hard-time multiplier, and stability/score-swing time
  factors.

The bounds prevent zero divisors, negative time factors, impossible reduction
shapes, and arithmetic hazards. The manifest `step` is an SPSA starting
recommendation, not a divisibility rule; every integer inside the UCI bounds is
valid.

## Recommended campaign sequence

Do not tune all groups against one opening sample and then call the result
stronger. Use independent stages:

1. Freeze compiler, target CPU, network checksum, book checksum, time control,
   concurrency, adjudication, and the base/candidate commits.
2. Tune search selectivity first: LMR, null move, futility/LMP, SEE/history, and
   ProbCut. Keep time parameters at defaults.
3. Validate the resulting vector with a paired pentanomial SPRT on a disjoint
   opening suite before using it as the base of another tune.
4. Tune time management separately at the intended tournament control. Fixed
   depth and fixed node tests cannot validate time parameters.
5. Reconfirm the combined vector at both STC and LTC, then run an external
   engine gauntlet. Promote values only after the predeclared tests pass.

Use one thread for the first search tune to avoid SMP noise. Tune high-thread
behavior in a separate campaign. Always preserve the raw worker logs,
pentanomial counts, manifest output, PGNs, and crash/time/illegal-move counts.

## Default-equivalence gate

The feature gate keeps the normal UCI surface free of experimental controls.
The tuning build starts at the exact production vector. During implementation,
an independent depth-7 start-position search with `Hash=16` produced identical
score, best move, PV, node count (`104684`), and TT hits (`32658`) in the normal
and tuning builds. A non-default LMR value changed the tree, confirming that the
live vector is connected rather than decorative.

This equivalence gate must be rerun whenever a parameter is added or a default
is changed. A tuned value is experimental evidence until it survives match
testing; never alter `SearchParameters::DEFAULT` merely because SPSA reports a
local optimum.

## Resumable local campaigns

[`scripts/local_spsa.py`](../scripts/local_spsa.py) provides a small-host SPSA
path when an OpenBench server is unavailable. Build a tuning binary with an
embedded, checksum-verified network, then select one coherent search-only group:

```sh
make -C openbench \
  EXE="$PWD/target/spsa-engine" \
  EVALFILE="$PWD/target/release/nn-c288c895ea92.nnue"

python3 scripts/local_spsa.py start \
  --fastchess /absolute/path/to/fastchess \
  --engine target/spsa-engine \
  --book /absolute/path/to/openings.pgn --book-format pgn \
  --evalfile embedded --output target/spsa-run \
  --parameters TuneLmrDivisorPct,TuneNullBaseReduction,TuneNullDepthDivisor \
  --iterations 20 --pairs-per-iteration 64 --seed 1448037451 \
  --tc 10+0.1 --concurrency 1 --threads 1 --hash-mb 64
```

`--evalfile` is mandatory. `embedded` requires the engine to advertise a
`<embedded:sha256:size>` identity. Supplying an external network path instead
copies and hashes that network into the campaign. The engine, FastChess, book,
and external network are all frozen and revalidated on every resume.

The book must contain at least `iterations * pairs-per-iteration` openings.
Each iteration records and uses a deterministic, disjoint, sequential opening
block derived from the fixed seed; it never restarts at opening one. Each
opening is played as a color-reversed pair. Resume or inspect a run with:

```sh
python3 scripts/local_spsa.py resume --lab target/spsa-run
python3 scripts/local_spsa.py inspect --lab target/spsa-run
```

Only one process may own a campaign. Interrupted FastChess iterations resume
from their saved `recovery.json`; completed cached results are accepted only
when their iteration, exact plus/minus vectors, expected game count, and W/D/L
counts match. Checkpoints are replaced atomically.

`manifest.json` freezes inputs and settings and is guarded by
`manifest.sha256`. Every iteration retains its exact
vectors, signed coordinate spans, actual half-radii, opening start, command,
console output, PGN, engine log, recovery state, and validated result.
`recommended.json` contains the bounded integer vector, live TuneManifest,
total sample size, final comparison score and standard error, Elo estimate, and
the current parameter resolution.

The update uses the actual signed span `plus - minus`, including clipped or odd
boundary spans. Even so, a small local campaign is noisy gradient exploration,
not promotion evidence and not a reliable Elo measurement. Freeze its proposed
vector into a separate candidate and require an independent, adequately sized
pentanomial SPRT on unused openings at STC and LTC before changing production
defaults.

## Initial certified local run (2026-08-08)

The first evidence-hardened smoke campaign used the large embedded Stockfish 18
network, one thread, 64 MiB hash, `0.2+0.02`, ten iterations, and eight paired
openings per iteration. Its 160 games used the disjoint opening range
14127--14206 and completed with no crash, stall, time-forfeit, illegal-move, or
unknown termination. The exploratory rounded proposal changed only three
values: `TuneFutilityBase=89`, `TuneFutilitySlope=119`, and
`TuneHistoryBonusScale=31`.

The proposal then played a separate 100-game, 50-pair fixed-size confirmation
at `1+0.01` on held-out openings 20001--20050 against the same frozen binary at
production defaults. It scored 35 wins, 26 draws, and 39 losses (48.0%, about
-13.9 logistic Elo, pair-aware approximate 95% interval -69.7 to +41.2 Elo),
with pentanomial counts `[4, 16, 16, 8, 6]` and zero abnormal terminations.
That result does not support promotion, so all production defaults remain
unchanged. The local evidence is retained under
`target/strength-baselines/local-spsa-search-10x8-certified` and
`target/strength-baselines/local-spsa-confirmation-100`.

## Elite selectivity campaign (2026-08-09)

A larger follow-up tuned the still-unexplored selectivity group:
`TuneNullVerifyDepth`, late-move base and slope, SEE margin, history-pruning
threshold, and the three ProbCut controls. The campaign used the checksum-
frozen large embedded network and tuning engine, one engine thread, 64 MiB
hash, concurrency four, `0.2+0.02`, 32 iterations, and 32 color-reversed
opening pairs per iteration. All 2,048 games on openings 24001--25024 ended
normally. The rounded candidate changed only:

- `TuneSeeMargin`: 70 to 68;
- `TuneHistoryPruneThreshold`: 2000 to 2048;
- `TuneProbCutBase`: 180 to 179.

The other five coordinates rounded back to production defaults. The complete
campaign, including the frozen manifest, every command, PGN, worker log,
checkpoint, and recommendation, is retained under
`target/strength-lab/elite-selectivity-spsa-32x32`.

The fixed candidate then passed its predeclared short-control gate on 500 fresh
paired openings (29001--29500): 457 wins, 146 draws, and 397 losses over 1,000
games, or +20.871 logistic Elo with a pair-aware approximate 95% interval from
+5.320 to +36.506 Elo. Pentanomial counts were `[46, 35, 292, 67, 60]`, with
zero failures. That result enabled the required longer-control confirmation;
it did not authorize changing defaults by itself.

At `1+0.01`, the first 1,000 games were positive but inconclusive (426 wins,
168 draws, 406 losses; +6.950 Elo, 95% interval -8.465 to +22.392), triggering
the predeclared 3,000-game extension. Across all 4,000 longer-control games on
fresh openings 32001--34000, the candidate scored 1,544 wins, 901 draws, and
1,555 losses: -0.955 Elo with a pair-aware approximate 95% interval of -8.597
to +6.685 and pentanomial counts `[179, 292, 1073, 273, 183]`. There were zero
failures.

The longer-control confirmation gate therefore failed. The candidate is
rejected or deferred, and all production search defaults remain unchanged.
The verified match labs and frozen combined decision are retained under
`target/strength-lab/elite-spsa-candidate-*`.
