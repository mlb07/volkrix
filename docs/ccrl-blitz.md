# Volkrix CCRL Blitz candidate

Volkrix is an independent Rust UCI engine. This candidate uses the large
Stockfish 18 SFNNv10 network as a separate CC0 data file. The archive contains
the executable, network, license and third-party notices, and UCI smoke
transcripts. The archive's SHA-256 sidecar identifies the exact package, and
`build-info.txt` records its source commit and toolchain.

Use the x86-64 archive for the host operating system. Extract it with the
network file beside the executable. On startup Volkrix discovers that file
automatically. The executable should report `id name Volkrix 0.2.4-rc.1`.
Use the `nn-c288c895ea92.nnue` file in the archive if the GUI explicitly sets
`EvalFile`.

For a single-threaded test, set `Threads=1` and the same `Hash` size for
all engines. CCRL Blitz uses 128 or 256 MiB for single-threaded engines,
pondering off, a shared opening book limited to 12 moves per side, and the
equivalent of two minutes plus one second per move on its reference machine.
Set `SyzygyPath` to the tester's tablebases and `SyzygyProbeLimit=6` when
using the six-piece set. Leave `DualEvalPolicy=off` to use the large network
throughout.

[CCRL Blitz testing conditions](https://computerchess.org.uk/404/about.html)

The release workflow creates checksum-verified Windows x86-64, Linux x86-64,
and macOS arm64 candidate archives on pushes to `codex/full-revamp`. The Linux
candidate archive also includes the interleaved x86-64 search benchmark and
the cached-versus-fresh NNUE profile. The same workflow runs a 20-game native
Linux x86-64 match against upstream Aspen 2.3.0 from the first ten local
calibration openings at 20+0.2. The archives and Linux match evidence remain
attached to the workflow run. Publishing a GitHub Release uploads the platform
archives to that release.

## Local strength validation

Volkrix used the same checksum-pinned Stockfish 18 network throughout the Apple
M4 tests.
The cache prototype and final candidate produced the same fixed-depth node count
and checksum as their local control. The final candidate also matched the
control's scores, principal variations, best moves, and node counts over 35
fixed-depth positions.

The final candidate used 25.5% less CPU time in the nine interleaved
single-thread depth-8 benchmark pairs. Both binaries searched 476,971 nodes
with checksum `ae517346e724f59b`.

| Candidate | Starting time + increment | W–D–L | Score | Paired Elo and 95% interval |
| --- | --- | --- | --- | --- |
| King refresh cache prototype | 4 s + 40 ms | 103–98–55 | 59.4% | +66 [ +38, +95 ] |
| 0.2.4-rc.1 | 15 s + 150 ms | 92–89–75 | 53.3% | +23 [ −3, +50 ] |

Both stages finished with zero abnormal games. The prototype used the old
development version label and kept the cache index inline. The release
candidate moves that index to the heap and uses the same cache algorithm.
The longer-clock interval includes zero.

These two stages used one thread, 64 MiB hash, and disjoint
`UHO_Lichess_4852_v1.epd` opening pairs. A separate 128-game match compared
the entire release candidate with the published branch tip `7257bc9` at
8+0.08. It finished 35–58–35, for 50% and a paired Elo interval of −36 to +36,
with zero abnormal games. That match used one thread, 64 MiB hash and 64 fresh
opening pairs from `8moves_v3.epd`, drawn with Python seed 20261125 after
excluding the 24 external gauntlet openings. It used no score-based adjudication.

The external tests used one thread,
128 MiB hash, and four concurrent games, with no evaluation-based adjudication
or tablebases. Each opponent played both colors of every opening. The opening
positions came from `8moves_v3.epd` after removing duplicate lines in file
order and sampling 24 entries with Python's `random.Random(20261124)`. The
first 20 were used at 20+0.2 and the last four at 120+1. The latter is the
nominal clock on the M4, without CCRL's reference-machine calibration.

[OpenBench opening book](https://github.com/AndyGrant/openbench-books/blob/master/8moves_v3.epd.zip)

Aspen 2.3.0, Drofa 4.1.0, and Zevra 2.7 were built for the M4 from their
tagged source. Aspen uses its original bundled weights, a SIMDe
translation of the x86 intrinsics, and a 16 MiB search-thread stack to avoid
exhausting the host's default stack. Drofa uses the upstream portable
sliding-attack path. Slate 0.5.3 is the author's Apple Silicon binary. FastChess
logged protocol warnings while continuing games so a legal best move following
a stale reported line could be played. The
opponents' reference ratings are from the November 21, 2026 CCRL Blitz list,
where Aspen held rank 200 at 3258. Local comparison scores depend on the
clock, hardware, builds and opening sample. Independent testing determines
official placement.

[CCRL Blitz ranking list](https://computerchess.org.uk/404/)

### External gauntlet results

| Opponent | CCRL anchor | 20+0.2 W–D–L | Score | 120+1 W–D–L | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aspen 2.3.0 | 3258 | 6–25–9 | 46.25% | 1–5–2 | 43.75% |
| Drofa 4.1.0 | 3286 | 5–14–21 | 30% | 2–4–2 | 50% |
| Zevra 2.7 | 3304 | 5–20–15 | 37.5% | 0–3–5 | 18.75% |
| Slate 0.5.3 | 3243 | 10–17–13 | 46.25% | 3–2–3 | 50% |

Volkrix scored 40% over 160 games at 20+0.2. Fitting the expected score
against the four reference ratings gave a local estimate of 3202. Resampling
whole openings across all four opponents gave a 95% sampling interval of
3167–3236. At 120+1, Volkrix scored 40.625% over 32 games for an estimate of
3207 and an opening-sampling interval of 3160–3251. The latter estimate uses
only four shared openings. All 192 games ended normally. The two intervals
cover opening sampling on the local setup. The test host used its ordinary
AC and battery power profiles over the campaign.
