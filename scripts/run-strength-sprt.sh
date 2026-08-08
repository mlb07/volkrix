#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
    cat <<'EOF'
usage: scripts/run-strength-sprt.sh \
  --fastchess /path/to/fastchess --baseline /path/to/base \
  --candidate /path/to/dev --book /path/to/openings.epd \
  --output-dir /new/run/directory \
  (--evalfile classical|/path/to/net.nnue | \
   --baseline-evalfile classical|/path/to/net.nnue \
   --candidate-evalfile classical|/path/to/net.nnue) \
  [--book-format epd|pgn] [--tc 10+0.1] [--rounds 100000] \
  [--concurrency 1] [--threads 1] [--hash-mb 64] \
  [--move-overhead-ms 10] [--syzygy-path none|/path] \
  [--syzygy-probe-limit 7] [--syzygy-50-move-rule true|false] \
  [--baseline-small-evalfile /path/to/small.nnue] \
  [--candidate-small-evalfile /path/to/small.nnue] \
  [--baseline-dual-policy off|small-fallback] \
  [--candidate-dual-policy off|small-fallback] \
  [--baseline-dual-threshold 200] [--candidate-dual-threshold 200] \
  [--elo0 0] [--elo1 3] [--alpha 0.05] [--beta 0.05] \
  [--time-margin-ms 1000] [--skip-compliance] [--dry-run]

The output directory must not exist. The script writes immutable checksums and
the exact command before launching a color-reversed pentanomial SPRT.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

fastchess=
baseline=
candidate=
book=
book_format=
output_dir=
evalfile=
baseline_evalfile=
candidate_evalfile=
baseline_small_evalfile=
candidate_small_evalfile=
baseline_dual_policy=off
candidate_dual_policy=off
baseline_dual_threshold=200
candidate_dual_threshold=200
tc="10+0.1"
rounds=100000
concurrency=1
threads=1
hash_mb=64
move_overhead_ms=10
syzygy_path=none
syzygy_probe_limit=7
syzygy_50_move_rule=true
elo0=0
elo1=3
alpha=0.05
beta=0.05
time_margin_ms=1000
compliance=1
dry_run=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --fastchess|--baseline|--candidate|--book|--book-format|--output-dir|--evalfile|--baseline-evalfile|--candidate-evalfile|--baseline-small-evalfile|--candidate-small-evalfile|--baseline-dual-policy|--candidate-dual-policy|--baseline-dual-threshold|--candidate-dual-threshold|--tc|--rounds|--concurrency|--threads|--hash-mb|--move-overhead-ms|--syzygy-path|--syzygy-probe-limit|--syzygy-50-move-rule|--elo0|--elo1|--alpha|--beta|--time-margin-ms)
            [ "$#" -ge 2 ] || die "missing value for $1"
            name="${1#--}"
            name="${name//-/_}"
            printf -v "$name" '%s' "$2"
            shift 2
            ;;
        --skip-compliance) compliance=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown argument '$1'" ;;
    esac
done

[ -n "$fastchess" ] || die "--fastchess is required"
[ -n "$baseline" ] || die "--baseline is required"
[ -n "$candidate" ] || die "--candidate is required"
[ -n "$book" ] || die "--book is required"
[ -n "$output_dir" ] || die "--output-dir is required"
if [ -n "$evalfile" ]; then
    [ -z "$baseline_evalfile" ] || die "do not combine --evalfile with per-engine evaluator flags"
    [ -z "$candidate_evalfile" ] || die "do not combine --evalfile with per-engine evaluator flags"
    baseline_evalfile="$evalfile"
    candidate_evalfile="$evalfile"
fi
[ -n "$baseline_evalfile" ] || die "an explicit baseline evaluator is required"
[ -n "$candidate_evalfile" ] || die "an explicit candidate evaluator is required"

for numeric in rounds concurrency threads hash_mb move_overhead_ms syzygy_probe_limit time_margin_ms baseline_dual_threshold candidate_dual_threshold; do
    value="${!numeric}"
    [[ "$value" =~ ^[0-9]+$ ]] || die "--${numeric//_/-} must be an integer"
done
[ "$rounds" -gt 0 ] || die "--rounds must be positive"
[ "$concurrency" -gt 0 ] || die "--concurrency must be positive"
[ "$threads" -gt 0 ] || die "--threads must be positive"
[ "$hash_mb" -gt 0 ] || die "--hash-mb must be positive"
[ "$threads" -le 64 ] || die "--threads must be between 1 and 64"
[ "$hash_mb" -le 512 ] || die "--hash-mb must be between 1 and 512"
[ "$move_overhead_ms" -le 5000 ] || die "--move-overhead-ms must be between 0 and 5000"
[ "$syzygy_probe_limit" -le 7 ] || die "--syzygy-probe-limit must be between 0 and 7"
case "$syzygy_50_move_rule" in true|false) ;; *) die "--syzygy-50-move-rule must be true or false" ;; esac

absolute_file() {
    [ -f "$1" ] || die "file does not exist: $1"
    local directory
    directory="$(cd "$(dirname "$1")" && pwd -P)"
    printf '%s/%s\n' "$directory" "$(basename "$1")"
}

absolute_directory() {
    [ -d "$1" ] || die "directory does not exist: $1"
    (cd "$1" && pwd -P)
}

fastchess="$(absolute_file "$fastchess")"
baseline="$(absolute_file "$baseline")"
candidate="$(absolute_file "$candidate")"
book="$(absolute_file "$book")"
[ -x "$fastchess" ] || die "FastChess is not executable: $fastchess"
[ -x "$baseline" ] || die "baseline is not executable: $baseline"
[ -x "$candidate" ] || die "candidate is not executable: $candidate"

normalize_evaluator() {
    if [ "$1" = classical ]; then
        printf 'classical\n'
    else
        absolute_file "$1"
    fi
}
baseline_evalfile="$(normalize_evaluator "$baseline_evalfile")"
candidate_evalfile="$(normalize_evaluator "$candidate_evalfile")"
[ -z "$baseline_small_evalfile" ] || baseline_small_evalfile="$(absolute_file "$baseline_small_evalfile")"
[ -z "$candidate_small_evalfile" ] || candidate_small_evalfile="$(absolute_file "$candidate_small_evalfile")"
for side in baseline candidate; do
    policy_name="${side}_dual_policy"
    threshold_name="${side}_dual_threshold"
    small_name="${side}_small_evalfile"
    eval_name="${side}_evalfile"
    policy="${!policy_name}"
    threshold="${!threshold_name}"
    small="${!small_name}"
    primary="${!eval_name}"
    case "$policy" in off|small-fallback) ;; *) die "--${side}-dual-policy must be off or small-fallback" ;; esac
    [ "$threshold" -le 2000 ] || die "--${side}-dual-threshold must be between 0 and 2000"
    if [ "$policy" = small-fallback ]; then
        [ -n "$small" ] || die "--${side}-dual-policy small-fallback requires --${side}-small-evalfile"
        [ "$primary" != classical ] || die "${side} dual evaluation requires a network EvalFile"
    fi
done
if [ "$syzygy_path" != none ]; then
    syzygy_path="$(absolute_directory "$syzygy_path")"
fi

if [ -z "$book_format" ]; then
    case "${book##*.}" in
        epd|EPD) book_format=epd ;;
        pgn|PGN) book_format=pgn ;;
        *) die "--book-format is required unless the book ends in .epd or .pgn" ;;
    esac
fi
case "$book_format" in epd|pgn) ;; *) die "--book-format must be epd or pgn" ;; esac

[ ! -e "$output_dir" ] && [ ! -L "$output_dir" ] ||
    die "output directory already exists; choose a new immutable run directory"
mkdir -p "$output_dir"
output_dir="$(absolute_directory "$output_dir")"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "sha256sum or shasum is required"
    fi
}

eval_option() {
    if [ "$1" = classical ]; then
        printf 'option.EvalFile='
    else
        printf 'option.EvalFile=%s' "$1"
    fi
}
syzygy_option="option.SyzygyPath="
if [ "$syzygy_path" != none ]; then
    syzygy_option="option.SyzygyPath=$syzygy_path"
fi
baseline_dual_options=()
if [ -n "$baseline_small_evalfile" ]; then
    baseline_dual_options+=("option.SmallEvalFile=$baseline_small_evalfile")
    baseline_dual_options+=("option.DualEvalThreshold=$baseline_dual_threshold")
    baseline_dual_options+=("option.DualEvalPolicy=$baseline_dual_policy")
fi
candidate_dual_options=()
if [ -n "$candidate_small_evalfile" ]; then
    candidate_dual_options+=("option.SmallEvalFile=$candidate_small_evalfile")
    candidate_dual_options+=("option.DualEvalThreshold=$candidate_dual_threshold")
    candidate_dual_options+=("option.DualEvalPolicy=$candidate_dual_policy")
fi
repository_root="$(cd "$script_dir/.." && pwd -P)"
repository_commit=unknown
repository_dirty=unknown
if git -C "$repository_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    repository_commit="$(git -C "$repository_root" rev-parse HEAD)"
    if [ -n "$(git -C "$repository_root" status --porcelain)" ]; then
        repository_dirty=true
    else
        repository_dirty=false
    fi
fi

command=(
    "$fastchess"
    -recover -repeat -games 2 -rounds "$rounds"
    -strict
    -ratinginterval 1 -scoreinterval 1 -autosaveinterval 0
    -report penta=true -variant standard -concurrency "$concurrency"
    -openings "file=$book" "format=$book_format" order=sequential
    -engine name=Candidate "cmd=$candidate" "dir=$(dirname "$candidate")"
        "option.Threads=$threads" "option.Hash=$hash_mb"
        "option.Move Overhead=$move_overhead_ms" "$(eval_option "$candidate_evalfile")"
        "$syzygy_option" "option.SyzygyProbeLimit=$syzygy_probe_limit"
        "option.Syzygy50MoveRule=$syzygy_50_move_rule"
)
if [ -n "$candidate_small_evalfile" ]; then
    command+=("${candidate_dual_options[@]}")
fi
command+=(
    -engine name=Baseline "cmd=$baseline" "dir=$(dirname "$baseline")"
        "option.Threads=$threads" "option.Hash=$hash_mb"
        "option.Move Overhead=$move_overhead_ms" "$(eval_option "$baseline_evalfile")"
        "$syzygy_option" "option.SyzygyProbeLimit=$syzygy_probe_limit"
        "option.Syzygy50MoveRule=$syzygy_50_move_rule"
)
if [ -n "$baseline_small_evalfile" ]; then
    command+=("${baseline_dual_options[@]}")
fi
command+=(
    -each "tc=$tc" proto=uci "timemargin=$time_margin_ms"
    -sprt "elo0=$elo0" "elo1=$elo1" "alpha=$alpha" "beta=$beta" model=normalized
    -pgnout "file=$output_dir/games.pgn" append=false
    -log "file=$output_dir/fastchess.log" level=info engine=true append=false
    -config "outname=$output_dir/recovery.json"
)

{
    printf 'schema=volkrix-strength-run-v1\n'
    printf 'created_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'host=%s\n' "$(uname -a)"
    printf 'repository_commit=%s\nrepository_dirty=%s\n' \
        "$repository_commit" "$repository_dirty"
    printf 'fastchess=%s\nfastchess_sha256=%s\n' "$fastchess" "$(sha256_file "$fastchess")"
    printf 'fastchess_version=%s\n' "$("$fastchess" --version 2>&1 | head -1 || true)"
    printf 'baseline=%s\nbaseline_sha256=%s\n' "$baseline" "$(sha256_file "$baseline")"
    printf 'candidate=%s\ncandidate_sha256=%s\n' "$candidate" "$(sha256_file "$candidate")"
    printf 'book=%s\nbook_sha256=%s\nbook_format=%s\n' "$book" "$(sha256_file "$book")" "$book_format"
    printf 'baseline_evalfile=%s\n' "$baseline_evalfile"
    [ "$baseline_evalfile" = classical ] ||
        printf 'baseline_evalfile_sha256=%s\n' "$(sha256_file "$baseline_evalfile")"
    printf 'candidate_evalfile=%s\n' "$candidate_evalfile"
    [ "$candidate_evalfile" = classical ] ||
        printf 'candidate_evalfile_sha256=%s\n' "$(sha256_file "$candidate_evalfile")"
    printf 'baseline_small_evalfile=%s\nbaseline_dual_policy=%s\nbaseline_dual_threshold=%s\n' \
        "$baseline_small_evalfile" "$baseline_dual_policy" "$baseline_dual_threshold"
    [ -z "$baseline_small_evalfile" ] ||
        printf 'baseline_small_evalfile_sha256=%s\n' "$(sha256_file "$baseline_small_evalfile")"
    printf 'candidate_small_evalfile=%s\ncandidate_dual_policy=%s\ncandidate_dual_threshold=%s\n' \
        "$candidate_small_evalfile" "$candidate_dual_policy" "$candidate_dual_threshold"
    [ -z "$candidate_small_evalfile" ] ||
        printf 'candidate_small_evalfile_sha256=%s\n' "$(sha256_file "$candidate_small_evalfile")"
    printf 'syzygy_path=%s\nsyzygy_probe_limit=%s\nsyzygy_50_move_rule=%s\n' \
        "$syzygy_path" "$syzygy_probe_limit" "$syzygy_50_move_rule"
    printf 'tc=%s\nrounds=%s\nconcurrency=%s\nthreads=%s\nhash_mb=%s\n' \
        "$tc" "$rounds" "$concurrency" "$threads" "$hash_mb"
    printf 'move_overhead_ms=%s\ntime_margin_ms=%s\n' "$move_overhead_ms" "$time_margin_ms"
    printf 'sprt_elo0=%s\nsprt_elo1=%s\nsprt_alpha=%s\nsprt_beta=%s\n' \
        "$elo0" "$elo1" "$alpha" "$beta"
} > "$output_dir/manifest.txt"
printf '%q ' "${command[@]}" > "$output_dir/command.sh"
printf '\n' >> "$output_dir/command.sh"
chmod 0444 "$output_dir/manifest.txt" "$output_dir/command.sh"

if [ "$dry_run" -eq 1 ]; then
    printf 'dry run prepared at %s\n' "$output_dir"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

if [ "$compliance" -eq 1 ]; then
    "$fastchess" --compliance "$baseline" > "$output_dir/baseline-compliance.log" 2>&1
    "$fastchess" --compliance "$candidate" > "$output_dir/candidate-compliance.log" 2>&1
fi
command -v python3 >/dev/null 2>&1 || die "python3 is required for the UCI smoke gate"
baseline_smoke=(python3 "$script_dir/uci_smoke.py"
    --engine "$baseline" --evalfile "$baseline_evalfile"
    --dual-policy "$baseline_dual_policy" --dual-threshold "$baseline_dual_threshold"
    --threads "$threads" --hash-mb "$hash_mb"
    --move-overhead-ms "$move_overhead_ms"
    --syzygy-probe-limit "$syzygy_probe_limit"
    --syzygy-50-move-rule "$syzygy_50_move_rule"
    --transcript "$output_dir/baseline-uci.log")
[ "$syzygy_path" = none ] || baseline_smoke+=(--syzygy-path "$syzygy_path")
[ -z "$baseline_small_evalfile" ] || baseline_smoke+=(--small-evalfile "$baseline_small_evalfile")
"${baseline_smoke[@]}"
candidate_smoke=(python3 "$script_dir/uci_smoke.py"
    --engine "$candidate" --evalfile "$candidate_evalfile"
    --dual-policy "$candidate_dual_policy" --dual-threshold "$candidate_dual_threshold"
    --threads "$threads" --hash-mb "$hash_mb"
    --move-overhead-ms "$move_overhead_ms"
    --syzygy-probe-limit "$syzygy_probe_limit"
    --syzygy-50-move-rule "$syzygy_50_move_rule"
    --transcript "$output_dir/candidate-uci.log")
[ "$syzygy_path" = none ] || candidate_smoke+=(--syzygy-path "$syzygy_path")
[ -z "$candidate_small_evalfile" ] || candidate_smoke+=(--small-evalfile "$candidate_small_evalfile")
"${candidate_smoke[@]}"

set +e
"${command[@]}" 2>&1 | tee "$output_dir/console.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$output_dir/exit-status"
[ "$status" -eq 0 ] || die "FastChess exited with status $status; artifacts retained at $output_dir"
for artifact in "$output_dir"/*; do
    [ -f "$artifact" ] || continue
    [ "$(basename "$artifact")" = artifacts.sha256 ] && continue
    printf '%s  %s\n' "$(sha256_file "$artifact")" "$(basename "$artifact")"
done > "$output_dir/artifacts.sha256"
chmod 0444 "$output_dir"/*
printf 'completed strength run: %s\n' "$output_dir"
