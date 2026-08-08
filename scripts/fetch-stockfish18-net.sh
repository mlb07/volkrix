#!/usr/bin/env sh
set -eu

umask 022

case "${1:-}" in
    --small)
        net_name="nn-37f18f62d772.nnue"
        expected_sha256="37f18f62d772f3107e1d6aaca3898c130c3c86f2ab63e6555fbbca20635a899d"
        shift
        ;;
    --big|"")
        if [ "${1:-}" = "--big" ]; then
            shift
        fi
        net_name="nn-c288c895ea92.nnue"
        expected_sha256="c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7"
        ;;
    --*)
        printf 'usage: %s [--big|--small] [destination]\n' "$0" >&2
        exit 2
        ;;
    *)
        net_name="nn-c288c895ea92.nnue"
        expected_sha256="c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7"
        ;;
esac
source_url="https://tests.stockfishchess.org/api/nn/${net_name}"
destination="${1:-${net_name}}"

if [ "$#" -gt 1 ]; then
    printf 'usage: %s [--big|--small] [destination]\n' "$0" >&2
    exit 2
fi

if [ -d "$destination" ]; then
    destination="${destination%/}/${net_name}"
fi

destination_dir=$(dirname "$destination")
if [ ! -d "$destination_dir" ]; then
    printf 'destination directory does not exist: %s\n' "$destination_dir" >&2
    exit 1
fi

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf 'a SHA-256 implementation (sha256sum or shasum) is required\n' >&2
        return 127
    fi
}

if [ -e "$destination" ] || [ -L "$destination" ]; then
    if [ ! -f "$destination" ]; then
        printf 'refusing to replace non-file destination: %s\n' "$destination" >&2
        exit 1
    fi
    existing_sha256="$(sha256_file "$destination")"
    if [ "$existing_sha256" = "$expected_sha256" ]; then
        printf '%s already exists and is verified\n' "$destination"
        exit 0
    fi
    printf 'refusing to overwrite %s: SHA-256 is %s, expected %s\n' \
        "$destination" "$existing_sha256" "$expected_sha256" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    printf 'curl is required to download the network\n' >&2
    exit 127
fi

# Keep the temporary file beside the destination so the final rename is atomic.
temporary_file="$(mktemp "${destination_dir}/.volkrix-${net_name}.XXXXXX")"
trap 'rm -f "$temporary_file"' EXIT HUP INT TERM

curl --fail --location --retry 3 --silent --show-error \
    --output "$temporary_file" "$source_url"
actual_sha256="$(sha256_file "$temporary_file")"
if [ "$actual_sha256" != "$expected_sha256" ]; then
    printf 'downloaded network SHA-256 is %s, expected %s\n' \
        "$actual_sha256" "$expected_sha256" >&2
    exit 1
fi

chmod 0644 "$temporary_file"
mv "$temporary_file" "$destination"
trap - EXIT HUP INT TERM
printf 'downloaded and verified %s\n' "$destination"
