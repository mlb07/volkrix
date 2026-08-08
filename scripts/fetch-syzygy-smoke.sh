#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /absolute/output/directory" >&2
    exit 2
fi

destination=$1
case "$destination" in
    /*) ;;
    *)
        echo "error: output directory must be an absolute path" >&2
        exit 2
        ;;
esac

base_url=http://tablebase.sesse.net/syzygy/3-4-5
temporary=$(mktemp -d "${TMPDIR:-/tmp}/volkrix-syzygy.XXXXXX")
trap 'rm -rf "$temporary"' EXIT HUP INT TERM

verify_sha256() {
    file=$1
    expected=$2
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    else
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    fi
    if [ "$actual" != "$expected" ]; then
        echo "error: SHA-256 mismatch for $(basename "$file")" >&2
        exit 1
    fi
}

fetch_one() {
    name=$1
    expected=$2
    target=$destination/$name
    if [ -f "$target" ]; then
        verify_sha256 "$target" "$expected"
        echo "verified existing $target"
        return
    fi
    curl --fail --location --retry 3 --output "$temporary/$name" "$base_url/$name"
    verify_sha256 "$temporary/$name" "$expected"
    mv "$temporary/$name" "$target"
    echo "installed $target"
}

mkdir -p "$destination"
fetch_one KQvK.rtbw 517667dff787162dbb1ed9d5d6484d30ee854e686ee0675c08d99ecf045d2d50
fetch_one KQvK.rtbz 71ea9444fa5bd42897d781a0c356975ea6f23e0f65a4254e470897031c161c8c

echo "Syzygy KQvK smoke fixture ready in $destination"
