#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Volkrix developer task runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fmt")
    subparsers.add_parser("clippy")
    subparsers.add_parser("test")
    subparsers.add_parser("bench")

    release_parser = subparsers.add_parser("release")
    release_parser.add_argument("--target", required=False)

    args = parser.parse_args()

    if args.command == "fmt":
        return run(["cargo", "fmt", "--check"])
    if args.command == "clippy":
        return run(["cargo", "clippy", "--all-targets", "--all-features", "--", "-D", "warnings"])
    if args.command == "test":
        return run(["cargo", "test"])
    if args.command == "bench":
        return run(["cargo", "bench"])
    if args.command == "release":
        command = ["cargo", "build", "--release"]
        if args.target:
            command.extend(["--target", args.target])
        return run(command)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

