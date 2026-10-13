#!/usr/bin/env python3
"""Compile and run direct accounting tests with strict warnings and UBSan."""

import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def compile_and_run(compiler: str, output: Path, extra_flags: list[str]) -> None:
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            *extra_flags,
            "-I",
            str(ROOT / "src"),
            str(ROOT / "tests/accounting_test.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    subprocess.run([str(output)], check=True, timeout=30)


def main() -> None:
    compiler = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory(prefix="financial-accounting-e2e-") as directory:
        work = Path(directory)
        compile_and_run(compiler, work / "accounting-tests", [])
        compile_and_run(
            compiler,
            work / "accounting-tests-ubsan",
            ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"],
        )
    print("PASS: strict and UBSan accounting transition executables")


if __name__ == "__main__":
    main()
