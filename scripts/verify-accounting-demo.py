#!/usr/bin/env python3
"""Verify source provenance, replay output, and full demo media decodability."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "docs/demo"


def run(*command: str) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    capture = json.loads((DEMO / "capture.json").read_text())
    manifest = json.loads((DEMO / "source-manifest.json").read_text())
    assert capture["source_commit"] == manifest["source_commit"]
    assert capture["source_tree"] == manifest["source_tree"]
    for entry in manifest["runtime_inputs"]:
        data = subprocess.run(
            ["git", "show", f'{manifest["source_commit"]}:{entry["path"]}'], cwd=ROOT, check=True, capture_output=True
        ).stdout
        assert sha256(data) == entry["sha256"], entry["path"]
        assert run("git", "rev-parse", f'{manifest["source_commit"]}:{entry["path"]}') == entry["git_blob"]

    with tempfile.TemporaryDirectory(prefix="financial-accounting-verify-") as directory:
        executable = Path(directory) / "accounting-demo"
        subprocess.run(
            ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-I", str(ROOT / "src"),
             str(ROOT / "examples/accounting_demo.cpp"), "-o", str(executable)], check=True,
        )
        result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    lines = [line.removeprefix("RECEIPT,") for line in result.stdout.splitlines() if line.startswith("RECEIPT,")]
    replay = "label,direction,fill_quantity,fill_price,position,average_price,realized_pnl,cash,commission\n" + "\n".join(lines) + "\n"
    assert replay.encode() == (DEMO / "accounting-receipt.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(replay)))
    assert len(rows) == 6
    assert float(rows[2]["position"]) == 11.0
    assert abs(float(rows[2]["average_price"]) - 106.666667) < 1e-6
    assert abs(float(rows[2]["realized_pnl"]) - 93.333333) < 1e-6
    assert float(rows[-1]["position"]) == 3.0
    assert float(rows[-1]["average_price"]) == 90.0

    for name, expected in capture["artifacts"].items():
        data = (DEMO / name).read_bytes()
        assert len(data) == expected["bytes"], name
        assert sha256(data) == expected["sha256"], name

    probe = json.loads(run(
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height:format=duration",
        "-of", "json", str(DEMO / "accounting-workflow.mp4")
    ))
    stream = probe["streams"][0]
    assert stream["codec_name"] == "h264"
    assert (int(stream["width"]), int(stream["height"])) == (1280, 720)
    assert abs(float(probe["format"]["duration"]) - 12.0) < 0.1
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(DEMO / "accounting-workflow.mp4"), "-f", "null", "-"], check=True)

    with Image.open(DEMO / "accounting-workflow.gif") as gif:
        assert gif.size == (960, 540)
        frames = 0
        while True:
            try:
                gif.seek(frames)
                gif.load()
                frames += 1
            except EOFError:
                break
        assert frames == 72, frames
    print("PASS: pinned source, replay receipt, artifact digests, and full GIF/MP4 decode")


if __name__ == "__main__":
    main()
