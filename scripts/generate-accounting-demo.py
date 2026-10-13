#!/usr/bin/env python3
"""Record the real C++ accounting scenario and render its source-output demo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "docs/demo"
RUNTIME_INPUTS = [
    "examples/accounting_demo.cpp",
    "src/core/Portfolio.h",
    "src/core/Event.h",
    "src/core/Utils.h",
    "src/data/PriceBar.h",
]


def run(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def source_manifest(source_commit: str) -> dict:
    entries = []
    for path in RUNTIME_INPUTS:
        data = subprocess.run(
            ["git", "show", f"{source_commit}:{path}"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        entries.append(
            {
                "path": path,
                "git_blob": run("git", "rev-parse", f"{source_commit}:{path}"),
                "sha256": sha256(data),
            }
        )
    return {
        "source_commit": source_commit,
        "source_tree": run("git", "rev-parse", f"{source_commit}^{{tree}}"),
        "runtime_inputs": entries,
    }


def record_receipt() -> tuple[str, list[dict[str, str]]]:
    with tempfile.TemporaryDirectory(prefix="financial-accounting-demo-") as directory:
        executable = Path(directory) / "accounting-demo"
        subprocess.run(
            [
                "c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                "-I", str(ROOT / "src"), str(ROOT / "examples/accounting_demo.cpp"), "-o", str(executable),
            ],
            check=True,
        )
        completed = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    receipt_lines = [line.removeprefix("RECEIPT,") for line in completed.stdout.splitlines() if line.startswith("RECEIPT,")]
    text = "label,direction,fill_quantity,fill_price,position,average_price,realized_pnl,cash,commission\n"
    text += "\n".join(receipt_lines) + "\n"
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != 6:
        raise RuntimeError(f"expected 6 recorded transitions, got {len(rows)}")
    return text, rows


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str = "#243855") -> None:
    draw.rounded_rectangle(box, radius=20, fill=fill, outline=outline, width=2)


def render_frame(rows: list[dict[str, str]], frame_index: int, frame_count: int) -> Image.Image:
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), "#07111f")
    draw = ImageDraw.Draw(image)
    title = font(34, True)
    subtitle = font(18)
    metric = font(28, True)
    small = font(16)
    tiny = font(13)

    draw.text((54, 38), "SIGNED-DELTA POSITION ACCOUNTING", font=title, fill="#f5f8ff")
    draw.text((56, 86), "ACTUAL C++ SIMULATION  •  FIXTURE DATA  •  NO LIVE ORDERS", font=subtitle, fill="#6ee7d8")
    draw.rounded_rectangle((1012, 40, 1225, 78), radius=19, fill="#12352f")
    draw.text((1034, 49), "VERIFIED REPLAY", font=small, fill="#7df9df")

    active = min(len(rows) - 1, int(frame_index / frame_count * len(rows)))
    progress = (frame_index + 1) / frame_count
    current = rows[active]

    panel(draw, (48, 128, 1232, 258), "#0d1b2e")
    step_width = 174
    for index, row in enumerate(rows):
        x = 72 + index * 190
        color = "#5eead4" if index <= active else "#32445e"
        draw.line((x + 12, 178, x + step_width, 178), fill=color, width=5)
        draw.ellipse((x, 166, x + 24, 190), fill=color)
        label = row["label"].replace("_", " ").upper()
        draw.text((x, 202), label, font=tiny, fill="#e7eefc" if index <= active else "#73829a")
        draw.text((x, 224), f'{row["direction"]} {float(row["fill_quantity"]):.0f} @ {float(row["fill_price"]):.0f}', font=tiny, fill=color)

    panel(draw, (48, 282, 758, 654), "#0d1b2e")
    draw.text((76, 309), "POSITION LEDGER", font=subtitle, fill="#a7b8d4")
    chart_left, chart_right = 94, 720
    center = 469
    draw.line((chart_left, center, chart_right, center), fill="#53647c", width=2)
    for index, row in enumerate(rows):
        x = chart_left + index * 105
        q = float(row["position"])
        y = center - q * 12
        fill = "#5eead4" if q >= 0 else "#fb7185"
        dim = index > active
        if dim:
            fill = "#32445e"
        draw.rounded_rectangle((x, min(center, y), x + 62, max(center, y)), radius=8, fill=fill)
        if index <= active:
            draw.text((x + 20, y - 25 if q >= 0 else y + 8), f"{q:+.0f}", font=small, fill="#eef4ff")
        draw.text((x, 574), str(index + 1), font=tiny, fill="#7d8da8")
    draw.text((92, 610), "LONG", font=small, fill="#5eead4")
    draw.text((660, 610), "SHORT", font=small, fill="#fb7185")

    panel(draw, (782, 282, 1232, 654), "#0d1b2e")
    draw.text((810, 309), f'STEP {active + 1}  /  {len(rows)}', font=subtitle, fill="#a7b8d4")
    draw.text((810, 350), current["label"].replace("_", " ").upper(), font=metric, fill="#f5f8ff")
    values = [
        ("POSITION", f'{float(current["position"]):+.0f} BTC', "#5eead4" if float(current["position"]) >= 0 else "#fb7185"),
        ("AVERAGE COST", f'${float(current["average_price"]):.2f}', "#c4b5fd"),
        ("REALIZED P&L", f'${float(current["realized_pnl"]):+.2f}', "#fbbf24"),
        ("CASH", f'${float(current["cash"]):,.2f}', "#93c5fd"),
    ]
    for index, (label, value, color) in enumerate(values):
        y = 414 + index * 54
        draw.text((812, y), label, font=small, fill="#7f91ad")
        draw.text((1000, y - 5), value, font=metric if index < 3 else subtitle, fill=color)

    draw.text((54, 682), "Observed values come from examples/accounting_demo.cpp at the pinned source commit.", font=tiny, fill="#6f819d")
    bar = int((width - 108) * progress)
    draw.rounded_rectangle((54, 704, width - 54, 710), radius=3, fill="#1e314b")
    draw.rounded_rectangle((54, 704, 54 + bar, 710), radius=3, fill="#5eead4")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", default="HEAD")
    args = parser.parse_args()
    source_commit = run("git", "rev-parse", args.source_commit)
    manifest = source_manifest(source_commit)
    receipt_text, rows = record_receipt()
    DEMO.mkdir(parents=True, exist_ok=True)
    (DEMO / "accounting-receipt.csv").write_text(receipt_text)
    (DEMO / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    frame_count, fps = 72, 6
    with tempfile.TemporaryDirectory(prefix="financial-accounting-frames-") as directory:
        frames = Path(directory)
        for index in range(frame_count):
            render_frame(rows, index, frame_count).save(frames / f"frame-{index:04d}.png", optimize=True)
        render_frame(rows, frame_count - 1, frame_count).save(DEMO / "accounting-poster.png", optimize=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "frame-%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(DEMO / "accounting-workflow.mp4")],
            check=True,
        )
        palette = frames / "palette.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "frame-%04d.png"),
             "-vf", "fps=6,scale=960:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff", str(palette)], check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "frame-%04d.png"), "-i", str(palette),
             "-lavfi", "fps=6,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0",
             str(DEMO / "accounting-workflow.gif")], check=True,
        )

    artifacts = {}
    for name in ["accounting-receipt.csv", "accounting-poster.png", "accounting-workflow.mp4", "accounting-workflow.gif"]:
        data = (DEMO / name).read_bytes()
        artifacts[name] = {"bytes": len(data), "sha256": sha256(data)}
    capture = {
        **manifest,
        "scope": "deterministic simulated fills; no brokerage, exchange, live market, profit, or latency claim",
        "recording": {"frames": frame_count, "fps": fps, "duration_seconds": frame_count / fps, "width": 1280, "height": 720},
        "observations": {
            "transitions": len(rows),
            "partial_close_position": float(rows[2]["position"]),
            "partial_close_basis": float(rows[2]["average_price"]),
            "partial_close_realized_pnl": float(rows[2]["realized_pnl"]),
            "final_position": float(rows[-1]["position"]),
            "final_average_price": float(rows[-1]["average_price"]),
            "final_realized_pnl": float(rows[-1]["realized_pnl"]),
        },
        "artifacts": artifacts,
    }
    (DEMO / "capture.json").write_text(json.dumps(capture, indent=2) + "\n")
    print(json.dumps(capture, indent=2))


if __name__ == "__main__":
    main()
