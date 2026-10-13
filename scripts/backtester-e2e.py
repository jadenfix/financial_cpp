#!/usr/bin/env python3
"""Build the real historical backtester and run bounded generated OHLCV files."""
import csv
from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    cmake = os.environ.get("CMAKE", "cmake")
    with tempfile.TemporaryDirectory(prefix="financial-backtester-e2e-") as directory:
        root = Path(directory)
        subprocess.run([cmake, "-S", str(ROOT), "-B", str(root / "build")], check=True)
        subprocess.run([cmake, "--build", str(root / "build"), "-j2"], check=True)
        datasets = {
            "stocks_april": ["quant_seconds_data_MSFT", "quant_seconds_data_NVDA", "quant_seconds_data_google"],
            "2024_only": ["btc_2024_data", "eth_2024_data", "sol_2024_data", "ada_2024_data"],
            "2024_2025": ["2024_to_april_2025_btc_data", "2024_to_april_2025_eth_data", "2024_to_april_2025_solana_data", "2024_to_april_2025_ada_data"],
        }
        for dataset, names in datasets.items():
            path = root / "data" / dataset
            path.mkdir(parents=True)
            for index, name in enumerate(names):
                with (path / f"{name}.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Open", "High", "Low", "Close", "Volume", "date_only", "time_only"])
                    for step in range(64):
                        price = 100 + index * 10 + step * 0.1
                        timestamp = datetime(2025, 4, 1, 9, 30) + timedelta(minutes=step)
                        writer.writerow([price, price + 1, price - 1, price + 0.2, 1000, timestamp.strftime("%Y-%m-%d"), timestamp.strftime("%H:%M:%S")])
        result = subprocess.run([str(root / "build/trading_system")], cwd=root, capture_output=True, text=True, timeout=60, check=True)
        if "No strategy results" in result.stdout:
            print(result.stdout, result.stderr)
        assert "No strategy results" not in result.stdout
        assert "ERROR:" not in result.stdout + result.stderr
        completed = result.stdout.count("===== Finished Strategy:")
        assert completed > 0
        assert "Comprehensive Run Invocation Complete" in result.stdout
        print(f"PASS: real backtester completed {completed} strategy/dataset configurations on 11 persisted 64-row OHLCV files")


if __name__ == "__main__":
    main()
