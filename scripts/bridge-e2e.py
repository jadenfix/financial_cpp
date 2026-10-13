#!/usr/bin/env python3
"""Train a bounded real Bayesian model, persist it, and predict through C++/Python."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import numpy as np
from sklearn.linear_model import BayesianRidge

ROOT = Path(__file__).resolve().parent.parent


def main():
    with tempfile.TemporaryDirectory(prefix="financial bridge e2e ") as directory:
        work = Path(directory)
        asset = work / "models with spaces" / "btc"
        asset.mkdir(parents=True)
        columns = ["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]
        (asset / "feature_cols.txt").write_text("\n".join(columns))
        random = np.random.default_rng(7)
        training = random.normal(size=(96, 5)) * np.array([500, 500, 500, 500, 200]) + np.array([40000, 40100, 39900, 40000, 1000])
        target = (training[:, 3] - 40000) * 1e-6 + random.normal(0, 1e-5, size=96)
        model = BayesianRidge().fit(training, target)
        joblib.dump(model, asset / "bayesian_ridge.joblib")
        driver = work / "bridge-driver"
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", str(ROOT / "tests/bridge_driver.cpp"), "-o", str(driver)], check=True)

        def predict(close):
            result = subprocess.run([str(driver), sys.executable, str(ROOT / "python_ml/ml_predictor.py"), str(asset.parent), str(close)], capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(
                    f"bridge failed for close={close}: stdout={result.stdout!r}, stderr={result.stderr!r}"
                )
            line = next(line for line in result.stdout.splitlines() if line.startswith("BRIDGE_RESULT "))
            observed, uncertainty = map(float, line.removeprefix("BRIDGE_RESULT ").split(","))
            expected = model.predict([[close, close + 100, close - 100, close, 1000]])[0]
            assert abs(observed - expected) < 1e-7, (observed, expected)
            assert np.isfinite(uncertainty) and uncertainty >= 0
            assert "(fallback)" not in result.stderr
            return {"close": close, "prediction": observed, "uncertainty": uncertainty}

        # Concurrent calls expose the old global /tmp/ml_bridge_input.json race.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(predict, (40000, 40100)))
        assert results[0]["prediction"] != results[1]["prediction"]
        missing = subprocess.run([str(driver), sys.executable, str(ROOT / "python_ml/ml_predictor.py"), str(work / "missing-model"), "40000"], capture_output=True, text=True, timeout=60)
        assert missing.returncode != 0
        assert "No trained model loaded" in missing.stderr
        assert "BRIDGE_RESULT" not in missing.stdout
        print(json.dumps({"status": "pass", "model": "trained BayesianRidge", "concurrent_predictions": results, "missing_model": "rejected"}, indent=2))


if __name__ == "__main__":
    main()
