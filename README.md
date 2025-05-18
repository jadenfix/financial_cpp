# Financial C++/Python ML Trading System

This project implements a Bayesian machine learning trading system with a C++ trading engine and Python ML components.

## Architecture

The system consists of:

1. **Python ML Components** (`python_ml/`)
   - `bayesian_ml_engine.py`: Core ML engine with Bayesian models for uncertainty estimation
   - `ml_predictor.py`: Script for making predictions from the C++ engine
   - `mock_model.py`: Mock model for testing without real models
   - `models_test/`: Directory for trained and mock models

2. **C++ Trading Engine** (`cpp/`)
   - `ml_bridge.h` and `ml_bridge.cpp`: Bridge for calling Python ML models from C++
   - Trading decision logic based on predictions and uncertainty

## Features

- **Bayesian Uncertainty Estimation**: ML models provide both predictions and uncertainty estimates
- **Position Sizing**: Trading position sizes can be adjusted based on prediction confidence
- **Mock Mode**: Testing can be done with mock models when real models aren't available
- **Multiple Models**: Supports ensembles of different ML models (Bayesian Ridge, XGBoost, AdaBoost, neural networks)

## How It Works

1. The C++ trading engine collects market data (OHLCV)
2. The ML bridge converts the data to JSON and calls the Python predictor
3. The Python predictor loads the appropriate models and makes predictions
4. The predictor returns both prediction and uncertainty values
5. The C++ engine makes trading decisions based on the prediction/uncertainty

## Usage

### Running the Test Program

```bash
# Compile the test program
g++ -std=c++17 ml_bridge.cpp -o ml_bridge_test

# Run the test program
./ml_bridge_test
```

### Using Mock Models

To use mock models for testing:

```bash
# Create mock model files
python3 python_ml/mock_model.py --model-dir python_ml/models_test --asset btc
```

The bridge can be configured to use mock models:

```cpp
// Use mock models (set last parameter to true)
MLBridge mlBridge(pythonPath, scriptPath, modelDir, "btc", true);
```

## Debugging

- Check `python_ml/models_test/btc/` directory for model files
- Set the `--use-mock` flag when running `ml_predictor.py` directly
- Examine raw output with the detailed logging in both C++ and Python components
