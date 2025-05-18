# Advanced Machine Learning for Trading

This directory contains a comprehensive ML pipeline for financial trading, featuring multiple advanced models:

- **Ensemble models** (Bayesian Ridge, Gradient Boosting, Random Forest, Gaussian Process)
- **Boosting models** (AdaBoost, XGBoost)
- **Deep learning models** (RNNs, LSTMs, CNNs)
- **Physics-based neural networks** (PINNs)
- **Hybrid stacked models** (LightGBM + CNN-LSTM + FinBERT-BiLSTM)

The system is designed to seamlessly integrate with the C++ trading engine while providing a full-featured Python ML pipeline.

## Requirements

- Python 3.7+
- NumPy, Pandas, Scikit-learn
- TensorFlow 2.x (for neural network models)
- XGBoost
- LightGBM (for hybrid models)
- Matplotlib (for visualization)

Install all dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow xgboost lightgbm matplotlib
```

## Data Format

The system expects preprocessed data files in CSV format with:
- Timestamp column (ISO format)
- Asset-specific columns with prefixes (e.g., `btc_close`, `eth_volume`)
- Target columns with format `{asset}_r_next` (e.g., `btc_r_next`)

Data files should be placed in the `data/ml_data/` directory with names like:
- `processed_data_btc.csv`
- `processed_data_eth.csv`
- etc.

## Training Models

### Train All Models

To train all model types for all assets:

```bash
python python_ml/train_all_models.py --assets btc eth ada solana
```

Options:
- `--data-dir`: Directory with preprocessed data (default: `data/ml_data`)
- `--model-dir`: Directory to save models (default: `python_ml/models`)
- `--assets`: Assets to train models for (default: btc, eth, ada, solana)
- `--no-tune`: Skip hyperparameter tuning
- `--test-after`: Run tests after training

### Train Specific Models

To train specific model types:

```bash
python python_ml/real_data_trainer.py --model-types ensemble xgboost rnn hybrid_stacked
```

## Making Predictions

### Real-Time Prediction

For real-time prediction with a specific model type:

```bash
python python_ml/real_time_predictor.py --data-file data/ml_data/processed_data_btc.csv --model-type rnn
```

Options:
- `--model-type`: Model to use (ensemble, adaboost, xgboost, rnn, cnn, pinn, hybrid_stacked)
- `--assets`: Assets to predict for
- `--confidence`: Confidence threshold (default: 0.6)

### Online Learning Mode

For online learning and prediction:

```bash
python python_ml/online_ml_trading.py --data-file data/ml_data/processed_data_btc.csv
```

## C++ Integration

The Python ML models can be easily integrated with the C++ trading engine using the bridge:

```bash
python python_ml/cpp_integration.py --mode predict --input latest_data.csv --output signals.csv --model-type hybrid_stacked
```

Modes:
- `predict`: Make predictions only
- `update`: Update models with new data
- `update_and_predict`: Update models and make predictions
- `compare`: Compare predictions from different model types

### Integration Example

From your C++ code, call the Python bridge as a subprocess:

```cpp
// Example of calling the Python bridge from C++
std::string command = "python3 python_ml/cpp_integration.py --mode predict --input "
                    + data_file + " --output " + output_file
                    + " --model-type " + model_type;
int result = system(command.c_str());
```

Then read the signals from the output file:

```cpp
// Read signals from the output file
std::ifstream signal_file(output_file);
// Process signals...
```

## Model Types

1. **ensemble**: Combines multiple models (Bayesian Ridge, GBM, RF, GP)
2. **adaboost**: AdaBoost regressor
3. **xgboost**: XGBoost regressor
4. **rnn**: Recurrent Neural Networks (LSTM and GRU)
5. **cnn**: Convolutional Neural Networks for time series
6. **pinn**: Physics-Informed Neural Networks
7. **hybrid_stacked**: LightGBM + CNN-LSTM + BiLSTM (simplified FinBERT)

## Testing and Evaluation

Run the test script to evaluate all models:

```bash
python python_ml/test_advanced_models.py --mode all
```

Modes:
- `train`: Train and evaluate models
- `predict`: Compare predictions from different models
- `bridge`: Test the C++ bridge
- `all`: Run all tests

## Directory Structure

```
python_ml/
├── advanced_models.py        # Advanced ML model implementations
├── bayesian_ml_engine.py     # Core ML engine with ensemble methods
├── cpp_integration.py        # C++ integration bridge
├── feature_engineering.py    # Feature engineering utilities
├── online_ml_trading.py      # Online learning system
├── real_data_trainer.py      # Training pipeline for real data
├── real_time_predictor.py    # Real-time prediction system
├── test_advanced_models.py   # Testing script for advanced models
├── test_real_data.py         # Testing script for real data
└── train_all_models.py       # Script to train all model types
```

## Model Directory Structure

After training, models are saved in the model directory with the following structure:

```
python_ml/models/
├── btc/                      # Models for BTC
│   ├── bayesian_ridge_model.joblib
│   ├── xgboost_model.joblib
│   ├── rnn_lstm_model/      # TensorFlow model directory
│   ├── ...
│   └── feature_cols.txt     # Feature list for this asset
├── eth/                      # Models for ETH
│   ├── ...
├── plots/                    # Performance plots
│   ├── btc_predictions.png
│   ├── btc_model_comparison.png
│   └── ...
└── ...
```

## Performance Considerations

- Neural network models (RNN, CNN, PINN, Hybrid) require more computational resources but can capture complex patterns.
- The ensemble model provides good balance between performance and speed.
- For real-time trading with limited resources, consider using XGBoost or the ensemble model.
- The C++ bridge allows updating models right before making predictions to ensure the latest data is incorporated.

## Advanced Usage

### Custom Hyperparameter Tuning

For more extensive hyperparameter tuning:

```bash
python python_ml/real_data_trainer.py --assets btc --model-types xgboost
```

Then modify the `_tune_hyperparameters` method in `real_data_trainer.py` with your desired parameter grid.

### Physics-Based Model Customization

To customize the physics constraints in the PINN model, modify the `_physics_constraint` method in `advanced_models.py` to incorporate your domain knowledge of financial markets. 