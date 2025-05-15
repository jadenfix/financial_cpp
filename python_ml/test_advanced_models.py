#!/usr/bin/env python3
"""
Test script for Advanced ML Models in the Bayesian ML Trading System.

This script:
1. Tests training and prediction with all advanced models on real data
2. Compares performance of different model types
3. Demonstrates how to use the C++ bridge with different model types
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import subprocess

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from python_ml.real_data_trainer import RealDataTrainer
from python_ml.real_time_predictor import RealTimePredictor

# Try to import advanced models
try:
    from python_ml.advanced_models import (
        RNNModel, CNNModel, PhysicsInformedNN, HybridStackedModel
    )
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("Advanced models not available. This test will only run basic models.")

def check_requirements():
    """Check if all required libraries are installed"""
    required_libraries = [
        "tensorflow",
        "xgboost",
        "lightgbm",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib"
    ]
    
    missing_libraries = []
    
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            missing_libraries.append(lib)
    
    if missing_libraries:
        print("WARNING: The following required libraries are missing:")
        for lib in missing_libraries:
            print(f"  - {lib}")
        print("\nTo install missing libraries:")
        print(f"pip install {' '.join(missing_libraries)}")
        return False
    
    print("All required libraries are installed.")
    return True

def train_and_test_advanced_models(data_dir, model_dir, assets, model_types, test_size=0.2):
    """
    Train and test advanced models
    
    Args:
        data_dir: Directory with preprocessed data files
        model_dir: Directory to save trained models
        assets: List of assets to train models for
        model_types: List of model types to train
        test_size: Proportion of data to use for testing
    """
    print(f"\n===== Training Advanced Models =====")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Assets: {assets}")
    print(f"Model types: {model_types}")
    
    # Create trainer
    trainer = RealDataTrainer(
        data_dir=data_dir,
        model_dir=model_dir,
        test_size=test_size,
        model_types=model_types
    )
    
    # Load data
    trainer.load_data(assets=assets)
    
    # Train models with minimal hyperparameter tuning to save time
    trainer.train_all_models(
        assets=assets,
        hyperparameter_tuning=False
    )
    
    print("\n===== Training Results =====")
    for asset, result in trainer.results.items():
        print(f"{asset.upper()}: Ensemble MSE={result['mse']:.8f}, R²={result['r2']:.4f}")
        
        # Print results for individual model types
        if 'model_results' in result:
            for model_type, metrics in result['model_results'].items():
                if model_type != 'ensemble':  # Already printed above
                    print(f"  {model_type}: MSE={metrics['mse']:.8f}, R²={metrics['r2']:.4f}")

def compare_model_predictions(data_file, model_dir, assets, model_types):
    """
    Compare predictions from different model types
    
    Args:
        data_file: CSV file with market data
        model_dir: Directory with trained models
        assets: List of assets to make predictions for
        model_types: List of model types to compare
    """
    print(f"\n===== Comparing Model Predictions =====")
    print(f"Data file: {data_file}")
    print(f"Model directory: {model_dir}")
    print(f"Assets: {assets}")
    print(f"Model types: {model_types}")
    
    # Load data
    print(f"Loading market data from {data_file}...")
    market_data = pd.read_csv(data_file)
    
    # Convert timestamp to datetime if present
    if 'timestamp' in market_data.columns:
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
        market_data['date_only'] = market_data['timestamp'].dt.date
        market_data['time_only'] = market_data['timestamp'].dt.time
        market_data.set_index('timestamp', inplace=True)

    # Test: Check that date_only and time_only columns exist and are correct
    assert 'date_only' in market_data.columns, "date_only column missing after timestamp conversion"
    assert 'time_only' in market_data.columns, "time_only column missing after timestamp conversion"
    # Check that the first date_only and time_only match the timestamp
    ts = market_data.index[0]
    assert market_data.iloc[0]['date_only'] == ts.date(), f"date_only mismatch: {market_data.iloc[0]['date_only']} != {ts.date()}"
    assert market_data.iloc[0]['time_only'] == ts.time(), f"time_only mismatch: {market_data.iloc[0]['time_only']} != {ts.time()}"
    
    print(f"Loaded {len(market_data)} rows of market data")
    
    # Dictionary to store predictions by model type
    all_predictions = {}
    
    # Make predictions with each model type
    for model_type in model_types:
        print(f"\nTesting model type: {model_type}")
        
        try:
            # Initialize predictor with this model type
            predictor = RealTimePredictor(
                model_dir=model_dir,
                confidence_threshold=0.6,
                model_type=model_type
            )
            
            # Make predictions for each asset
            asset_predictions = {}
            
            for asset in assets:
                if asset in predictor.models:
                    try:
                        # Predict on the last 5 rows to see how the model performs on recent data
                        recent_data = market_data.iloc[-5:]
                        prediction, uncertainty, confidence, signal = predictor.predict(recent_data, asset)
                        
                        asset_predictions[asset] = {
                            'prediction': prediction,
                            'uncertainty': uncertainty,
                            'confidence': confidence,
                            'signal': signal
                        }
                        
                        print(f"  {asset.upper()}: Signal={signal}, Prediction={prediction:.6f}, "
                             f"Confidence={confidence:.4f}, Uncertainty={uncertainty:.6f}")
                    except Exception as e:
                        print(f"  Error predicting {asset} with {model_type}: {e}")
                else:
                    print(f"  {asset.upper()}: No model available")
            
            all_predictions[model_type] = asset_predictions
            
        except Exception as e:
            print(f"Error with model type {model_type}: {e}")
    
    # Compare predictions across model types
    print("\n===== Model Comparison Summary =====")
    for asset in assets:
        print(f"\nAsset: {asset.upper()}")
        
        # Create a table for this asset
        comparison_data = []
        for model_type in model_types:
            if model_type in all_predictions and asset in all_predictions[model_type]:
                pred = all_predictions[model_type][asset]
                comparison_data.append([
                    model_type,
                    f"{pred['prediction']:.6f}",
                    f"{pred['confidence']:.4f}",
                    pred['signal']
                ])
        
        # Print comparison table
        if comparison_data:
            print(f"{'Model Type':20} {'Prediction':12} {'Confidence':10} {'Signal':6}")
            print("-" * 60)
            for row in comparison_data:
                print(f"{row[0]:20} {row[1]:12} {row[2]:10} {row[3]:6}")
        else:
            print("No predictions available for this asset.")
    
    # Create and save comparative plots
    plot_dir = os.path.join(model_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for asset in assets:
        plt.figure(figsize=(12, 6))
        
        for model_type in model_types:
            if model_type in all_predictions and asset in all_predictions[model_type]:
                pred = all_predictions[model_type][asset]['prediction']
                plt.bar(model_type, pred, label=model_type)
        
        plt.title(f'Model Prediction Comparison for {asset.upper()}')
        plt.xlabel('Model Type')
        plt.ylabel('Predicted Return')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(plot_dir, f'{asset}_model_comparison.png'))
        plt.close()
    
    print(f"\nComparison plots saved to {plot_dir}")

def test_cpp_bridge(data_file, model_dir, assets, model_type):
    """
    Test the C++ bridge with different model types
    
    Args:
        data_file: CSV file with market data
        model_dir: Directory with trained models
        assets: List of assets to make predictions for
        model_type: Model type to use for prediction
    """
    print(f"\n===== Testing C++ Bridge with {model_type} =====")
    
    # Output file for predictions
    output_file = os.path.join(model_dir, f'{model_type}_predictions.csv')
    
    # Command to run the C++ bridge
    cmd = [
        'python3',
        'python_ml/cpp_integration.py',
        '--mode', 'predict',
        '--input', data_file,
        '--output', output_file,
        '--model-dir', model_dir,
        '--model-type', model_type
    ]
    
    if assets:
        cmd.extend(['--assets'] + assets)
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Command output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check if output file was created
        if os.path.exists(output_file):
            try:
                predictions = pd.read_csv(output_file)
                print(f"\nPredictions from {model_type} model:")
                print(predictions)
            except Exception as e:
                print(f"Error reading predictions: {e}")
        else:
            print(f"Warning: Output file {output_file} was not created")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ bridge: {e}")
        print("Command output:")
        print(e.stdout)
        print("Error output:")
        print(e.stderr)

def main():
    """Main function to test advanced models"""
    parser = argparse.ArgumentParser(description='Test Advanced ML Models for Trading')
    parser.add_argument('--data-dir', type=str, default='data/ml_data',
                      help='Directory with preprocessed data files')
    parser.add_argument('--model-dir', type=str, default='python_ml/models',
                      help='Directory to save/load models')
    parser.add_argument('--assets', type=str, nargs='+', default=['btc', 'eth'],
                      help='Assets to test (default: btc, eth)')
    parser.add_argument('--mode', choices=['train', 'predict', 'bridge', 'all'], 
                      default='all', help='Test mode')
    
    args = parser.parse_args()
    
    # Check requirements first
    if not check_requirements():
        print("Warning: Missing some required libraries")
    
    # Determine available model types
    if ADVANCED_MODELS_AVAILABLE:
        model_types = ['ensemble', 'adaboost', 'xgboost', 'rnn', 'cnn', 'pinn', 'hybrid_stacked']
    else:
        model_types = ['ensemble', 'adaboost', 'xgboost']
        print("Note: Only basic models will be tested (advanced models not available)")
    
    # Run the requested mode
    if args.mode in ['train', 'all']:
        train_and_test_advanced_models(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            assets=args.assets,
            model_types=model_types,
            test_size=0.2
        )
    
    if args.mode in ['predict', 'all']:
        # Use the first asset's data file for prediction testing
        asset = args.assets[0] if args.assets else 'btc'
        data_file = os.path.join(args.data_dir, f'processed_data_{asset}.csv')
        
        compare_model_predictions(
            data_file=data_file,
            model_dir=args.model_dir,
            assets=args.assets,
            model_types=model_types
        )
    
    if args.mode in ['bridge', 'all']:
        # Test C++ bridge with each model type
        asset = args.assets[0] if args.assets else 'btc'
        data_file = os.path.join(args.data_dir, f'processed_data_{asset}.csv')
        
        for model_type in model_types:
            test_cpp_bridge(
                data_file=data_file,
                model_dir=args.model_dir,
                assets=args.assets,
                model_type=model_type
            )
    
    print("\nAdvanced model testing complete!")

if __name__ == '__main__':
    main() 