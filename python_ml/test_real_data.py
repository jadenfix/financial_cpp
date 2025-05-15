#!/usr/bin/env python3
"""
Test script for the Bayesian ML Trading System using real data from ml_data directory.
This script demonstrates training and prediction with the real data files.
"""

import os
import sys
import pandas as pd
from datetime import datetime
import argparse

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from python_ml.real_data_trainer import RealDataTrainer
from python_ml.real_time_predictor import RealTimePredictor

def train_models(data_dir, model_dir, assets, test_size=0.2, no_tuning=False):
    """
    Train models using real data
    
    Args:
        data_dir: Directory with preprocessed data files
        model_dir: Directory to save trained models
        assets: List of assets to train models for
        test_size: Proportion of data to use for testing
        no_tuning: Whether to skip hyperparameter tuning
    """
    print(f"\n===== Training Models with Real Data =====")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Assets: {assets}")
    
    # Create trainer
    trainer = RealDataTrainer(
        data_dir=data_dir,
        model_dir=model_dir,
        test_size=test_size
    )
    
    # Load data
    trainer.load_data(assets=assets)
    
    # Train models
    trainer.train_all_models(
        assets=assets,
        hyperparameter_tuning=not no_tuning
    )
    
    print("\n===== Training Results =====")
    for asset, result in trainer.results.items():
        print(f"{asset.upper()}: MSE={result['mse']:.8f}, R²={result['r2']:.4f}")
        print(f"Features used: {len(result['feature_cols'])}")

def make_predictions(data_file, model_dir, assets, confidence=0.6):
    """
    Make predictions using trained models
    
    Args:
        data_file: CSV file with market data
        model_dir: Directory with trained models
        assets: List of assets to make predictions for
        confidence: Confidence threshold for trade signals
    """
    print(f"\n===== Making Predictions with Real Data =====")
    print(f"Data file: {data_file}")
    print(f"Model directory: {model_dir}")
    print(f"Assets: {assets}")
    
    # Create predictor
    predictor = RealTimePredictor(
        model_dir=model_dir,
        confidence_threshold=confidence
    )
    
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
    
    # Make predictions for the last 5 rows of data
    last_rows = min(5, len(market_data))
    recent_data = market_data.iloc[-last_rows:]
    
    print(f"\nMaking predictions for the most recent {last_rows} data points:")
    
    for i, (idx, row) in enumerate(recent_data.iterrows()):
        print(f"\nData point {i+1} - {idx}:")
        
        # Create a dataframe with just this row
        row_df = pd.DataFrame([row])
        
        # Set index if timestamp was present
        if isinstance(idx, datetime):
            row_df.index = [idx]
        
        # Predict for each asset
        for asset in assets:
            try:
                prediction, uncertainty, confidence, signal = predictor.predict(row_df, asset)
                print(f"  {asset.upper()}: Signal={signal}, Prediction={prediction:.6f}, "
                      f"Confidence={confidence:.4f}, Uncertainty={uncertainty:.6f}")
            except Exception as e:
                print(f"  {asset.upper()}: Error - {str(e)}")
    
    # Also make prediction with all data at once to test batch prediction
    print("\nBatch prediction for all assets:")
    results = predictor.predict_multiple(market_data, assets)
    
    for asset, result in results.items():
        if 'error' in result:
            print(f"  {asset.upper()}: {result['error']}")
        else:
            print(f"  {asset.upper()}: Signal={result['signal']}, "
                  f"Prediction={result['prediction']:.6f}, "
                  f"Confidence={result['confidence']:.4f}")
    
    # Save predictions to CSV
    predictor.save_predictions_to_csv('test_predictions.csv')
    print("\nPredictions saved to test_predictions.csv")

def main():
    """Main function to run the test script"""
    parser = argparse.ArgumentParser(description='Test Bayesian ML Trading System with real data')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='both',
                      help='Mode to run: train models, make predictions, or both')
    parser.add_argument('--data-dir', type=str, default='data/ml_data',
                      help='Directory with preprocessed data files')
    parser.add_argument('--model-dir', type=str, default='python_ml/models',
                      help='Directory to save/load models')
    parser.add_argument('--assets', type=str, nargs='+', default=['btc', 'eth', 'ada', 'solana'],
                      help='Assets to process')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--no-tuning', action='store_true',
                      help='Skip hyperparameter tuning')
    parser.add_argument('--confidence', type=float, default=0.6,
                      help='Confidence threshold for trade signals')
    
    args = parser.parse_args()
    
    # Run the requested mode
    if args.mode in ['train', 'both']:
        train_models(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            assets=args.assets,
            test_size=args.test_size,
            no_tuning=args.no_tuning
        )
    
    if args.mode in ['predict', 'both']:
        # For prediction we'll use the btc data file
        data_file = os.path.join(args.data_dir, 'processed_data_btc.csv')
        make_predictions(
            data_file=data_file,
            model_dir=args.model_dir,
            assets=args.assets,
            confidence=args.confidence
        )
    
    print("\nTest completed successfully!")

if __name__ == '__main__':
    main() 