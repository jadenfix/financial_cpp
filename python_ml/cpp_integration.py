#!/usr/bin/env python3
"""
C++ Integration Script for Bayesian ML Trading System

This script:
1. Serves as a bridge between the C++ trading engine and Python ML system
2. Receives market data from C++ code through CSV files
3. Makes predictions and returns signals with confidence
4. Handles model updating (training) when requested
5. Takes care of properly formatting data between systems
6. Supports advanced models (RNNs, CNNs, AdaBoost, XGBoost, Hybrid Stacked models, PINNs)
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import json
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("python_ml/ml_bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_bridge")

# Add python_ml to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_ml.real_time_predictor import RealTimePredictor
from python_ml.real_data_trainer import RealDataTrainer

# Try to import the advanced models module
try:
    from python_ml.advanced_models import (
        RNNModel, CNNModel, PhysicsInformedNN, HybridStackedModel
    )
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    logger.warning("Advanced models not available. Using basic models only.")

def process_ohlcv_data(input_file):
    """
    Process OHLCV data from CSV file into format suitable for ML models
    
    The input file is expected to have columns:
    - timestamp (optional)
    - symbol (optional)
    - Either:
      a) Asset-specific columns like btc_open, btc_close, etc.
      b) Generic columns like open, high, low, close, volume along with a symbol column
    
    Returns:
        DataFrame with all data in the correct format
    """
    try:
        # Read the data
        logger.info(f"Reading data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date_only'] = df['timestamp'].dt.date
            df['time_only'] = df['timestamp'].dt.time
            df.set_index('timestamp', inplace=True)
            logger.info(f"Set timestamp as index")
        
        # Check if data already has asset-specific columns (like btc_open)
        asset_prefixes = ['btc_', 'eth_', 'ada_', 'solana_']
        has_asset_columns = any(any(col.startswith(prefix) for col in df.columns) 
                               for prefix in asset_prefixes)
        
        # If data doesn't have asset columns but has symbol column, transform it
        if not has_asset_columns and 'symbol' in df.columns:
            logger.info(f"Transforming data from symbol-based format to asset-specific columns")
            
            # Standard columns to transform
            std_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Find the actual column names (case-insensitive)
            col_mapping = {}
            for std_col in std_cols:
                for col in df.columns:
                    if col.lower() == std_col:
                        col_mapping[std_col] = col
                        break
            
            # Check if we found all needed columns
            missing_cols = [col for col in std_cols if col not in col_mapping]
            if missing_cols:
                logger.warning(f"Missing price columns: {missing_cols}")
            
            # Create a new dataframe with asset-specific columns
            transformed_data = pd.DataFrame(index=df.index if df.index.name == 'timestamp' else df['timestamp'])
            
            # Group by symbol and create asset-specific columns
            for symbol, group in df.groupby('symbol'):
                asset = symbol.lower()
                
                # Only process supported assets
                if not any(asset.startswith(prefix.replace('_', '')) for prefix in asset_prefixes):
                    continue
                
                # Add columns for this asset
                for std_col, actual_col in col_mapping.items():
                    transformed_data[f"{asset}_{std_col}"] = group[actual_col].values
            
            df = transformed_data
            logger.info(f"Transformed data has {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing OHLCV data: {e}")
        traceback.print_exc()
        return None

def make_predictions(input_file, output_file, model_dir='python_ml/models',
                    confidence_threshold=0.6, assets=None, model_type='ensemble'):
    """
    Make predictions using trained models and output signals
    
    Args:
        input_file: Path to CSV file with market data
        output_file: Path to output file for predictions and signals
        model_dir: Directory with trained models
        confidence_threshold: Threshold for confidence to generate signals
        assets: List of assets to predict for, or None for all available
        model_type: Type of model to use for prediction ('ensemble', 'rnn', 'cnn', 
                   'adaboost', 'xgboost', 'pinn', 'hybrid_stacked')
        
    Returns:
        0 on success, 1 on error
    """
    try:
        logger.info(f"Making predictions using data from {input_file}")
        logger.info(f"Using model type: {model_type}")
        
        # Initialize predictor
        predictor = RealTimePredictor(
            model_dir=model_dir,
            confidence_threshold=confidence_threshold,
            model_type=model_type
        )
        
        # Process market data
        market_data = process_ohlcv_data(input_file)
        
        if market_data is None or len(market_data) == 0:
            logger.error(f"No valid market data found in {input_file}")
            return 1
        
        # Default to supported assets if none specified
        if assets is None:
            assets = predictor.supported_assets
        
        # Make predictions
        logger.info(f"Making predictions for assets: {assets}")
        results = predictor.predict_multiple(market_data, assets)
        
        if not results:
            logger.warning("No prediction results generated")
            return 1
        
        # Format results for output
        formatted_results = []
        for asset, result in results.items():
            if 'error' in result:
                logger.warning(f"Error predicting for {asset}: {result['error']}")
                continue
                
            formatted_results.append({
                "asset": asset,
                "prediction": float(result["prediction"]),
                "uncertainty": float(result["uncertainty"]),
                "confidence": float(result["confidence"]),
                "signal": result["signal"],
                "model_type": model_type,
                "timestamp": datetime.now().isoformat()
            })
        
        # Save results
        if formatted_results:
            if output_file.endswith('.csv'):
                pd.DataFrame(formatted_results).to_csv(output_file, index=False)
            elif output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(formatted_results, f, indent=2)
            else:
                # Default to CSV
                pd.DataFrame(formatted_results).to_csv(output_file, index=False)
            
            logger.info(f"Saved {len(formatted_results)} predictions to {output_file}")
        else:
            logger.warning("No valid predictions to save")
        
        # Print prediction summary
        for result in formatted_results:
            logger.info(f"{result['asset'].upper()}: Signal={result['signal']}, "
                      f"Prediction={result['prediction']:.6f}, "
                      f"Confidence={result['confidence']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        traceback.print_exc()
        return 1

def update_models(input_file, model_dir='python_ml/models', 
                 test_size=0.2, assets=None, hyperparameter_tuning=False,
                 model_types=None):
    """
    Update (retrain) models with new data
    
    Args:
        input_file: Path to CSV file with market data for training
        model_dir: Directory to save trained models
        test_size: Proportion of data to use for testing
        assets: List of assets to train models for, or None for all available
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        model_types: List of model types to train ('ensemble', 'rnn', 'cnn', 
                   'adaboost', 'xgboost', 'pinn', 'hybrid_stacked', or None for all)
        
    Returns:
        0 on success, 1 on error
    """
    try:
        logger.info(f"Updating models using data from {input_file}")
        
        if model_types is None:
            # Include basic models plus advanced models if available
            model_types = ['ensemble', 'adaboost', 'xgboost']
            if ADVANCED_MODELS_AVAILABLE:
                model_types.extend(['rnn', 'cnn', 'pinn', 'hybrid_stacked'])
        
        logger.info(f"Model types to train: {model_types}")
        
        # Check if we should use the original ml_data directory instead
        if os.path.isdir(input_file):
            logger.info(f"Input is a directory, using RealDataTrainer directly")
            data_dir = input_file
            
            # Initialize trainer
            trainer = RealDataTrainer(
                data_dir=data_dir,
                model_dir=model_dir,
                test_size=test_size,
                model_types=model_types
            )
            
            # Load data
            trainer.load_data(assets=assets)
            
            # Train models
            trainer.train_all_models(
                assets=assets,
                hyperparameter_tuning=hyperparameter_tuning
            )
            
            # Print results
            for asset, result in trainer.results.items():
                logger.info(f"{asset.upper()}: MSE={result['mse']:.8f}, R²={result['r2']:.4f}")
            
            return 0
        
        # Process the input file
        market_data = process_ohlcv_data(input_file)
        
        if market_data is None or len(market_data) == 0:
            logger.error(f"No valid market data found in {input_file}")
            return 1
        
        # Use online updating through the predictor
        predictor = RealTimePredictor(
            model_dir=model_dir,
            confidence_threshold=0.6  # Default threshold
        )
        
        # Default to supported assets if none specified
        if assets is None:
            assets = predictor.supported_assets
        
        # Update each model
        for asset in assets:
            if asset not in predictor.models:
                logger.warning(f"No model found for {asset}, skipping update")
                continue
            
            logger.info(f"Updating model for {asset}")
            updated = predictor.update_model_online(market_data, asset, model_types=model_types)
            
            if updated:
                logger.info(f"Successfully updated model for {asset}")
            else:
                logger.warning(f"Failed to update model for {asset}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error updating models: {e}")
        traceback.print_exc()
        return 1

def update_and_predict(input_file, output_file, model_dir='python_ml/models',
                     confidence_threshold=0.6, assets=None, model_type='ensemble'):
    """
    Update models with new data and then make predictions
    
    Args:
        input_file: Path to CSV file with market data
        output_file: Path to output file for predictions
        model_dir: Directory for models
        confidence_threshold: Threshold for confidence to generate signals
        assets: List of assets to process, or None for all available
        model_type: Type of model to use ('ensemble', 'rnn', 'cnn', etc.)
        
    Returns:
        0 on success, 1 on error
    """
    try:
        logger.info(f"Updating models and making predictions using data from {input_file}")
        
        # First update the models
        result = update_models(
            input_file=input_file,
            model_dir=model_dir,
            assets=assets,
            hyperparameter_tuning=False,  # Don't do hyperparameter tuning for online updates
            model_types=[model_type] if model_type != 'ensemble' else None
        )
        
        if result != 0:
            logger.warning("Model update failed, continuing with prediction using existing models")
        
        # Then make predictions
        result = make_predictions(
            input_file=input_file,
            output_file=output_file,
            model_dir=model_dir,
            confidence_threshold=confidence_threshold,
            assets=assets,
            model_type=model_type
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in update_and_predict: {e}")
        traceback.print_exc()
        return 1

def compare_models(input_file, output_file, model_dir='python_ml/models',
                 assets=None, confidence_threshold=0.6):
    """
    Compare predictions from different model types
    
    Args:
        input_file: Path to CSV file with market data
        output_file: Path to output file for comparison results
        model_dir: Directory with trained models
        assets: List of assets to compare, or None for all available
        confidence_threshold: Threshold for confidence to generate signals
        
    Returns:
        0 on success, 1 on error
    """
    try:
        logger.info(f"Comparing models using data from {input_file}")
        
        # Define model types to compare
        model_types = ['ensemble', 'adaboost', 'xgboost']
        if ADVANCED_MODELS_AVAILABLE:
            model_types.extend(['rnn', 'cnn', 'pinn', 'hybrid_stacked'])
        
        # Process market data once
        market_data = process_ohlcv_data(input_file)
        
        if market_data is None or len(market_data) == 0:
            logger.error(f"No valid market data found in {input_file}")
            return 1
        
        # Make predictions with each model type
        all_results = {}
        
        for model_type in model_types:
            logger.info(f"Testing model type: {model_type}")
            
            # Initialize predictor with this model type
            try:
                predictor = RealTimePredictor(
                    model_dir=model_dir,
                    confidence_threshold=confidence_threshold,
                    model_type=model_type
                )
                
                # Default to supported assets if none specified
                if assets is None:
                    current_assets = predictor.supported_assets
                else:
                    current_assets = assets
                
                # Make predictions
                results = predictor.predict_multiple(market_data, current_assets)
                
                # Store results
                all_results[model_type] = results
                
            except Exception as e:
                logger.warning(f"Error with model type {model_type}: {e}")
                all_results[model_type] = {"error": str(e)}
        
        # Format comparison results
        comparison = []
        
        # Get all assets across all models
        all_assets = set()
        for model_results in all_results.values():
            if isinstance(model_results, dict):
                all_assets.update(model_results.keys())
        
        # For each asset, compare model predictions
        for asset in all_assets:
            asset_comparison = {"asset": asset}
            
            for model_type, results in all_results.items():
                if isinstance(results, dict) and asset in results:
                    result = results[asset]
                    if 'error' not in result:
                        asset_comparison[f"{model_type}_prediction"] = float(result["prediction"])
                        asset_comparison[f"{model_type}_confidence"] = float(result["confidence"])
                        asset_comparison[f"{model_type}_signal"] = result["signal"]
            
            comparison.append(asset_comparison)
        
        # Save comparison results
        if comparison:
            if output_file.endswith('.csv'):
                pd.DataFrame(comparison).to_csv(output_file, index=False)
            elif output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(comparison, f, indent=2)
            else:
                # Default to CSV
                pd.DataFrame(comparison).to_csv(output_file, index=False)
            
            logger.info(f"Saved model comparison to {output_file}")
        else:
            logger.warning("No comparison results to save")
        
        # Print summary
        for asset_result in comparison:
            logger.info(f"Asset: {asset_result['asset']}")
            for model_type in model_types:
                pred_key = f"{model_type}_prediction"
                sig_key = f"{model_type}_signal"
                conf_key = f"{model_type}_confidence"
                
                if pred_key in asset_result:
                    logger.info(f"  {model_type}: Signal={asset_result[sig_key]}, "
                             f"Prediction={asset_result[pred_key]:.6f}, "
                             f"Confidence={asset_result[conf_key]:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        traceback.print_exc()
        return 1

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="C++ Integration for Bayesian ML Trading System")
    parser.add_argument('--mode', choices=['predict', 'update', 'update_and_predict', 'compare'], 
                      default='predict', help='Operation mode')
    parser.add_argument('--input', required=True, 
                      help='Input file or directory with market data')
    parser.add_argument('--output', 
                      help='Output file for predictions (required for predict modes)')
    parser.add_argument('--model-dir', default='python_ml/models',
                      help='Directory for models')
    parser.add_argument('--assets', nargs='+',
                      help='List of assets to process (default: all available)')
    parser.add_argument('--confidence', type=float, default=0.6,
                      help='Confidence threshold for trade signals (0-1)')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--no-tuning', action='store_true',
                      help='Skip hyperparameter tuning when updating models')
    parser.add_argument('--model-type', 
                      choices=['ensemble', 'rnn', 'cnn', 'adaboost', 'xgboost', 
                             'pinn', 'hybrid_stacked'], 
                      default='ensemble',
                      help='Model type to use for prediction')
    parser.add_argument('--model-types', nargs='+',
                      choices=['ensemble', 'rnn', 'cnn', 'adaboost', 'xgboost', 
                             'pinn', 'hybrid_stacked'],
                      help='Model types to train (for update mode)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['predict', 'update_and_predict', 'compare'] and not args.output:
        logger.error("Output file is required for prediction and comparison modes")
        return 1
    
    # Execute the requested operation
    if args.mode == 'predict':
        return make_predictions(
            input_file=args.input,
            output_file=args.output,
            model_dir=args.model_dir,
            confidence_threshold=args.confidence,
            assets=args.assets,
            model_type=args.model_type
        )
    elif args.mode == 'update':
        return update_models(
            input_file=args.input,
            model_dir=args.model_dir,
            test_size=args.test_size,
            assets=args.assets,
            hyperparameter_tuning=not args.no_tuning,
            model_types=args.model_types
        )
    elif args.mode == 'update_and_predict':
        return update_and_predict(
            input_file=args.input,
            output_file=args.output,
            model_dir=args.model_dir,
            confidence_threshold=args.confidence,
            assets=args.assets,
            model_type=args.model_type
        )
    elif args.mode == 'compare':
        return compare_models(
            input_file=args.input,
            output_file=args.output,
            model_dir=args.model_dir,
            assets=args.assets,
            confidence_threshold=args.confidence
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 