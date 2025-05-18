#!/usr/bin/env python3
"""
Train All Models Script for Bayesian ML Trading System

This script:
1. Trains all model types for all assets
2. Performs hyperparameter tuning for key models
3. Evaluates and compares performance 
4. Saves trained models for real-time prediction

Supported model types:
- Ensemble (bayesian_ridge, gradient_boosting, random_forest, gaussian_process)
- AdaBoost
- XGBoost 
- RNNs (LSTM, GRU)
- CNNs
- Physics-Informed Neural Networks (PINNs)
- Hybrid Stacked model (LightGBM + CNN-LSTM + FinBERT-BiLSTM)
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess

# Add python_ml to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_ml.real_data_trainer import RealDataTrainer

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        # Basic dependencies
        import sklearn
        import numpy
        import pandas
        import matplotlib
        
        # Check for TensorFlow (for neural networks)
        advanced_available = False
        try:
            import tensorflow
            print("TensorFlow is installed. Neural network models will be available.")
            advanced_available = True
        except ImportError:
            print("TensorFlow is not installed. Neural network models will not be available.")
            print("To install: pip install tensorflow")
        
        # Check for XGBoost
        try:
            import xgboost
            print("XGBoost is installed.")
        except ImportError:
            print("XGBoost is not installed. XGBoost models will not be available.")
            print("To install: pip install xgboost")
        
        # Check for LightGBM (for hybrid models)
        try:
            import lightgbm
            print("LightGBM is installed.")
        except ImportError:
            print("LightGBM is not installed. Hybrid stacked models will have limited functionality.")
            print("To install: pip install lightgbm")
        
        # Basic dependencies are present
        print("Basic dependencies are installed.")
        
        return advanced_available
    
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required dependencies before running this script.")
        return False

def train_all_models(data_dir, model_dir, assets, tune=True, memory_efficient=True, 
                    cv_splits=5, extended_tuning=False, robust_cv=True):
    """
    Train all model types for all assets
    
    Args:
        data_dir: Directory with preprocessed data files
        model_dir: Directory to save trained models
        assets: List of assets to train models for
        tune: Whether to perform hyperparameter tuning
        memory_efficient: Whether to use memory-efficient mode (one asset at a time)
        cv_splits: Number of cross-validation splits for time series
        extended_tuning: Whether to use extended hyperparameter grids
        robust_cv: Whether to use more robust time series CV with purging and embargoes
    """
    print(f"\n===== Starting Training of All Models =====")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Assets: {assets}")
    print(f"Hyperparameter tuning: {'Enabled' if tune else 'Disabled'}")
    print(f"Extended tuning: {'Enabled' if extended_tuning else 'Disabled'}")
    print(f"Robust time series CV: {'Enabled' if robust_cv else 'Disabled'}")
    print(f"CV splits: {cv_splits}")
    print(f"Memory-efficient mode: {'Enabled' if memory_efficient else 'Disabled'}")
    
    start_time = time.time()
    
    # Check if advanced models are available
    advanced_available = check_dependencies()
    
    # Define model types to train
    model_types = ['ensemble', 'adaboost', 'xgboost']
    if advanced_available:
        model_types.extend(['rnn', 'cnn', 'pinn', 'hybrid_stacked'])
    
    print(f"Models to be trained: {model_types}")
    
    all_results = {}
    
    if memory_efficient:
        # Process one asset at a time to save memory
        for asset in assets:
            print(f"\n===== Processing {asset.upper()} =====")
            
            # Create trainer with robust time series settings
            trainer = RealDataTrainer(
                data_dir=data_dir,
                model_dir=model_dir,
                test_size=0.2,
                cv_splits=cv_splits,
                model_types=model_types,
                use_robust_cv=robust_cv
            )
            
            # Load only this asset's data
            trainer.load_data(assets=[asset])
            
            # Use full dataset instead of limiting to 5000 rows
            if robust_cv:
                print(f"Using full dataset for {asset} with robust time series validation.")
            else:
                # For less robust approaches, still limit data if needed
                if len(trainer.datasets[asset]) > 10000:
                    trainer.datasets[asset] = trainer.datasets[asset].iloc[-10000:]
                    print(f"Using last 10000 rows for {asset} for efficiency.")
            
            # Train models for this asset
            trainer.train_all_models(
                assets=[asset], 
                hyperparameter_tuning=tune,
                extended_tuning=extended_tuning
            )
            
            # Store results
            all_results.update(trainer.results)
            
            # Clean up to free memory
            del trainer
            import gc
            gc.collect()
    else:
        # Process all assets at once (original behavior)
        # Create trainer
        trainer = RealDataTrainer(
            data_dir=data_dir,
            model_dir=model_dir,
            test_size=0.2,
            cv_splits=cv_splits,
            model_types=model_types,
            use_robust_cv=robust_cv
        )
        
        # Load data
        trainer.load_data(assets=assets)
        
        # Train models with extended tuning if specified
        trainer.train_all_models(
            assets=assets, 
            hyperparameter_tuning=tune,
            extended_tuning=extended_tuning
        )
        
        all_results = trainer.results
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n===== Training Complete =====")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Print results for each asset and model type
    print("\n===== Model Performance =====")
    
    # Prepare data for comparison chart
    performance_data = {}
    
    for asset, result in all_results.items():
        print(f"\n{asset.upper()}: MSE={result['mse']:.8f}, R²={result['r2']:.4f}")
        
        # Store ensemble results
        if asset not in performance_data:
            performance_data[asset] = {}
        performance_data[asset]['ensemble'] = result['r2']
        
        # Print results for individual model types
        if 'model_results' in result:
            for model_type, metrics in result['model_results'].items():
                if model_type != 'ensemble':  # Already printed above
                    print(f"  {model_type}: MSE={metrics['mse']:.8f}, R²={metrics['r2']:.4f}")
                    performance_data[asset][model_type] = metrics['r2']
    
    # Create comparison charts
    plot_dir = os.path.join(model_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for asset in assets:
        if asset in performance_data:
            plt.figure(figsize=(12, 6))
            
            model_types = list(performance_data[asset].keys())
            r2_values = [performance_data[asset][model] for model in model_types]
            
            plt.bar(model_types, r2_values)
            plt.title(f'Model Performance (R²) for {asset.upper()}')
            plt.xlabel('Model Type')
            plt.ylabel('R² Score')
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(plot_dir, f'{asset}_model_comparison.png'))
            plt.close()
    
    print(f"\nComparison charts saved to {plot_dir}")
    
    return all_results

def main():
    """Main function to train all models"""
    parser = argparse.ArgumentParser(description='Train all ML models for trading')
    parser.add_argument('--data-dir', type=str, default='data/ml_data',
                       help='Directory with preprocessed data files')
    parser.add_argument('--model-dir', type=str, default='python_ml/models',
                       help='Directory to save trained models')
    parser.add_argument('--assets', type=str, nargs='+', default=['btc', 'eth', 'ada', 'solana'],
                       help='Assets to train models for')
    parser.add_argument('--no-tune', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--test-after', action='store_true',
                       help='Run tests after training to verify models')
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Use memory-efficient mode (one asset at a time)')
    parser.add_argument('--cv-splits', type=int, default=5,
                       help='Number of cross-validation splits for time series validation')
    parser.add_argument('--extended-tuning', action='store_true',
                       help='Use extended hyperparameter grids for more thorough tuning')
    parser.add_argument('--robust-cv', action='store_true',
                       help='Use robust time series CV with purging and embargoes')
    
    args = parser.parse_args()
    
    # Train all models
    results = train_all_models(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        assets=args.assets,
        tune=not args.no_tune,
        memory_efficient=args.memory_efficient,
        cv_splits=args.cv_splits,
        extended_tuning=args.extended_tuning,
        robust_cv=args.robust_cv
    )
    
    # Optionally run tests to verify the models
    if args.test_after:
        print("\n===== Running Tests to Verify Models =====")
        
        # Run the test script
        cmd = [
            'python3',
            'python_ml/test_advanced_models.py',
            '--data-dir', args.data_dir,
            '--model-dir', args.model_dir,
            '--assets'] + args.assets + ['--mode', 'predict']
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    print("\nTraining and testing complete!")

if __name__ == '__main__':
    main() 