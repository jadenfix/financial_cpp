#!/usr/bin/env python3
"""
Real Data Trainer for Bayesian ML Trading System

This script:
1. Loads preprocessed data from ml_data directory
2. Trains asset-specific models using cross-validation and hyperparameter tuning
3. Evaluates performance on a test set
4. Saves the trained models for real-time predictions
5. Supports advanced models including RNNs, CNNs, AdaBoost, XGBoost, PINNs, and Hybrid stacked models
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add python_ml to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_ml.bayesian_ml_engine import BayesianMLEngine

# Try to import advanced models
try:
    from python_ml.advanced_models import (
        RNNModel, CNNModel, PhysicsInformedNN, HybridStackedModel
    )
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("Advanced models not available. Using basic models only.")

class RealDataTrainer:
    """Trainer for Bayesian ML models on real cryptocurrency data"""
    
    def __init__(self, data_dir='data/ml_data', model_dir='python_ml/models', 
                 test_size=0.2, cv_splits=5, random_state=42, model_types=None,
                 use_robust_cv=False):
        """
        Initialize the trainer
        
        Args:
            data_dir: Directory with preprocessed data files
            model_dir: Directory to save trained models
            test_size: Proportion of data to use for testing
            cv_splits: Number of cross-validation splits
            random_state: Random seed for reproducibility
            model_types: List of model types to train ('ensemble', 'rnn', 'cnn', 
                       'adaboost', 'xgboost', 'pinn', 'hybrid_stacked', or None for all)
            use_robust_cv: Whether to use robust time series CV with purging and embargoes
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.use_robust_cv = use_robust_cv
        
        # Set model types to train
        if model_types is None:
            # Default to ensemble plus advanced models if available
            self.model_types = ['ensemble', 'adaboost', 'xgboost']
            if ADVANCED_MODELS_AVAILABLE:
                self.model_types.extend(['rnn', 'cnn', 'pinn', 'hybrid_stacked'])
        else:
            self.model_types = model_types
        
        print(f"Models to be trained: {self.model_types}")
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize dictionaries for data and models
        self.datasets = {}  # Will hold dataframes by asset
        self.models = {}    # Will hold trained models by asset
        self.results = {}   # Will hold evaluation results
        
        # Target column format pattern (e.g., 'btc_r_next', 'eth_r_next', etc.)
        self.target_pattern = '{asset}_r_next'
        
        # Asset prefixes
        self.asset_prefixes = {
            'btc': 'btc_',
            'eth': 'eth_',
            'ada': 'ada_',
            'solana': 'solana_'
        }
    
    def load_data(self, assets=None):
        """
        Load preprocessed data files for specified assets
        
        Args:
            assets: List of assets to load (e.g., ['btc', 'eth']) or None for all available
        """
        if assets is None:
            # Default to these assets
            assets = ['btc', 'eth', 'ada', 'solana']
        
        print(f"Loading data for assets: {assets}")
        
        for asset in assets:
            file_path = os.path.join(self.data_dir, f'processed_data_{asset}.csv')
            if not os.path.exists(file_path):
                print(f"Warning: File not found for {asset}: {file_path}")
                continue
            
            print(f"Reading data for {asset}...")
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date_only'] = df['timestamp'].dt.date
                df['time_only'] = df['timestamp'].dt.time
                df.set_index('timestamp', inplace=True)
            
            self.datasets[asset] = df
            print(f"Loaded {len(df)} rows for {asset}")
            
            # Print column names to verify
            print(f"Columns for {asset}:")
            print(df.columns.tolist()[:10])  # Print first 10 columns
            
            # Check for target column
            target_col = self.target_pattern.format(asset=asset)
            if target_col not in df.columns:
                print(f"Warning: Target column '{target_col}' not found. Available targets:")
                target_cols = [col for col in df.columns if 'next' in col]
                print(target_cols)
    
    def prepare_features_and_target(self, asset, feature_groups=None):
        """
        Prepare features and target for a specific asset
        
        Args:
            asset: Asset to prepare (e.g., 'btc', 'eth')
            feature_groups: List of feature groups to use or None for default
            
        Returns:
            X: Feature dataframe
            y: Target series
            feature_names: List of feature names
        """
        if asset not in self.datasets:
            raise ValueError(f"Data for {asset} not loaded. Call load_data() first.")
        
        df = self.datasets[asset]
        target_col = self.target_pattern.format(asset=asset)
        
        if target_col not in df.columns:
            # Try to find alternative target column
            possible_targets = [col for col in df.columns if f'{asset}' in col and 'next' in col]
            if possible_targets:
                target_col = possible_targets[0]
                print(f"Using alternative target column: {target_col}")
            else:
                raise ValueError(f"No suitable target column found for {asset}")
        
        # Extract all columns that start with the asset prefix
        asset_prefix = self.asset_prefixes[asset]
        feature_cols = [col for col in df.columns if col.startswith(asset_prefix) and col != target_col]
        
        # Also add other assets' close and return columns as features
        other_assets = [a for a in self.datasets.keys() if a != asset]
        for other_asset in other_assets:
            other_prefix = self.asset_prefixes[other_asset]
            # Add close and return features from other assets
            for feature in ['close', 'r', 'r_ma1h', 'r_vol1h', 'r_ma1d', 'r_vol1d']:
                col_pattern = f'{other_prefix}{feature}'
                matching_cols = [col for col in df.columns if col.startswith(col_pattern) 
                               and not col.endswith('_next')]  # Exclude target columns
                feature_cols.extend(matching_cols)
        
        # Add lagged columns (these are already in the dataset)
        lag_cols = [col for col in df.columns if 'lag' in col]
        feature_cols.extend(lag_cols)
        
        # Remove duplicates and sort
        feature_cols = sorted(list(set(feature_cols)))
        
        # Create feature dataframe and target series
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X.fillna(0, inplace=True)
        y.fillna(0, inplace=True)
        
        print(f"Prepared {len(X)} samples with {len(feature_cols)} features for {asset}")
        print(f"Target column: {target_col}")
        
        return X, y, feature_cols
    
    def train_model_for_asset(self, asset, hyperparameter_tuning=True, extended_tuning=False):
        """
        Train a model for a specific asset with cross-validation and optional hyperparameter tuning
        
        Args:
            asset: Asset to train model for (e.g., 'btc', 'eth')
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            extended_tuning: Whether to use extended hyperparameter grids
            
        Returns:
            Trained model
        """
        print(f"\n===== Training model for {asset.upper()} =====")
        
        # Prepare features and target
        X, y, feature_cols = self.prepare_features_and_target(asset)
        
        # Split data into training and testing sets
        # For time series, always use the last portion as the test set
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
        
        # Initialize the ML engine
        model_path = os.path.join(self.model_dir, asset)
        os.makedirs(model_path, exist_ok=True)
        ml_engine = BayesianMLEngine(model_dir=model_path)
        
        # Get models to train based on model_types
        models_to_train = []
        for model_type in self.model_types:
            if model_type == 'ensemble':
                # For ensemble, train all basic models
                models_to_train.extend(['bayesian_ridge', 'gbm', 'random_forest', 
                                      'online_sgd'])
                # Add gaussian_process separately to handle memory constraints
                if len(X_train) > 10000:
                    print("Dataset is large. Training Gaussian Process on subset to avoid memory issues.")
                    # Create a subsample for GP training
                    subsample_size = min(5000, len(X_train))
                    # Use systematic sampling to get a representative subset
                    step = len(X_train) // subsample_size
                    indices = np.arange(0, len(X_train), step)[:subsample_size]
                    X_train_gp = X_train.iloc[indices]
                    y_train_gp = y_train.iloc[indices]
                    
                    # Train GP separately
                    try:
                        print(f"Training gaussian_process on {len(X_train_gp)} samples...")
                        ml_engine.models['gaussian_process'].fit(X_train_gp, y_train_gp)
                        print("Finished training gaussian_process")
                    except Exception as e:
                        print(f"Error training gaussian_process: {e}")
                else:
                    models_to_train.append('gaussian_process')
            elif model_type == 'adaboost':
                models_to_train.append('adaboost')
            elif model_type == 'xgboost':
                models_to_train.append('xgboost')
            elif model_type == 'rnn' and ADVANCED_MODELS_AVAILABLE:
                models_to_train.extend(['rnn_lstm', 'rnn_gru'])
            elif model_type == 'cnn' and ADVANCED_MODELS_AVAILABLE:
                models_to_train.append('cnn')
            elif model_type == 'pinn' and ADVANCED_MODELS_AVAILABLE:
                models_to_train.append('pinn')
            elif model_type == 'hybrid_stacked' and ADVANCED_MODELS_AVAILABLE:
                models_to_train.append('hybrid_stacked')
        
        # Remove duplicates and 'gaussian_process' if it's already handled separately
        models_to_train = [m for m in set(models_to_train) if m != 'gaussian_process' or 'gaussian_process' not in models_to_train]
        print(f"Training the following models: {models_to_train}")
        
        if hyperparameter_tuning:
            print(f"Performing {'extended' if extended_tuning else 'standard'} hyperparameter tuning...")
            self._tune_hyperparameters(ml_engine, X_train, y_train, models_to_train, extended_tuning)
        
        # Train the model
        print("Training final model...")
        ml_engine.train(X_train, y_train, models_to_train=models_to_train, save=True)
        
        # Evaluate the model
        print("Evaluating on test set...")
        
        # For ensemble prediction (default behavior)
        predictions, uncertainty = ml_engine.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Ensemble Test MSE: {mse:.8f}")
        print(f"Ensemble Test R²: {r2:.4f}")
        
        # Calculate and report uncertainty metrics
        mean_uncertainty = np.mean(uncertainty)
        print(f"Mean prediction uncertainty: {mean_uncertainty:.6f}")
        
        # Evaluate calibration - how well uncertainty predicts actual errors
        abs_errors = np.abs(y_test - predictions)
        calibration_corr = np.corrcoef(uncertainty, abs_errors)[0, 1]
        print(f"Uncertainty calibration correlation: {calibration_corr:.4f}")
        
        # Evaluate individual models if requested
        model_results = {
            'ensemble': {
                'mse': mse, 
                'r2': r2, 
                'mean_uncertainty': mean_uncertainty,
                'calibration_corr': calibration_corr
            }
        }
        
        # Evaluate each model type separately if not using ensemble
        for model_type in [mt for mt in self.model_types if mt != 'ensemble']:
            try:
                if model_type == 'adaboost':
                    model_names = ['adaboost']
                elif model_type == 'xgboost':
                    model_names = ['xgboost']
                elif model_type == 'rnn' and ADVANCED_MODELS_AVAILABLE:
                    model_names = ['rnn_lstm', 'rnn_gru']
                elif model_type == 'cnn' and ADVANCED_MODELS_AVAILABLE:
                    model_names = ['cnn']
                elif model_type == 'pinn' and ADVANCED_MODELS_AVAILABLE:
                    model_names = ['pinn']
                elif model_type == 'hybrid_stacked' and ADVANCED_MODELS_AVAILABLE:
                    model_names = ['hybrid_stacked']
                else:
                    continue
                
                # Print the models that will be used for evaluation
                print(f"Evaluating {model_type} using models: {model_names}")
                
                # Check if all required models are available and valid
                valid_models = []
                for model_name in model_names:
                    if model_name in ml_engine.models:
                        # For advanced models, make sure the inner keras model exists
                        if model_name in ['rnn_lstm', 'rnn_gru', 'cnn', 'pinn', 'hybrid_stacked']:
                            inner_model = ml_engine.models[model_name].named_steps['model']
                            if hasattr(inner_model, 'model') and inner_model.model is not None:
                                valid_models.append(model_name)
                            else:
                                print(f"Model {model_name} exists but has no valid inner model")
                        else:
                            valid_models.append(model_name)
                
                if valid_models:
                    print(f"Valid models for {model_type}: {valid_models}")
                    pred, _ = ml_engine.predict(X_test, models_to_use=valid_models)
                    model_mse = mean_squared_error(y_test, pred)
                    model_r2 = r2_score(y_test, pred)
                    print(f"{model_type} Test MSE: {model_mse:.8f}")
                    print(f"{model_type} Test R²: {model_r2:.4f}")
                    model_results[model_type] = {'mse': model_mse, 'r2': model_r2}
                else:
                    print(f"No valid models available for {model_type}")
                    
            except Exception as e:
                print(f"Error evaluating {model_type}: {e}")
        
        # Plot results
        self._plot_predictions(asset, y_test, predictions, X_test.index)
        
        # Store results
        self.models[asset] = ml_engine
        self.results[asset] = {
            'mse': mse,
            'r2': r2,
            'feature_cols': feature_cols,
            'model_results': model_results
        }
        
        return ml_engine
    
    def _tune_hyperparameters(self, ml_engine, X, y, models_to_tune=None, extended_tuning=False):
        """
        Perform hyperparameter tuning using cross-validation
        
        Args:
            ml_engine: BayesianMLEngine instance
            X: Training features
            y: Target values
            models_to_tune: List of model names to tune or None for all
            extended_tuning: Whether to use extended hyperparameter grids
        """
        # Create appropriate cross-validation strategy
        if self.use_robust_cv:
            print("Using robust time series cross-validation with purging and embargoes")
            tscv = self._create_robust_time_series_cv(X)
        else:
            print(f"Using standard TimeSeriesSplit with {self.cv_splits} splits")
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        # Only tune models that are present in the engine
        if models_to_tune is None:
            models_to_tune = list(ml_engine.models.keys())
        else:
            models_to_tune = [m for m in models_to_tune if m in ml_engine.models]
        
        # Tune XGBoost if available
        if 'xgboost' in models_to_tune:
            try:
                xgb_model = ml_engine.models['xgboost'].named_steps['model']
                
                # Define parameter grid
                if extended_tuning:
                    param_grid = {
                        'n_estimators': [50, 100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                        'max_depth': [3, 4, 5, 6, 7],
                        'min_child_weight': [1, 2, 3],
                        'subsample': [0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                        'gamma': [0, 0.1, 0.2]
                    }
                else:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                
                # Create grid search with cross-validation
                grid_search = GridSearchCV(
                    estimator=xgb_model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                grid_search.fit(X, y)
                
                # Print best parameters
                print(f"Best XGBoost parameters: {grid_search.best_params_}")
                print(f"Best XGBoost MSE: {-grid_search.best_score_:.8f}")
                
                # Update the model with best parameters
                ml_engine.models['xgboost'].named_steps['model'] = grid_search.best_estimator_
            except Exception as e:
                print(f"Error tuning XGBoost: {e}")
        
        # Tune Gradient Boosting if available
        if 'gbm' in models_to_tune:
            try:
                gbm_model = ml_engine.models['gbm'].named_steps['model']
                
                # Define parameter grid for GBM
                if extended_tuning:
                    gbm_param_grid = {
                        'n_estimators': [50, 100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'max_depth': [2, 3, 4, 5],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'subsample': [0.7, 0.8, 0.9, 1.0]
                    }
                else:
                    gbm_param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [2, 3, 4]
                    }
                
                # Create grid search for GBM
                gbm_grid_search = GridSearchCV(
                    estimator=gbm_model,
                    param_grid=gbm_param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                gbm_grid_search.fit(X, y)
                
                # Print best parameters
                print(f"Best GBM parameters: {gbm_grid_search.best_params_}")
                print(f"Best GBM MSE: {-gbm_grid_search.best_score_:.8f}")
                
                # Update the model with best parameters
                ml_engine.models['gbm'].named_steps['model'] = gbm_grid_search.best_estimator_
            except Exception as e:
                print(f"Error tuning GBM: {e}")
        
        # Tune AdaBoost if available
        if 'adaboost' in models_to_tune:
            try:
                adaboost_model = ml_engine.models['adaboost'].named_steps['model']
                
                # Define parameter grid for AdaBoost
                if extended_tuning:
                    ada_param_grid = {
                        'n_estimators': [50, 100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                        'loss': ['linear', 'square', 'exponential']
                    }
                else:
                    ada_param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1.0],
                        'loss': ['linear', 'square', 'exponential']
                    }
                
                # Create grid search for AdaBoost
                ada_grid_search = GridSearchCV(
                    estimator=adaboost_model,
                    param_grid=ada_param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                ada_grid_search.fit(X, y)
                
                # Print best parameters
                print(f"Best AdaBoost parameters: {ada_grid_search.best_params_}")
                print(f"Best AdaBoost MSE: {-ada_grid_search.best_score_:.8f}")
                
                # Update the model with best parameters
                ml_engine.models['adaboost'].named_steps['model'] = ada_grid_search.best_estimator_
            except Exception as e:
                print(f"Error tuning AdaBoost: {e}")
                
        # Tune Random Forest if available
        if 'random_forest' in models_to_tune:
            try:
                rf_model = ml_engine.models['random_forest'].named_steps['model']
                
                # Define parameter grid for Random Forest
                if extended_tuning:
                    rf_param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
                    }
                else:
                    rf_param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 20],
                        'min_samples_split': [2, 5]
                    }
                
                # Create grid search for Random Forest
                rf_grid_search = GridSearchCV(
                    estimator=rf_model,
                    param_grid=rf_param_grid,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                rf_grid_search.fit(X, y)
                
                # Print best parameters
                print(f"Best Random Forest parameters: {rf_grid_search.best_params_}")
                print(f"Best Random Forest MSE: {-rf_grid_search.best_score_:.8f}")
                
                # Update the model with best parameters
                ml_engine.models['random_forest'].named_steps['model'] = rf_grid_search.best_estimator_
            except Exception as e:
                print(f"Error tuning Random Forest: {e}")
    
    def _plot_predictions(self, asset, y_true, y_pred, index):
        """
        Plot true vs predicted values
        
        Args:
            asset: Asset name
            y_true: True values
            y_pred: Predicted values
            index: DataFrame index for the x-axis
        """
        plt.figure(figsize=(12, 6))
        plt.plot(index, y_true, label='True')
        plt.plot(index, y_pred, label='Predicted')
        plt.title(f'Predictions for {asset.upper()}')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_dir = os.path.join(self.model_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{asset}_predictions.png'))
        plt.close()
    
    def train_all_models(self, assets=None, hyperparameter_tuning=True, extended_tuning=False):
        """
        Train models for all specified assets
        
        Args:
            assets: List of assets to train models for or None for all available
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            extended_tuning: Whether to use extended hyperparameter grids
        """
        if assets is None:
            assets = list(self.datasets.keys())
        
        for asset in assets:
            print(f"\n===== Training models for {asset.upper()} with {'extended' if extended_tuning else 'standard'} tuning =====")
            self.train_model_for_asset(asset, hyperparameter_tuning=hyperparameter_tuning, extended_tuning=extended_tuning)
        
        # Save feature information
        self.save_feature_info()
    
    def save_feature_info(self):
        """Save feature information for all trained models"""
        for asset, result in self.results.items():
            feature_file = os.path.join(self.model_dir, asset, 'feature_cols.txt')
            with open(feature_file, 'w') as f:
                f.write('\n'.join(result['feature_cols']))

    def _create_robust_time_series_cv(self, X, embargo_size=5):
        """
        Create a time series cross-validation with purging and embargoes
        
        Args:
            X: Feature dataframe with datetime index
            embargo_size: Number of samples to embargo between train and test sets
            
        Returns:
            A generator of train/test indices for cross-validation
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Determine fold size
        fold_size = n_samples // (self.cv_splits + 1)  # +1 to ensure test sets don't overlap
        
        for i in range(self.cv_splits):
            # Start with a rolling window approach
            test_start = n_samples - (i + 1) * fold_size
            test_end = n_samples - i * fold_size
            
            # Apply embargo: remove samples before test set
            if embargo_size > 0 and test_start > embargo_size:
                train_end = test_start - embargo_size
            else:
                train_end = test_start
            
            # Get train/test indices
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices

def main():
    """Main function to train models using real data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models using real market data')
    parser.add_argument('--data-dir', type=str, default='data/ml_data',
                       help='Directory with preprocessed data files')
    parser.add_argument('--model-dir', type=str, default='python_ml/models',
                       help='Directory to save trained models')
    parser.add_argument('--assets', type=str, nargs='+', default=['btc', 'eth', 'ada', 'solana'],
                       help='Assets to train models for')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--no-hyperparameter-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--model-types', type=str, nargs='+',
                       choices=['ensemble', 'rnn', 'cnn', 'adaboost', 'xgboost', 
                              'pinn', 'hybrid_stacked'],
                       default=['ensemble', 'adaboost', 'xgboost'],
                       help='Model types to train')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RealDataTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        test_size=args.test_size,
        model_types=args.model_types
    )
    
    # Load data
    trainer.load_data(assets=args.assets)
    
    # Train models
    trainer.train_all_models(
        assets=args.assets,
        hyperparameter_tuning=not args.no_hyperparameter_tuning
    )
    
    print("Training complete!")
    for asset, result in trainer.results.items():
        print(f"{asset.upper()}: MSE={result['mse']:.8f}, R²={result['r2']:.4f}")
        
        # Print results for individual model types
        if 'model_results' in result:
            for model_type, metrics in result['model_results'].items():
                if model_type != 'ensemble':  # Already printed above
                    print(f"  {model_type}: MSE={metrics['mse']:.8f}, R²={metrics['r2']:.4f}")

if __name__ == '__main__':
    main() 