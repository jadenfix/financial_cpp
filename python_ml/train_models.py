#!/usr/bin/env python3
"""
ML Model Training Script for Trading System

This script:
1. Loads processed data for a specific coin
2. Trains multiple ML models (Bayesian, XGBoost, etc.)
3. Evaluates model performance
4. Saves models in the correct format for the C++ trading engine

Usage:
    python train_models.py --coin btc --data-dir data/ml_data --model-dir python_ml/models
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

# Try to import xgboost, but don't fail if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Skipping XGBoost model.")

# Try to import neural network modules, but don't fail if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow/Keras not available. Skipping neural network models.")

class ModelTrainer:
    """Class to train and evaluate ML models for time series prediction"""
    
    def __init__(self, 
                 coin='btc', 
                 data_dir='data/ml_data', 
                 model_dir='python_ml/models',
                 target_col='target_next_close_btc',
                 test_size=0.2,
                 random_state=42):
        """
        Initialize the model trainer
        
        Args:
            coin: Cryptocurrency symbol (btc, eth, etc.)
            data_dir: Directory containing processed data files
            model_dir: Directory to save trained models
            target_col: Target column for prediction
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.coin = coin
        self.data_dir = data_dir
        self.model_dir = os.path.join(model_dir, coin)
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.timestamp_col = 'timestamp'
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize data
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Initialize feature columns
        self.feature_cols = []
        
        # Initialize models
        self.models = {}
        self.scaler = StandardScaler()
        
        # Metrics
        self.metrics = {}
        
    def load_data(self):
        """Load the processed data file for the specified coin"""
        data_file = os.path.join(self.data_dir, f'processed_data_{self.coin}.csv')
        
        print(f"Loading data from {data_file}...")
        
        # Check if file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load the data
        try:
            self.data = pd.read_csv(data_file)
            print(f"Loaded {len(self.data)} rows with {len(self.data.columns)} columns")
            
            # Basic data info
            print(f"Data columns: {self.data.columns.tolist()[:5]}... (and {len(self.data.columns)-5} more)")
            print(f"Data range: {self.data[self.timestamp_col].min()} to {self.data[self.timestamp_col].max()}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def setup_features(self):
        """Set up feature columns, excluding target and timestamp"""
        # Get all columns except target and timestamp
        all_cols = [col for col in self.data.columns 
                    if col != self.target_col and col != self.timestamp_col]
        
        # Check if we already have feature_cols.txt in the model directory
        feature_cols_file = os.path.join(self.model_dir, 'feature_cols.txt')
        
        if os.path.exists(feature_cols_file):
            print(f"Loading feature columns from {feature_cols_file}...")
            with open(feature_cols_file, 'r') as f:
                self.feature_cols = [line.strip() for line in f.readlines() if line.strip()]
            
            # Ensure all feature columns exist in the data
            missing_cols = [col for col in self.feature_cols if col not in self.data.columns]
            if missing_cols:
                print(f"Warning: {len(missing_cols)} feature columns from feature_cols.txt "
                      f"are not in the data: {missing_cols[:5]}...")
                
            # Use only columns that exist in the data
            self.feature_cols = [col for col in self.feature_cols if col in self.data.columns]
        else:
            print(f"Creating new feature columns list (all columns except target and timestamp)...")
            self.feature_cols = all_cols
            
            # Save feature columns to file
            with open(feature_cols_file, 'w') as f:
                f.write('\n'.join(self.feature_cols))
        
        print(f"Using {len(self.feature_cols)} features")
        print(f"First 5 features: {self.feature_cols[:5]}...")
        
        return True
    
    def prepare_data(self):
        """Prepare data for training by splitting into features and target"""
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return False
        
        if not self.feature_cols:
            print("Feature columns not set up. Please set up features first.")
            return False
        
        print(f"Preparing data for training...")
        
        # Check if target column exists
        if self.target_col not in self.data.columns:
            print(f"Error: Target column '{self.target_col}' not found in data.")
            return False
        
        # Get features and target
        X = self.data[self.feature_cols].copy()
        y = self.data[self.target_col].copy()
        
        # Split data into train and test sets - use time series split
        # We'll use the last test_size portion of the data for testing
        split_idx = int(len(X) * (1 - self.test_size))
        
        # Sort by timestamp first to ensure proper time-based split
        if self.timestamp_col in self.data.columns:
            print("Sorting data by timestamp...")
            sort_idx = self.data[self.timestamp_col].argsort()
            X = X.iloc[sort_idx]
            y = y.iloc[sort_idx]
        
        # Now split
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Testing set: {len(self.X_test)} samples")
        
        # Scale features
        print("Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return True
    
    def train_models(self):
        """Train multiple regression models"""
        if self.X_train is None or self.y_train is None:
            print("Data not prepared. Please prepare data first.")
            return False
        
        print(f"Training models on {len(self.X_train)} samples...")
        
        # Dictionary to store model objects
        models = {}
        
        # 1. Bayesian Ridge Regression
        print("Training Bayesian Ridge Regression...")
        try:
            br = BayesianRidge(compute_score=True, max_iter=300)
            br.fit(self.X_train_scaled, self.y_train)
            models['bayesian_ridge'] = br
            print("Bayesian Ridge Regression trained successfully.")
        except Exception as e:
            print(f"Error training Bayesian Ridge Regression: {e}")
        
        # 2. Gradient Boosting Regressor
        print("Training Gradient Boosting Regressor...")
        try:
            gbr = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
            gbr.fit(self.X_train, self.y_train)  # No need to scale for tree-based models
            models['gradient_boosting'] = gbr
            print("Gradient Boosting Regressor trained successfully.")
        except Exception as e:
            print(f"Error training Gradient Boosting Regressor: {e}")
        
        # 3. Random Forest Regressor
        print("Training Random Forest Regressor...")
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf.fit(self.X_train, self.y_train)  # No need to scale for tree-based models
            models['random_forest'] = rf
            print("Random Forest Regressor trained successfully.")
        except Exception as e:
            print(f"Error training Random Forest Regressor: {e}")
            
        # 4. AdaBoost Regressor
        print("Training AdaBoost Regressor...")
        try:
            ada = AdaBoostRegressor(n_estimators=100, random_state=self.random_state)
            ada.fit(self.X_train, self.y_train)
            models['adaboost'] = ada
            print("AdaBoost Regressor trained successfully.")
        except Exception as e:
            print(f"Error training AdaBoost Regressor: {e}")
        
        # 5. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("Training XGBoost Regressor...")
            try:
                xgb_reg = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    random_state=self.random_state
                )
                xgb_reg.fit(self.X_train, self.y_train)
                models['xgboost'] = xgb_reg
                print("XGBoost Regressor trained successfully.")
            except Exception as e:
                print(f"Error training XGBoost Regressor: {e}")
        
        # 6. Neural Network (if available)
        if KERAS_AVAILABLE:
            print("Training Neural Network...")
            try:
                # Simple feed-forward neural network
                model = keras.Sequential([
                    keras.layers.Dense(64, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(16, activation='relu'),
                    keras.layers.Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mse')
                
                # Early stopping to prevent overfitting
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                # Train model
                model.fit(
                    self.X_train_scaled, self.y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                models['neural_network'] = model
                print("Neural Network trained successfully.")
            except Exception as e:
                print(f"Error training Neural Network: {e}")
        
        self.models = models
        print(f"Trained {len(models)} models successfully.")
        
        return len(models) > 0
    
    def evaluate_models(self):
        """Evaluate model performance on test set"""
        if not self.models:
            print("No models trained. Please train models first.")
            return False
        
        print(f"Evaluating {len(self.models)} models on {len(self.X_test)} test samples...")
        
        # Dictionary to store metrics
        metrics = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            try:
                # Predict on test set
                if name == 'neural_network' and KERAS_AVAILABLE:
                    y_pred = model.predict(self.X_test_scaled).flatten()
                elif name in ['bayesian_ridge']:
                    y_pred = model.predict(self.X_test_scaled)
                else:
                    y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                metrics[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                print(f"  {name} metrics: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        self.metrics = metrics
        
        # Find best model based on RMSE
        best_model = min(metrics.items(), key=lambda x: x[1]['rmse'])
        print(f"\nBest model: {best_model[0]} with RMSE={best_model[1]['rmse']:.6f}")
        
        return True
    
    def save_models(self):
        """Save trained models to disk"""
        if not self.models:
            print("No models trained. Please train models first.")
            return False
        
        print(f"Saving {len(self.models)} models to {self.model_dir}...")
        
        # Create additional directories
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save feature columns
        feature_cols_file = os.path.join(self.model_dir, 'feature_cols.txt')
        with open(feature_cols_file, 'w') as f:
            f.write('\n'.join(self.feature_cols))
        
        # Save scaler
        scaler_file = os.path.join(self.model_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_file)
        
        # Save each model
        for name, model in self.models.items():
            try:
                if name == 'neural_network' and KERAS_AVAILABLE:
                    # Save Keras model in .keras format
                    model_file = os.path.join(self.model_dir, f'{name}.keras')
                    model.save(model_file)
                else:
                    # Save scikit-learn model
                    model_file = os.path.join(self.model_dir, f'{name}.joblib')
                    joblib.dump(model, model_file)
                
                print(f"Saved {name} model to {model_file}")
            except Exception as e:
                print(f"Error saving {name} model: {e}")
        
        # Save metrics
        if self.metrics:
            metrics_file = os.path.join(self.model_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"Saved metrics to {metrics_file}")
        
        # Save model info
        info = {
            'coin': self.coin,
            'target_column': self.target_col,
            'timestamp': datetime.now().isoformat(),
            'num_features': len(self.feature_cols),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'models': list(self.models.keys()),
            'best_model': min(self.metrics.items(), key=lambda x: x[1]['rmse'])[0] if self.metrics else None
        }
        
        info_file = os.path.join(self.model_dir, 'model_info.json')
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Saved model info to {info_file}")
        
        return True
    
    def run_pipeline(self):
        """Run the full training pipeline"""
        print(f"Starting training pipeline for {self.coin}...")
        print(f"Target column: {self.target_col}")
        
        # Load data
        if not self.load_data():
            return False
        
        # Set up features
        if not self.setup_features():
            return False
        
        # Prepare data
        if not self.prepare_data():
            return False
        
        # Train models
        if not self.train_models():
            return False
        
        # Evaluate models
        if not self.evaluate_models():
            return False
        
        # Save models
        if not self.save_models():
            return False
        
        print(f"Training pipeline completed successfully!")
        return True


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Train ML models for crypto trading')
    parser.add_argument('--coin', type=str, default='btc', help='Coin to train models for')
    parser.add_argument('--data-dir', type=str, default='data/ml_data', help='Directory containing processed data files')
    parser.add_argument('--model-dir', type=str, default='python_ml/models', help='Directory to save trained models')
    parser.add_argument('--target-col', type=str, default=None, help='Target column for prediction')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set default target column based on coin
    if args.target_col is None:
        args.target_col = f'target_next_close_{args.coin}'
    
    # Create trainer
    trainer = ModelTrainer(
        coin=args.coin,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Run pipeline
    success = trainer.run_pipeline()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main()) 