#!/usr/bin/env python3
"""
Bayesian ML Engine for Crypto Trading

This module provides a BayesianMLEngine class that:
1. Loads trained ML models (Bayesian Ridge, XGBoost, etc.)
2. Makes predictions with uncertainty estimates
3. Combines models into an ensemble prediction
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import glob
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import BayesianRidge

# Try to import keras, but don't fail if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow/Keras not available. Neural network models will be skipped.")

class BayesianMLEngine:
    """Bayesian ML Engine for prediction with uncertainty"""
    
    def __init__(self, model_dir='python_ml/models/btc'):
        """
        Initialize the ML engine
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.weights = {}
        self.scaler = None
        self.feature_cols = None
        
    def load(self):
        """
        Load trained models from disk
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        print(f"Loading models from {self.model_dir}...")
        
        # Check if directory exists
        if not os.path.exists(self.model_dir):
            print(f"Error: Model directory not found: {self.model_dir}")
            return False
        
        # Load feature columns
        try:
            feature_file = os.path.join(self.model_dir, 'feature_cols.txt')
            if os.path.exists(feature_file):
                with open(feature_file, 'r') as f:
                    self.feature_cols = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded {len(self.feature_cols)} feature columns")
            else:
                print(f"Warning: feature_cols.txt not found in {self.model_dir}")
                return False
        except Exception as e:
            print(f"Error loading feature columns: {e}")
            return False
        
        # Load scaler
        try:
            scaler_file = os.path.join(self.model_dir, 'scaler.joblib')
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
                print(f"Loaded scaler from {scaler_file}")
            else:
                print(f"Warning: Scaler not found at {scaler_file}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            # Continue without scaler - not critical
        
        # Find and load all models
        try:
            # Load scikit-learn models (joblib files)
            for model_file in glob.glob(os.path.join(self.model_dir, '*.joblib')):
                model_name = os.path.basename(model_file).split('.')[0]
                try:
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    print(f"Loaded {model_name} model from {model_file}")
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}")
            
            # Load Keras models if available
            if KERAS_AVAILABLE:
                for model_file in glob.glob(os.path.join(self.model_dir, '*.keras')):
                    model_name = os.path.basename(model_file).split('.')[0]
                    try:
                        model = keras.models.load_model(model_file)
                        self.models[model_name] = model
                        print(f"Loaded {model_name} model from {model_file}")
                    except Exception as e:
                        print(f"Error loading {model_name} model: {e}")
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        
        # Load model metrics to determine weights
        try:
            metrics_file = os.path.join(self.model_dir, 'metrics.json')
            if os.path.exists(metrics_file):
                import json
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Use inverse RMSE as weight
                for model_name, model_metrics in metrics.items():
                    if model_name in self.models:
                        rmse = model_metrics.get('rmse', 1.0)
                        # Avoid division by zero
                        if rmse > 0:
                            self.weights[model_name] = 1.0 / rmse
                        else:
                            self.weights[model_name] = 1.0
                
                # Normalize weights
                if self.weights:
                    total_weight = sum(self.weights.values())
                    if total_weight > 0:
                        for model_name in self.weights:
                            self.weights[model_name] /= total_weight
                    
                    print(f"Model weights based on performance:")
                    for model_name, weight in self.weights.items():
                        print(f"  {model_name}: {weight:.4f}")
                else:
                    # If no weights, use equal weighting
                    for model_name in self.models:
                        self.weights[model_name] = 1.0 / len(self.models)
            else:
                # If no metrics file, use equal weighting
                for model_name in self.models:
                    self.weights[model_name] = 1.0 / len(self.models)
                print(f"No metrics file found. Using equal weights for all models.")
        except Exception as e:
            print(f"Error loading model metrics: {e}")
            # Continue with equal weights
            for model_name in self.models:
                self.weights[model_name] = 1.0 / len(self.models)
        
        if not self.models:
            print(f"No models loaded from {self.model_dir}")
            return False
        
        print(f"Successfully loaded {len(self.models)} models")
        return True
    
    def predict(self, X):
        """
        Make prediction using ensemble of models
        
        Args:
            X: DataFrame or array of features
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if not self.models:
            raise ValueError("No models loaded. Call load() first.")
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if self.feature_cols is not None:
                X = pd.DataFrame(X, columns=self.feature_cols)
            else:
                X = pd.DataFrame(X)
        
        # Check if we have the expected features
        if self.feature_cols is not None:
            missing_cols = [col for col in self.feature_cols if col not in X.columns]
            if missing_cols:
                print(f"Warning: Missing {len(missing_cols)} required features")
                # Create missing columns with zeros
                for col in missing_cols:
                    X[col] = 0.0
            
            # Reorder columns to match expected order
            X = X[self.feature_cols]
        
        # Apply scaler if available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions from each model
        predictions = {}
        for name, model in self.models.items():
            try:
                if name == 'neural_network' and KERAS_AVAILABLE:
                    pred = model.predict(X_scaled).flatten()
                elif name in ['bayesian_ridge']:
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                predictions[name] = pred
            except Exception as e:
                print(f"Error predicting with {name} model: {e}")
        
        # Combine predictions using weighted average
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # Weighted average of predictions
        weighted_preds = np.zeros(X.shape[0])
        weight_sum = 0.0
        
        for name, pred in predictions.items():
            if name in self.weights:
                weight = self.weights[name]
                weighted_preds += weight * pred
                weight_sum += weight
        
        # Normalize if weights don't sum to 1
        if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
            weighted_preds /= weight_sum
        
        return weighted_preds
    
    def predict_with_uncertainty(self, X):
        """
        Make prediction with uncertainty estimates
        
        Args:
            X: DataFrame or array of features
            
        Returns:
            tuple: (predictions, uncertainties)
        """
        if not self.models:
            raise ValueError("No models loaded. Call load() first.")
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if self.feature_cols is not None:
                X = pd.DataFrame(X, columns=self.feature_cols)
            else:
                X = pd.DataFrame(X)
        
        # Check if we have the expected features
        if self.feature_cols is not None:
            missing_cols = [col for col in self.feature_cols if col not in X.columns]
            if missing_cols:
                print(f"Warning: Missing {len(missing_cols)} required features")
                # Create missing columns with zeros
                for col in missing_cols:
                    X[col] = 0.0
            
            # Reorder columns to match expected order
            X = X[self.feature_cols]
        
        # Apply scaler if available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions from each model
        all_predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network' and KERAS_AVAILABLE:
                    pred = model.predict(X_scaled).flatten()
                elif name in ['bayesian_ridge']:
                    # For Bayesian Ridge, we can use built-in uncertainty
                    try:
                        pred, std = model.predict(X_scaled, return_std=True)
                        # Store standard deviation for later
                        all_predictions[name + '_std'] = std
                    except (TypeError, ValueError) as e:
                        # Some versions of scikit-learn might not support return_std
                        print(f"Warning: Could not get uncertainty from {name}: {e}")
                        pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                all_predictions[name] = pred
            except Exception as e:
                print(f"Error predicting with {name} model: {e}")
        
        if not all_predictions:
            raise ValueError("No valid predictions from any model")
        
        # Compute ensemble prediction (weighted average)
        predictions = np.zeros(X.shape[0])
        weight_sum = 0.0
        
        for name, pred in all_predictions.items():
            if name in self.weights and not name.endswith('_std'):
                weight = self.weights[name]
                predictions += weight * pred
                weight_sum += weight
        
        # Normalize if weights don't sum to 1
        if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
            predictions /= weight_sum
        
        # Estimate uncertainty
        # Method 1: If we have Bayesian Ridge with std, use that directly
        if 'bayesian_ridge_std' in all_predictions:
            uncertainties = all_predictions['bayesian_ridge_std']
        # Method 2: Use disagreement between models as uncertainty estimate
        elif len(all_predictions) > 1:
            # Get predictions from each model
            model_preds = []
            for name, pred in all_predictions.items():
                if not name.endswith('_std'):
                    model_preds.append(pred)
            
            # Compute standard deviation across models
            if model_preds:
                model_preds = np.array(model_preds)
                uncertainties = np.std(model_preds, axis=0)
            else:
                # Fallback
                uncertainties = np.ones_like(predictions) * 0.01
        else:
            # No uncertainty available, use small default value
            uncertainties = np.ones_like(predictions) * 0.01
        
        return predictions, uncertainties

# Example usage
if __name__ == "__main__":
    # Example data
    X = pd.DataFrame({'feature1': np.random.randn(100), 
                     'feature2': np.random.randn(100)})
    y = pd.Series(X['feature1'] * 0.5 + X['feature2'] * 0.3 + np.random.randn(100) * 0.1)
    
    # Initialize and load
    engine = BayesianMLEngine()
    engine.load()
    
    # Make predictions
    X_test = pd.DataFrame({'feature1': [0.5], 'feature2': [0.3]})
    pred, _ = engine.predict_with_uncertainty(X_test)
    print(f"Prediction: {pred}")
    
    # Online update
    engine.update_online(pd.Series({'feature1': 0.6, 'feature2': 0.4}), 0.5)

    # Hyperparameter tuning
    param_grid = {'alpha_1': [1e-6, 1e-5, 1e-4], 'alpha_2': [1e-6, 1e-5, 1e-4]}
    grid = GridSearchCV(BayesianRidge(), param_grid, cv=5)
    grid.fit(X_scaled, y)
    best_model = grid.best_estimator_

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(best_model, X, y, cv=tscv)
    print("CV scores:", scores) 