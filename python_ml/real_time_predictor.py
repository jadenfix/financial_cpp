#!/usr/bin/env python3
"""
Real-time Predictor for Bayesian ML Trading System

This module:
1. Loads trained models for each asset
2. Makes predictions on new market data as it arrives
3. Provides uncertainty estimates with each prediction
4. Outputs trading signals based on predictions and confidence levels
5. Supports advanced models (RNNs, CNNs, AdaBoost, XGBoost, etc.)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
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

class RealTimePredictor:
    """
    Real-time predictor for making trading decisions using Bayesian ML models
    """
    
    def __init__(self, model_dir='python_ml/models', confidence_threshold=0.6, model_type='ensemble'):
        """
        Initialize the real-time predictor
        
        Args:
            model_dir: Directory containing trained models
            confidence_threshold: Minimum confidence level for actionable predictions
            model_type: Type of model to use ('ensemble', 'rnn', 'cnn', 'adaboost', 'xgboost', 
                       'pinn', 'hybrid_stacked')
        """
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        
        # Check if advanced models are available if requested
        if model_type not in ['ensemble', 'adaboost', 'xgboost'] and not ADVANCED_MODELS_AVAILABLE:
            print(f"Warning: {model_type} model type requested but advanced models not available. Falling back to ensemble.")
            self.model_type = 'ensemble'
        
        # Asset settings
        self.supported_assets = ['btc', 'eth', 'ada', 'solana']
        self.asset_prefixes = {
            'btc': 'btc_',
            'eth': 'eth_',
            'ada': 'ada_',
            'solana': 'solana_'
        }
        
        # Target column format pattern
        self.target_pattern = '{asset}_r_next'
        
        # Initialize dictionaries for models and feature lists
        self.models = {}
        self.feature_lists = {}
        
        # Load models for all available assets
        self._load_models()
        
        # Prediction history
        self.prediction_history = {asset: [] for asset in self.supported_assets}
        
        # Latest signals
        self.current_signals = {asset: "FLAT" for asset in self.supported_assets}
        
        print(f"Real-time predictor initialized with confidence threshold: {confidence_threshold}")
        print(f"Using model type: {self.model_type}")
    
    def _load_models(self):
        """Load trained models for all available assets"""
        for asset in self.supported_assets:
            asset_dir = os.path.join(self.model_dir, asset)
            if os.path.exists(asset_dir):
                try:
                    # Load ML engine
                    self.models[asset] = BayesianMLEngine(model_dir=asset_dir, use_advanced_models=True)
                    
                    # Load feature list
                    feature_file = os.path.join(asset_dir, 'feature_cols.txt')
                    if os.path.exists(feature_file):
                        with open(feature_file, 'r') as f:
                            self.feature_lists[asset] = [line.strip() for line in f]
                    else:
                        # Try the joblib format as fallback
                        feature_file_joblib = os.path.join(asset_dir, 'feature_cols.joblib')
                        if os.path.exists(feature_file_joblib):
                            self.feature_lists[asset] = joblib.load(feature_file_joblib)
                        else:
                            print(f"Warning: No feature list found for {asset}")
                    
                    print(f"Loaded model for {asset} with {len(self.feature_lists.get(asset, []))} features")
                except Exception as e:
                    print(f"Error loading model for {asset}: {e}")
            else:
                print(f"No model directory found for {asset}")
    
    def prepare_features(self, market_data, asset):
        """
        Prepare features for prediction
        
        Args:
            market_data: DataFrame with market data
            asset: Asset to prepare features for
            
        Returns:
            DataFrame with features ready for prediction
        """
        if asset not in self.supported_assets:
            raise ValueError(f"Unsupported asset: {asset}")
        
        if asset not in self.feature_lists:
            raise ValueError(f"No feature list available for {asset}")
        
        # Check if we have the required columns in the data
        required_cols = self.feature_lists[asset]
        missing_cols = [col for col in required_cols if col not in market_data.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns for {asset}: {missing_cols[:5]}...")
            # Add missing columns with zeros
            for col in missing_cols:
                market_data[col] = 0
        
        # Extract required features in the correct order
        X = market_data[required_cols].copy()
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        return X
    
    def predict(self, market_data, asset):
        """
        Make a prediction for the given asset
        
        Args:
            market_data: DataFrame with market data
            asset: Asset to make prediction for
            
        Returns:
            prediction: Predicted return
            uncertainty: Uncertainty estimate
            confidence: Confidence level (1 / (1 + uncertainty))
            signal: Trading signal (LONG, SHORT, FLAT)
        """
        if asset not in self.models:
            raise ValueError(f"No model loaded for {asset}")
        
        # Prepare features
        X = self.prepare_features(market_data, asset)
        
        # Get predictions based on model type
        if self.model_type == 'ensemble':
            # Use all models in ensemble
            prediction, uncertainty = self.models[asset].predict_with_uncertainty(X)
        elif self.model_type in ['adaboost', 'xgboost']:
            # Use the specific boosting model
            model_name = self.model_type
            prediction, _ = self.models[asset].predict(X, models_to_use=[model_name])
            
            # For boosting models, use a fixed uncertainty value since they don't provide it
            uncertainty = np.array([0.2])  # Medium uncertainty
        elif ADVANCED_MODELS_AVAILABLE:
            # Use advanced models if available
            if self.model_type == 'rnn':
                model_names = ['rnn_lstm', 'rnn_gru']
                prediction, _ = self.models[asset].predict(X, models_to_use=model_names)
                uncertainty = np.array([0.3])  # Slightly higher uncertainty
            
            elif self.model_type == 'cnn':
                prediction, _ = self.models[asset].predict(X, models_to_use=['cnn'])
                uncertainty = np.array([0.3])
            
            elif self.model_type == 'pinn':
                prediction, _ = self.models[asset].predict(X, models_to_use=['pinn'])
                uncertainty = np.array([0.25])
            
            elif self.model_type == 'hybrid_stacked':
                prediction, _ = self.models[asset].predict(X, models_to_use=['hybrid_stacked'])
                uncertainty = np.array([0.15])  # Lower uncertainty (more confident)
            
            else:
                # Fallback to ensemble if model type not recognized
                prediction, uncertainty = self.models[asset].predict_with_uncertainty(X)
        else:
            # Fallback to ensemble if advanced models not available
            prediction, uncertainty = self.models[asset].predict_with_uncertainty(X)
        
        # Calculate confidence
        confidence = 1.0 / (1.0 + uncertainty[0])
        
        # Determine signal based on prediction and confidence
        signal = "FLAT"
        if prediction[0] > 0 and confidence > self.confidence_threshold:
            signal = "LONG"
        elif prediction[0] < 0 and confidence > self.confidence_threshold:
            signal = "SHORT"
        
        # Store result
        result = {
            "timestamp": datetime.now().isoformat(),
            "asset": asset,
            "prediction": float(prediction[0]),
            "uncertainty": float(uncertainty[0]),
            "confidence": float(confidence),
            "signal": signal,
            "model_type": self.model_type
        }
        
        # Update history and current signal
        self.prediction_history[asset].append(result)
        self.current_signals[asset] = signal
        
        return prediction[0], uncertainty[0], confidence, signal
    
    def predict_multiple(self, market_data, assets=None):
        """
        Make predictions for multiple assets
        
        Args:
            market_data: DataFrame with market data
            assets: List of assets to predict for, or None for all available
            
        Returns:
            Dictionary of prediction results by asset
        """
        if assets is None:
            assets = [asset for asset in self.models.keys()]
        
        results = {}
        
        for asset in assets:
            if asset in self.models:
                try:
                    prediction, uncertainty, confidence, signal = self.predict(market_data, asset)
                    results[asset] = {
                        "prediction": prediction,
                        "uncertainty": uncertainty,
                        "confidence": confidence,
                        "signal": signal
                    }
                except Exception as e:
                    results[asset] = {"error": str(e)}
            else:
                results[asset] = {"error": "No model loaded for this asset"}
        
        return results
    
    def get_current_signal(self, asset):
        """Get the current trading signal for an asset"""
        return self.current_signals.get(asset, "FLAT")
    
    def get_signal_with_position_size(self, asset, max_position=1.0):
        """
        Get trading signal with suggested position size based on confidence
        
        Args:
            asset: Asset to get signal for
            max_position: Maximum position size (1.0 = 100%)
            
        Returns:
            signal: Trading signal (LONG, SHORT, FLAT)
            position_size: Suggested position size as a fraction of max_position
        """
        if asset not in self.prediction_history or not self.prediction_history[asset]:
            return "FLAT", 0.0
        
        # Get latest prediction
        latest = self.prediction_history[asset][-1]
        
        # Base position size on confidence
        # Normalize between threshold and 1.0
        position_size = 0.0
        
        if latest["confidence"] > self.confidence_threshold:
            # Scale position size based on confidence
            normalized_confidence = (latest["confidence"] - self.confidence_threshold) / (1.0 - self.confidence_threshold)
            position_size = max_position * min(normalized_confidence, 1.0)
        
        return latest["signal"], position_size
    
    def update_model_online(self, market_data, asset, model_types=None):
        """
        Update the model with new data in an online fashion
        
        Args:
            market_data: DataFrame with market data including target variable
            asset: Asset to update model for
            model_types: List of model types to update or None for all
            
        Returns:
            bool: Whether the model was updated
        """
        if asset not in self.models:
            return False
        
        # Get target column
        target_col = self.target_pattern.format(asset=asset)
        
        if target_col not in market_data.columns:
            print(f"Warning: Target column {target_col} not found for online update")
            return False
        
        # Prepare features
        X = self.prepare_features(market_data, asset)
        y = market_data[target_col]
        
        # Ensure we have valid data
        valid_idx = ~np.isnan(y.values)
        if not any(valid_idx):
            print("No valid target values for online update")
            return False
        
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        # Update the model
        updated = self.models[asset].update_online(X_valid, y_valid)
        
        if updated:
            print(f"Model for {asset} updated online with {len(X_valid)} samples")
        
        return updated
    
    def save_predictions_to_csv(self, output_file='predictions.csv'):
        """Save all predictions to a CSV file"""
        all_predictions = []
        
        for asset, predictions in self.prediction_history.items():
            all_predictions.extend(predictions)
        
        if all_predictions:
            pd.DataFrame(all_predictions).to_csv(output_file, index=False)
            print(f"Saved {len(all_predictions)} predictions to {output_file}")
    
    def format_prediction_response(self, asset, prediction, uncertainty, confidence, signal):
        """
        Format a prediction result for API response
        
        Args:
            asset: Asset name
            prediction: Predicted return
            uncertainty: Uncertainty estimate
            confidence: Confidence level
            signal: Trading signal
            
        Returns:
            dict: Formatted prediction result
        """
        _, position_size = self.get_signal_with_position_size(asset)
        
        return {
            "asset": asset,
            "prediction": round(float(prediction), 6),
            "uncertainty": round(float(uncertainty), 6),
            "confidence": round(float(confidence), 4),
            "signal": signal,
            "position_size": round(float(position_size), 4),
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Command-line interface for real-time prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time prediction from Bayesian ML models')
    parser.add_argument('--data-file', type=str, required=True, 
                      help='CSV file with market data to predict on')
    parser.add_argument('--model-dir', type=str, default='python_ml/models',
                      help='Directory containing trained models')
    parser.add_argument('--assets', type=str, nargs='+',
                      default=['btc', 'eth', 'ada', 'solana'],
                      help='Assets to make predictions for')
    parser.add_argument('--confidence', type=float, default=0.6,
                      help='Confidence threshold for actionable predictions')
    parser.add_argument('--output', type=str, default='predictions.csv',
                      help='Output file for prediction results')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                      help='Output format')
    parser.add_argument('--model-type', 
                      choices=['ensemble', 'rnn', 'cnn', 'adaboost', 'xgboost', 
                             'pinn', 'hybrid_stacked'], 
                      default='ensemble',
                      help='Model type to use for prediction')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RealTimePredictor(
        model_dir=args.model_dir,
        confidence_threshold=args.confidence,
        model_type=args.model_type
    )
    
    # Load market data
    try:
        print(f"Loading market data from {args.data_file}...")
        market_data = pd.read_csv(args.data_file)
        
        # Convert timestamp to datetime if present
        if 'timestamp' in market_data.columns:
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            market_data['date_only'] = market_data['timestamp'].dt.date
            market_data['time_only'] = market_data['timestamp'].dt.time
            market_data.set_index('timestamp', inplace=True)
        
        print(f"Loaded {len(market_data)} rows of market data")
    except Exception as e:
        print(f"Error loading market data: {e}")
        return 1
    
    # Make predictions
    results = predictor.predict_multiple(market_data, args.assets)
    
    # Print results
    print("\n===== Prediction Results =====")
    for asset, result in results.items():
        if 'error' in result:
            print(f"{asset}: {result['error']}")
        else:
            print(f"{asset}: Prediction={result['prediction']:.6f}, "
                 f"Confidence={result['confidence']:.4f}, Signal={result['signal']}")
    
    # Save predictions
    predictor.save_predictions_to_csv(args.output)
    
    # Also save as JSON if requested
    if args.format == 'json':
        json_output = args.output.replace('.csv', '.json')
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nPredictions saved to {json_output}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 