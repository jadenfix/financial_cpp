#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import argparse
import json
import time
from datetime import datetime
import joblib
from pathlib import Path

# Import our custom modules
from bayesian_ml_engine import BayesianMLEngine
from feature_engineering import FeatureEngineer

class OnlineTradingML:
    """
    Main class for online trading ML with Bayesian updating.
    This class coordinates feature extraction, model training/updating,
    and prediction for real-time trading signals.
    """
    
    def __init__(self, model_dir="models", buffer_size=1000, confidence_threshold=0.6):
        """Initialize the Online Trading ML system"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.ml_engine = {}  # Dictionary of ML engines by asset
        
        # Settings
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        
        # Asset settings
        self.supported_assets = ['btc', 'eth', 'ada', 'solana']
        self.asset_prefixes = {
            'btc': 'btc_',
            'eth': 'eth_',
            'ada': 'ada_',
            'solana': 'solana_'
        }
        
        # Target column format
        self.target_pattern = '{asset}_r_next'
        
        # Data buffers for each asset
        self.data_buffers = {asset: None for asset in self.supported_assets}
        self.feature_buffers = {asset: None for asset in self.supported_assets}
        self.prediction_buffers = {asset: [] for asset in self.supported_assets}
        
        # Tracking
        self.last_update_time = {asset: None for asset in self.supported_assets}
        self.trade_signals = {asset: "FLAT" for asset in self.supported_assets}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for each asset"""
        for asset in self.supported_assets:
            asset_model_dir = os.path.join(self.model_dir, asset)
            if os.path.exists(asset_model_dir):
                print(f"Loading existing model for {asset}...")
                self.ml_engine[asset] = BayesianMLEngine(model_dir=asset_model_dir)
            else:
                print(f"No existing model found for {asset}. Will initialize when data is available.")
    
    def load_market_data(self, input_file, asset=None):
        """
        Load market data from CSV file
        Expected format: columns with btc_*, eth_*, ada_*, solana_* prefixes
        
        Args:
            input_file: Path to CSV file
            asset: Optional specific asset to load (defaults to all)
            
        Returns:
            Dictionary of DataFrames by asset
        """
        try:
            print(f"Loading data from {input_file}...")
            df = pd.read_csv(input_file)
            
            # Convert timestamp to datetime if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date_only'] = df['timestamp'].dt.date
                df['time_only'] = df['timestamp'].dt.time
                df.set_index('timestamp', inplace=True)
            
            # If we only want a specific asset
            if asset is not None and asset in self.supported_assets:
                assets_to_load = [asset]
            else:
                assets_to_load = self.supported_assets
            
            # Store in buffers
            for asset_name in assets_to_load:
                prefix = self.asset_prefixes[asset_name]
                # Check if columns exist for this asset
                asset_cols = [col for col in df.columns if col.startswith(prefix)]
                if not asset_cols:
                    print(f"Warning: No columns found for {asset_name} in the data file")
                    continue
                
                # Store in buffer
                self.data_buffers[asset_name] = df
                self.last_update_time[asset_name] = datetime.now()
                print(f"Loaded data for {asset_name}: {len(df)} rows")
            
            return self.data_buffers
            
        except Exception as e:
            print(f"Error loading market data: {e}")
            return {}
    
    def update_market_data(self, new_data, asset=None):
        """
        Update market data buffer with new data
        
        Args:
            new_data: DataFrame with new market data
            asset: Optional specific asset to update (defaults to all)
        """
        if asset is not None and asset in self.supported_assets:
            assets_to_update = [asset]
        else:
            assets_to_update = self.supported_assets
        
        # Update each asset's buffer
        for asset_name in assets_to_update:
            if self.data_buffers[asset_name] is None:
                self.data_buffers[asset_name] = new_data
            else:
                # Append new data, avoiding duplicates
                existing_data = self.data_buffers[asset_name]
                combined_data = pd.concat([existing_data, new_data])
                self.data_buffers[asset_name] = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Keep only the most recent buffer_size rows
            if len(self.data_buffers[asset_name]) > self.buffer_size:
                self.data_buffers[asset_name] = self.data_buffers[asset_name].iloc[-self.buffer_size:]
            
            # Update timestamp
            self.last_update_time[asset_name] = datetime.now()
    
    def prepare_features_for_asset(self, asset):
        """
        Prepare features for a specific asset from the current data buffer
        
        Args:
            asset: Asset to prepare features for
            
        Returns:
            DataFrame with features
        """
        if asset not in self.supported_assets:
            raise ValueError(f"Unsupported asset: {asset}")
        
        if self.data_buffers[asset] is None:
            raise ValueError(f"No data available for {asset}")
        
        # Get data
        df = self.data_buffers[asset]
        
        # Extract asset-specific columns
        prefix = self.asset_prefixes[asset]
        feature_cols = [col for col in df.columns if col.startswith(prefix) and col != self.target_pattern.format(asset=asset)]
        
        # Also add other assets' close and return columns as features
        other_assets = [a for a in self.supported_assets if a != asset and self.data_buffers[a] is not None]
        for other_asset in other_assets:
            other_prefix = self.asset_prefixes[other_asset]
            # Add close and return features from other assets
            for feature in ['close', 'r', 'r_ma1h', 'r_vol1h', 'r_ma1d', 'r_vol1d']:
                col_pattern = f'{other_prefix}{feature}'
                matching_cols = [col for col in df.columns if col.startswith(col_pattern) 
                               and not col.endswith('_next')]  # Exclude target columns
                feature_cols.extend(matching_cols)
        
        # Add lagged columns
        lag_cols = [col for col in df.columns if 'lag' in col]
        feature_cols.extend(lag_cols)
        
        # Remove duplicates and sort
        feature_cols = sorted(list(set(feature_cols)))
        
        # Create feature dataframe
        X = df[feature_cols].copy()
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        # Store in feature buffer
        self.feature_buffers[asset] = X
        
        return X
    
    def process_new_data(self, asset, new_data=None):
        """
        Process new market data for a specific asset: extract features, update model, generate predictions
        
        Args:
            asset: The asset to process
            new_data: Optional new data to add first
        
        Returns:
            dict: Prediction results with signal and confidence
        """
        if asset not in self.supported_assets:
            return {"error": f"Unsupported asset: {asset}"}
        
        if new_data is not None:
            self.update_market_data(new_data, asset)
        
        if self.data_buffers[asset] is None:
            return {"error": f"No market data available for {asset}"}
        
        # Prepare features
        try:
            X = self.prepare_features_for_asset(asset)
            
            # Get target if available (for training/updating)
            target_col = self.target_pattern.format(asset=asset)
            y = None
            if target_col in self.data_buffers[asset].columns:
                y = self.data_buffers[asset][target_col]
            
            # Initialize ML engine if needed
            if asset not in self.ml_engine:
                asset_model_dir = os.path.join(self.model_dir, asset)
                os.makedirs(asset_model_dir, exist_ok=True)
                self.ml_engine[asset] = BayesianMLEngine(model_dir=asset_model_dir)
            
            # If we have target data, train or update the model
            if y is not None:
                # Remove NaN targets
                valid_mask = ~y.isna()
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                if len(X_valid) > 0:
                    # If this is first time or we have enough new data, do full training
                    if not hasattr(self.ml_engine[asset].models['bayesian_ridge'], 'coef_') or len(X_valid) > 100:
                        print(f"Training model for {asset} with {len(X_valid)} samples")
                        self.ml_engine[asset].train(X_valid, y_valid, save=True)
                    else:
                        # Online update with latest data point
                        latest_X = X_valid.iloc[-1:]
                        latest_y = y_valid.iloc[-1]
                        print(f"Updating model for {asset} with new data point")
                        self.ml_engine[asset].update_online(latest_X, latest_y)
            
            # Make prediction for the latest data point
            latest_features = X.iloc[[-1]]
            prediction, uncertainty = self.ml_engine[asset].predict_with_uncertainty(latest_features)
            
            # Calculate confidence based on uncertainty
            confidence = 1.0 / (1.0 + uncertainty[0])
            
            # Determine signal based on prediction and confidence
            signal = "FLAT"
            if prediction[0] > 0 and confidence > self.confidence_threshold:
                signal = "LONG"
            elif prediction[0] < 0 and confidence > self.confidence_threshold:
                signal = "SHORT"
            
            # Store prediction
            pred_result = {
                "timestamp": datetime.now().isoformat(),
                "asset": asset,
                "prediction": float(prediction[0]),
                "uncertainty": float(uncertainty[0]),
                "confidence": float(confidence),
                "signal": signal
            }
            
            self.prediction_buffers[asset].append(pred_result)
            self.trade_signals[asset] = signal
            
            return pred_result
            
        except Exception as e:
            return {"error": f"Error processing data for {asset}: {str(e)}"}
    
    def process_all_assets(self, new_data=None):
        """
        Process data for all available assets
        
        Args:
            new_data: Optional new data to add first
        
        Returns:
            dict: Prediction results for all assets
        """
        results = {}
        
        for asset in self.supported_assets:
            if self.data_buffers[asset] is not None:
                result = self.process_new_data(asset, new_data)
                results[asset] = result
        
        return results
    
    def get_latest_prediction(self, asset):
        """Get the latest prediction for an asset"""
        if asset in self.prediction_buffers and self.prediction_buffers[asset]:
            return self.prediction_buffers[asset][-1]
        return None
    
    def get_trade_signal(self, asset):
        """Get the current trade signal for an asset"""
        return self.trade_signals.get(asset, "FLAT")
    
    def save_state(self):
        """Save the current state of all models"""
        for asset, engine in self.ml_engine.items():
            engine._save_models()
        
        print("Saved model state for all assets")
    
    def load_state(self):
        """Load saved state for all models"""
        for asset in self.supported_assets:
            asset_model_dir = os.path.join(self.model_dir, asset)
            if os.path.exists(asset_model_dir):
                self.ml_engine[asset] = BayesianMLEngine(model_dir=asset_model_dir)
                
        print("Loaded model state for all assets")

def main():
    """Main function to run the online trading ML system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Online ML Trading System')
    parser.add_argument('--data-file', type=str, required=True,
                      help='CSV file with market data')
    parser.add_argument('--model-dir', type=str, default='python_ml/models',
                      help='Directory to save/load models')
    parser.add_argument('--assets', type=str, nargs='+', 
                      default=['btc', 'eth', 'ada', 'solana'],
                      help='Assets to process')
    parser.add_argument('--confidence', type=float, default=0.6,
                      help='Confidence threshold for trade signals')
    parser.add_argument('--output', type=str, default='predictions.csv',
                      help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Initialize system
    ml_system = OnlineTradingML(
        model_dir=args.model_dir,
        confidence_threshold=args.confidence
    )
    
    # Load data
    ml_system.load_market_data(args.data_file)
    
    # Process data for all assets
    results = ml_system.process_all_assets()
    
    # Print results
    print("\n===== Prediction Results =====")
    for asset, result in results.items():
        if 'error' in result:
            print(f"{asset}: {result['error']}")
        else:
            print(f"{asset}: Prediction={result['prediction']:.6f}, "
                 f"Confidence={result['confidence']:.4f}, Signal={result['signal']}")
    
    # Save predictions to CSV
    predictions = []
    for asset in args.assets:
        pred = ml_system.get_latest_prediction(asset)
        if pred:
            predictions.append(pred)
    
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")
    
    # Save model state
    ml_system.save_state()

if __name__ == '__main__':
    main() 