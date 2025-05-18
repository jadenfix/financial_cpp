#!/usr/bin/env python3
"""
Mock ML model for testing the C++ to Python integration

This simple module provides a mock implementation of the ML prediction
without requiring actual trained models, which helps test the integration pipeline.
"""

import numpy as np
import pandas as pd
import random
import os
import json
import sys

class MockMLEngine:
    """Mock ML engine that returns simple deterministic predictions"""
    
    def __init__(self, model_dir=None):
        """Initialize the mock engine"""
        self.model_dir = model_dir
        self.loaded = True
        
    def load(self):
        """Pretend to load models"""
        print(f"Mock: Loading models from {self.model_dir}", file=sys.stderr)
        return True
        
    def predict(self, X):
        """Make a simple prediction based on input data"""
        # Get a deterministic but variable prediction based on input values
        if isinstance(X, pd.DataFrame) and len(X) > 0:
            # Use the sum of feature values to generate a prediction between -0.01 and 0.01
            total = X.sum().sum()
            # Normalize to a small range
            prediction = (np.sin(total) * 0.01)
            return np.array([prediction])
        return np.array([0.0])
        
    def predict_with_uncertainty(self, X):
        """Make prediction with uncertainty estimate"""
        prediction = self.predict(X)
        # Generate uncertainty based on absolute value of prediction
        uncertainty = np.abs(prediction) * 0.5 + 0.0002
        return prediction, uncertainty

def create_mock_model_files(model_dir, asset="btc"):
    """Create mock model files in the specified directory"""
    asset_dir = os.path.join(model_dir, asset)
    os.makedirs(asset_dir, exist_ok=True)
    
    # Create a dummy feature columns file
    with open(os.path.join(asset_dir, "feature_cols.txt"), "w") as f:
        f.write("\n".join([
            f"{asset}_open",
            f"{asset}_high",
            f"{asset}_low",
            f"{asset}_close",
            f"{asset}_volume"
        ]))
    
    # Create a dummy model config file
    config = {
        "model_type": "mock",
        "features": [
            f"{asset}_open",
            f"{asset}_high",
            f"{asset}_low",
            f"{asset}_close",
            f"{asset}_volume"
        ],
        "version": "1.0.0",
        "created_at": "2025-05-16T00:00:00Z"
    }
    
    with open(os.path.join(asset_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created mock model files in {asset_dir}")
    return asset_dir

if __name__ == "__main__":
    # If run directly, create mock model files
    import argparse
    
    parser = argparse.ArgumentParser(description="Create mock ML model files")
    parser.add_argument("--model-dir", type=str, default="python_ml/models_test",
                       help="Directory to create mock model files in")
    parser.add_argument("--asset", type=str, default="btc",
                       help="Asset to create mock model for")
    
    args = parser.parse_args()
    
    create_mock_model_files(args.model_dir, args.asset) 