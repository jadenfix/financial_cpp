#!/usr/bin/env python3
"""
ML Predictor for C++ Integration

This script:
1. Receives a single row of market data (via stdin or file)
2. Makes predictions using the trained ML models
3. Returns prediction and uncertainty estimate as CSV

For use with the C++ trading engine.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import traceback

# Add python_ml to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the mock model module first
try:
    from python_ml.mock_model import MockMLEngine
    MOCK_MODEL_AVAILABLE = True
except ImportError:
    MOCK_MODEL_AVAILABLE = False
    print(f"Warning: Mock model module not available.", file=sys.stderr)

# Import ML classes
try:
    from python_ml.bayesian_ml_engine import BayesianMLEngine
    ML_ENGINE_AVAILABLE = True
except ImportError:
    ML_ENGINE_AVAILABLE = False
    print(f"Warning: BayesianMLEngine not available. Will use mock model if available.", file=sys.stderr)

# Try to import feature_engineering, but don't fail if it's not available
try:
    from python_ml.feature_engineering import extract_features
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    print(f"Warning: Feature engineering module not available. Using raw features only.", file=sys.stderr)

class MLPredictor:
    """ML predictor for C++ integration"""
    
    def __init__(self, model_dir='python_ml/models', asset='btc', use_mock=False):
        """
        Initialize the predictor
        
        Args:
            model_dir: Base directory for models
            asset: Asset to predict (e.g., 'btc', 'eth')
            use_mock: Whether to force using the mock model
        """
        self.asset = asset
        self.model_dir = os.path.join(model_dir, asset)
        self.ml_engine = None
        self.feature_cols = None
        self.using_mock = False

        # Use mock model if explicitly requested
        if use_mock and MOCK_MODEL_AVAILABLE:
            print(f"Using mock ML engine for {asset} (explicitly requested)", file=sys.stderr)
            self.ml_engine = MockMLEngine(model_dir=self.model_dir)
            self.using_mock = True
        
        # Try loading the real ML engine if not using mock
        elif ML_ENGINE_AVAILABLE and not use_mock:
            try:
                self.ml_engine = BayesianMLEngine(model_dir=self.model_dir)
                success = self.ml_engine.load()
                
                if not success:
                    print(f"Warning: Failed to load models from {self.model_dir}", file=sys.stderr)
                    self.ml_engine = None
            except Exception as e:
                print(f"Error loading ML engine: {e}", file=sys.stderr)
                traceback.print_exc()
                self.ml_engine = None
        
        # If real ML engine didn't load, try using mock as fallback
        if self.ml_engine is None and MOCK_MODEL_AVAILABLE:
            print(f"Using mock ML engine for {asset} (fallback)", file=sys.stderr)
            self.ml_engine = MockMLEngine(model_dir=self.model_dir)
            self.using_mock = True
        
        # Load feature columns
        self.feature_cols = self._load_feature_cols()
    
    def _load_feature_cols(self):
        """Load feature column names from model directory"""
        feature_file = os.path.join(self.model_dir, 'feature_cols.txt')
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # If feature file doesn't exist but we're using mock, create default features
        if self.using_mock:
            return [
                f"{self.asset}_open",
                f"{self.asset}_high",
                f"{self.asset}_low",
                f"{self.asset}_close",
                f"{self.asset}_volume"
            ]
            
        return None
    
    def predict(self, data_row):
        """
        Make prediction for a single data row
        
        Args:
            data_row: Dict or DataFrame row with market data
            
        Returns:
            prediction: Predicted return
            uncertainty: Uncertainty estimate
        """
        # If ML engine is not available, return defaults
        if self.ml_engine is None:
            print(f"Using dummy prediction due to missing ML engine", file=sys.stderr)
            return 0.0, 1.0
            
        # Convert dict to DataFrame if needed
        if isinstance(data_row, dict):
            df = pd.DataFrame([data_row])
        else:
            df = pd.DataFrame([data_row.to_dict()])
            
        # Convert all numeric columns to float64 to avoid dtype warnings
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(np.float64)
        
        # Print available columns for debugging
        print(f"Input data columns: {df.columns.tolist()}", file=sys.stderr)
        
        # Skip feature engineering - we expect all required features to be provided directly
        
        # Select only the columns needed for prediction
        if self.feature_cols is not None:
            # Check for missing required features
            missing_cols = [col for col in self.feature_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing {len(missing_cols)} required features: {missing_cols[:5]}...", file=sys.stderr)
                print(f"Will use zeros for missing features. For better predictions, provide all required features.", file=sys.stderr)
            
            # Create a new DataFrame with all required columns initialized to 0
            X = pd.DataFrame(0, index=df.index, columns=self.feature_cols, dtype=np.float64)
            
            # Copy available values from the original DataFrame
            for col in self.feature_cols:
                if col in df.columns:
                    X.loc[:, col] = df[col].values.astype(np.float64)
            
            print(f"Using features: {self.feature_cols[:5]}... ({len(self.feature_cols)} total)", file=sys.stderr)
        else:
            # Use all numeric columns
            X = df.select_dtypes(include=['float64', 'int64']).astype(np.float64)
            print(f"Using all numeric features: {X.columns.tolist()}", file=sys.stderr)
        
        # Make prediction with uncertainty
        try:
            prediction, uncertainty = self.ml_engine.predict_with_uncertainty(X)
            return prediction[0], uncertainty[0]
        except Exception as e:
            print(f"Error during prediction: {e}. Returning default values.", file=sys.stderr)
            traceback.print_exc()
            return 0.0, 1.0  # Return neutral prediction with high uncertainty

def parse_input():
    """Parse input data from stdin or file"""
    parser = argparse.ArgumentParser(description='ML predictor for C++ integration')
    parser.add_argument('--model-dir', type=str, default='python_ml/models',
                       help='Base directory for trained models')
    parser.add_argument('--asset', type=str, default='btc',
                       help='Asset to predict')
    parser.add_argument('--input-file', type=str,
                       help='Input file with data (if not using stdin)')
    parser.add_argument('--json-input', action='store_true',
                       help='Whether input is in JSON format (default is CSV)')
    parser.add_argument('--use-mock', action='store_true',
                       help='Force using mock model even if real models exist')
    
    args = parser.parse_args()
    
    # Get data from stdin or file
    try:
        if args.input_file:
            print(f"Reading input from file: {args.input_file}", file=sys.stderr)
            if args.json_input:
                try:
                    with open(args.input_file, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from file: {e}", file=sys.stderr)
                    with open(args.input_file, 'r') as f:
                        content = f.read()
                    print(f"File content: {content}", file=sys.stderr)
                    raise
            else:
                data = pd.read_csv(args.input_file).iloc[0].to_dict()
        else:
            print(f"Reading input from stdin", file=sys.stderr)
            if args.json_input:
                stdin_data = sys.stdin.read().strip()
                if not stdin_data:
                    print(f"Warning: Empty stdin data", file=sys.stderr)
                    data = {}
                else:
                    try:
                        data = json.loads(stdin_data)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON from stdin: {e}", file=sys.stderr)
                        print(f"Stdin content: {stdin_data}", file=sys.stderr)
                        raise
            else:
                data = pd.read_csv(sys.stdin).iloc[0].to_dict()
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    return data, args

def main():
    """Main function"""
    # Set up basic logging
    sys.stderr.write(f"ML Predictor starting at {datetime.now().isoformat()}\n")
    
    try:
        # Parse input
        data, args = parse_input()
        
        # Create predictor
        predictor = MLPredictor(
            model_dir=args.model_dir, 
            asset=args.asset,
            use_mock=args.use_mock
        )
        
        # Make prediction
        prediction, uncertainty = predictor.predict(data)
        
        # Print result to stdout (as CSV)
        print(f"{prediction},{uncertainty}")
        
        sys.stderr.write(f"Prediction complete: {prediction}, {uncertainty}\n")
        return 0
    except Exception as e:
        print(f"0.0,1.0")  # Output default values
        sys.stderr.write(f"Error in main: {e}\n")
        traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 