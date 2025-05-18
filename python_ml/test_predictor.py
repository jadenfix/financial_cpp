#!/usr/bin/env python3
"""
Test script for the ML predictor

This script tests the ML predictor with sample data
to verify it works correctly without C++ integration.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from ml_predictor import MLPredictor

def main():
    """
Test the ML predictor
    """
    print("\nML Predictor Test")
    print("================")
    
    # Create sample data
    sample_data = {
        "btc_open": 40000.0,
        "btc_high": 41000.0,
        "btc_low": 39500.0,
        "btc_close": 40500.0,
        "btc_volume": 1000.0
    }
    
    # Create predictor
    predictor = MLPredictor(model_dir="models_test", asset="btc")
    
    # Make prediction
    prediction, uncertainty = predictor.predict(sample_data)
    
    # Print result
    print(f"\nInput data: {sample_data}")
    print(f"Prediction: {prediction}")
    print(f"Uncertainty: {uncertainty}")
    
    # Test with additional features
    sample_data_extended = sample_data.copy()
    sample_data_extended.update({
        "btc_rsi_14": 55.5,
        "btc_sma_20": 39800.0,
        "btc_ema_20": 39750.0
    })
    
    # Make prediction
    prediction2, uncertainty2 = predictor.predict(sample_data_extended)
    
    # Print result
    print(f"\nInput data with additional features: {len(sample_data_extended)} features")
    print(f"Prediction: {prediction2}")
    print(f"Uncertainty: {uncertainty2}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 