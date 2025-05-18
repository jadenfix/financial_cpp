#!/usr/bin/env python3
"""
C++ Call Wrapper for Bayesian ML Trading System

This is a simplified wrapper script that can be called directly from C++ code.
It provides a simplified interface designed for easy integration with the C++ trading engine.

Usage:
    python3 cpp_call_wrapper.py predict <data_file> <output_file> [--asset=btc] [--model=ensemble]
    python3 cpp_call_wrapper.py update <data_file> [--asset=btc] [--model=ensemble]
    python3 cpp_call_wrapper.py update_and_predict <data_file> <output_file> [--asset=btc] [--model=ensemble]
"""

import sys
import os
import argparse
import subprocess
import pandas as pd
from datetime import datetime

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def predict(data_file, output_file, asset="btc", model_type="ensemble", confidence=0.6):
    """
    Make a prediction with the ML model
    
    Args:
        data_file: Path to input data file
        output_file: Path to output file
        asset: Asset to predict for
        model_type: Type of model to use
        confidence: Confidence threshold
    
    Returns:
        0 on success, 1 on error
    """
    log_message(f"Making prediction for {asset} using {model_type} model")
    
    try:
        # Validate input file
        if not os.path.exists(data_file):
            log_message(f"Error: Input file not found: {data_file}")
            return 1
        
        # Call the cpp_integration.py script
        cmd = [
            "python3",
            os.path.join(os.path.dirname(__file__), "cpp_integration.py"),
            "--mode", "predict",
            "--input", data_file,
            "--output", output_file,
            "--assets", asset,
            "--model-type", model_type,
            "--confidence", str(confidence)
        ]
        
        log_message(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log_message(f"Error running prediction: {result.stderr}")
            return 1
        
        # Check if output file was created
        if not os.path.exists(output_file):
            log_message(f"Error: Output file was not created: {output_file}")
            return 1
        
        # Load and print the results
        try:
            results = pd.read_csv(output_file)
            signal = results['signal'].iloc[0] if not results.empty else "UNKNOWN"
            log_message(f"Prediction successful. Signal: {signal}")
        except Exception as e:
            log_message(f"Warning: Couldn't read output file: {e}")
        
        return 0
        
    except Exception as e:
        log_message(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return 1

def update(data_file, asset="btc", model_type="ensemble"):
    """
    Update the ML model with new data
    
    Args:
        data_file: Path to input data file
        asset: Asset to update model for
        model_type: Type of model to update
    
    Returns:
        0 on success, 1 on error
    """
    log_message(f"Updating model for {asset} of type {model_type}")
    
    try:
        # Validate input file
        if not os.path.exists(data_file):
            log_message(f"Error: Input file not found: {data_file}")
            return 1
        
        # Call the cpp_integration.py script
        cmd = [
            "python3",
            os.path.join(os.path.dirname(__file__), "cpp_integration.py"),
            "--mode", "update",
            "--input", data_file,
            "--assets", asset
        ]
        
        # Add model types if not ensemble (which includes all basic models)
        if model_type != "ensemble":
            cmd.extend(["--model-types", model_type])
        
        log_message(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log_message(f"Error updating model: {result.stderr}")
            return 1
        
        log_message(f"Model update successful")
        return 0
        
    except Exception as e:
        log_message(f"Error in update: {e}")
        import traceback
        traceback.print_exc()
        return 1

def update_and_predict(data_file, output_file, asset="btc", model_type="ensemble", confidence=0.6):
    """
    Update the ML model with new data and then make a prediction
    
    Args:
        data_file: Path to input data file
        output_file: Path to output file
        asset: Asset to process
        model_type: Type of model to use
        confidence: Confidence threshold
    
    Returns:
        0 on success, 1 on error
    """
    log_message(f"Updating and predicting for {asset} using {model_type} model")
    
    try:
        # Validate input file
        if not os.path.exists(data_file):
            log_message(f"Error: Input file not found: {data_file}")
            return 1
        
        # Call the cpp_integration.py script
        cmd = [
            "python3",
            os.path.join(os.path.dirname(__file__), "cpp_integration.py"),
            "--mode", "update_and_predict",
            "--input", data_file,
            "--output", output_file,
            "--assets", asset,
            "--model-type", model_type,
            "--confidence", str(confidence)
        ]
        
        log_message(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log_message(f"Error in update_and_predict: {result.stderr}")
            return 1
        
        # Check if output file was created
        if not os.path.exists(output_file):
            log_message(f"Error: Output file was not created: {output_file}")
            return 1
        
        # Load and print the results
        try:
            results = pd.read_csv(output_file)
            signal = results['signal'].iloc[0] if not results.empty else "UNKNOWN"
            log_message(f"Update and prediction successful. Signal: {signal}")
        except Exception as e:
            log_message(f"Warning: Couldn't read output file: {e}")
        
        return 0
        
    except Exception as e:
        log_message(f"Error in update_and_predict: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main function to parse arguments and call appropriate function"""
    # Simple argument parsing
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    
    command = sys.argv[1]
    data_file = sys.argv[2]
    
    # Parse arguments
    asset = "btc"
    model_type = "ensemble"
    confidence = 0.6
    
    # Extract named arguments
    for arg in sys.argv[3:]:
        if arg.startswith("--asset="):
            asset = arg.split("=")[1]
        elif arg.startswith("--model="):
            model_type = arg.split("=")[1]
        elif arg.startswith("--confidence="):
            confidence = float(arg.split("=")[1])
    
    # Execute command
    if command == "predict":
        if len(sys.argv) < 4:
            print("Error: Missing output file")
            print(__doc__)
            return 1
        output_file = sys.argv[3]
        return predict(data_file, output_file, asset, model_type, confidence)
    
    elif command == "update":
        return update(data_file, asset, model_type)
    
    elif command == "update_and_predict":
        if len(sys.argv) < 4:
            print("Error: Missing output file")
            print(__doc__)
            return 1
        output_file = sys.argv[3]
        return update_and_predict(data_file, output_file, asset, model_type, confidence)
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 