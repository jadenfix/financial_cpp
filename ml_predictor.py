#!/usr/bin/env python3
"""
ML Predictor Wrapper

This is a simple wrapper script that calls the actual implementation
in python_ml/ml_predictor.py with the same arguments.

This exists to simplify the interface and avoid path issues.
"""

import os
import sys
import subprocess

def main():
    # Get the absolute path to the actual script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    actual_script = os.path.join(script_dir, "python_ml", "ml_predictor.py")
    
    # Check if the script exists
    if not os.path.exists(actual_script):
        print(f"Error: Could not find {actual_script}", file=sys.stderr)
        return 1
    
    # Pass all arguments to the actual script
    cmd = [sys.executable, actual_script] + sys.argv[1:]
    
    try:
        # Run the actual script and pass through its output
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running predictor: {e}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 