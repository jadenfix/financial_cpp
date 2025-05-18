#!/usr/bin/env python3
import pandas as pd
import sys

def sample_csv(file_path, nrows=5):
    """
    Sample the first few rows of a large CSV file and print info about columns.
    
    Args:
        file_path: Path to the CSV file
        nrows: Number of rows to sample
    """
    print(f"\nSampling file: {file_path}")
    print("-" * 80)
    
    # Read a sample of the file
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        
        # Basic info
        print(f"Number of columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        print("\nSample data (first {nrows} rows):")
        print(df.head(nrows).to_string())
        
        # Data types
        print("\nData types:")
        print(df.dtypes)
        
        # Check for target columns
        target_cols = [col for col in df.columns if "_r_next" in col]
        if target_cols:
            print(f"\nFound target columns: {target_cols}")
        
        # Check timestamp format
        if 'timestamp' in df.columns:
            print("\nTimestamp sample:")
            print(df['timestamp'].head(3).tolist())
    
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # Default to BTC file if no argument provided
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/ml_data/processed_data_btc.csv"
    sample_csv(file_path) 