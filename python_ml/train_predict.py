import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Import our custom modules
from bayesian_ml_engine import BayesianMLEngine
from feature_engineering import FeatureEngineer
from online_ml_trading import OnlineTradingML

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'update'], required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--confidence', type=float, default=0.6)
    parser.add_argument('--symbol', default=None)
    args = parser.parse_args()

    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Initialize the trading ML system
    trading_ml = OnlineTradingML(model_dir=MODEL_PATH, confidence_threshold=args.confidence)
    
    try:
        # Load market data
        market_data = trading_ml.load_market_data(args.input)
        
        if not market_data:
            print(f"No valid market data found in {args.input}")
            return 1
        
        # Process all symbols or just the specified one
        symbols_to_process = [args.symbol] if args.symbol else market_data.keys()
        
        results = []
        
        for symbol in symbols_to_process:
            if symbol not in market_data:
                print(f"Symbol {symbol} not found in input data")
                continue
            
            print(f"Processing {symbol}...")
            
            # Process data based on mode
            if args.mode == 'train':
                result = trading_ml.process_new_data(symbol, market_data[symbol])
            elif args.mode == 'predict':
                # Update data but don't retrain
                trading_ml.update_market_data(symbol, market_data[symbol])
                latest_data = trading_ml.market_data_buffer[symbol].iloc[[-1]]
                features = trading_ml.feature_engineer.extract_features(latest_data, symbol)
                features = trading_ml.feature_engineer.clean_features(features)
                
                pred, uncertainty = trading_ml.ml_engine.predict_with_uncertainty(features)
                confidence = 1.0 / (1.0 + uncertainty[0])
                
                signal = "FLAT"
                if pred[0] > 0 and confidence > trading_ml.confidence_threshold:
                    signal = "LONG"
                elif pred[0] < 0 and confidence > trading_ml.confidence_threshold:
                    signal = "SHORT"
                
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "prediction": float(pred[0]),
                    "uncertainty": float(uncertainty[0]),
                    "confidence": float(confidence),
                    "signal": signal
                }
                
                trading_ml.trade_signals[symbol] = signal
                
            elif args.mode == 'update':
                # Update the model with new data
                result = trading_ml.process_new_data(symbol, market_data[symbol])
            
            if 'error' not in result:
                results.append(result)
                print(f"  Signal: {result['signal']}, Prediction: {result['prediction']:.6f}, Confidence: {result['confidence']:.4f}")
            else:
                print(f"  Error: {result['error']}")
        
        # Save results to output file
        if results:
            pd.DataFrame(results).to_csv(args.output, index=False)
            print(f"Saved {len(results)} predictions to {args.output}")
        
        # Save state
        trading_ml.save_state()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
