#!/usr/bin/env python3
"""
Test script for the Bayesian ML trading system.
This script creates sample data, trains models, and tests predictions.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from bayesian_ml_engine import BayesianMLEngine
from feature_engineering import FeatureEngineer
from online_ml_trading import OnlineTradingML

def generate_sample_data(n_samples=500, symbol="BTC"):
    """Generate synthetic OHLCV data for testing"""
    # Start date
    start_date = datetime(2023, 1, 1)
    
    # Generate dates at 1-minute intervals
    dates = [start_date + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate price series with random walk + trend + seasonality
    np.random.seed(42)
    
    # Base price and random walk component
    price = 100.0
    prices = [price]
    for _ in range(n_samples - 1):
        # Random walk with drift
        price = price * (1 + np.random.normal(0.0001, 0.001))
        prices.append(price)
    
    prices = np.array(prices)
    
    # Add trend
    trend = np.linspace(0, 0.1, n_samples)
    prices = prices * (1 + trend)
    
    # Add seasonality (hourly pattern)
    hourly_pattern = 0.002 * np.sin(np.linspace(0, 2 * np.pi * (n_samples / 60), n_samples))
    prices = prices * (1 + hourly_pattern)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': symbol,
        'open': prices * (1 + np.random.normal(0, 0.0005, n_samples)),
        'high': prices * (1 + np.random.normal(0.001, 0.0007, n_samples)),
        'low': prices * (1 + np.random.normal(-0.001, 0.0007, n_samples)),
        'close': prices,
        'volume': np.random.normal(1000000, 200000, n_samples) * (1 + 0.5 * hourly_pattern)
    })
    
    # Fix high/low values to be consistent
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
    
    return data

def test_feature_engineering():
    """Test the feature engineering module"""
    print("\n=== Testing Feature Engineering ===")
    
    # Generate sample data
    data = generate_sample_data(n_samples=200)
    
    # Set timestamp as index and convert to OHLCV format
    df = data.copy()
    df = df.set_index('timestamp')
    symbol = df['symbol'].iloc[0]
    df = df.drop('symbol', axis=1)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Extract features
    engineer = FeatureEngineer()
    features = engineer.extract_features(df, symbol)
    features = engineer.clean_features(features)
    
    print(f"Generated {len(features.columns)} features")
    print("Feature sample:")
    print(features.head(3))
    
    # Plot some features
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(features.index, features['close'], label='Close Price')
    plt.plot(features.index, features['ma_20'], label='MA(20)')
    plt.legend()
    plt.title('Price and Moving Average')
    
    plt.subplot(3, 1, 2)
    plt.plot(features.index, features['rsi_14'], label='RSI(14)')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.legend()
    plt.title('RSI Indicator')
    
    plt.subplot(3, 1, 3)
    plt.plot(features.index, features['volatility_20d'], label='20d Volatility')
    plt.legend()
    plt.title('Volatility')
    
    plt.tight_layout()
    plt.savefig('feature_engineering_test.png')
    print("Saved feature plot to 'feature_engineering_test.png'")
    
    return features, df

def test_bayesian_ml_engine():
    """Test the Bayesian ML engine"""
    print("\n=== Testing Bayesian ML Engine ===")
    
    # Get features from previous test
    features, price_data = test_feature_engineering()
    
    # Add target (next return)
    features['target'] = price_data['Close'].pct_change().shift(-1)
    features = features.dropna(subset=['target'])
    
    # Split into training and test sets
    train_size = int(len(features) * 0.7)
    train_features = features.iloc[:train_size]
    test_features = features.iloc[train_size:]
    
    X_train = train_features.drop('target', axis=1)
    y_train = train_features['target']
    X_test = test_features.drop('target', axis=1)
    y_test = test_features['target']
    
    # Initialize and train the ML engine
    ml_engine = BayesianMLEngine(model_dir="test_models")
    ml_engine.train(X_train, y_train)
    
    # Make predictions
    predictions, model_preds = ml_engine.predict(X_test)
    mean_pred, uncertainty = ml_engine.predict_with_uncertainty(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - predictions) ** 2)
    print(f"Test MSE: {mse:.8f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_test.index, y_test.values, label='Actual Returns', alpha=0.5)
    plt.plot(y_test.index, predictions, label='Predicted Returns', alpha=0.8)
    plt.fill_between(
        y_test.index, 
        mean_pred - 2*uncertainty, 
        mean_pred + 2*uncertainty, 
        alpha=0.2, 
        color='orange', 
        label='Uncertainty (±2σ)'
    )
    plt.legend()
    plt.title('Return Predictions vs Actual')
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    cum_actual = (1 + y_test).cumprod()
    cum_pred_strategy = (1 + (predictions * np.sign(predictions))).cumprod()
    cum_buy_hold = (1 + y_test).cumprod()
    
    plt.plot(y_test.index, cum_actual, label='Actual Cumulative Return')
    plt.plot(y_test.index, cum_pred_strategy, label='Strategy Cumulative Return')
    plt.plot(y_test.index, cum_buy_hold, label='Buy & Hold Return')
    plt.legend()
    plt.title('Cumulative Returns')
    
    plt.tight_layout()
    plt.savefig('bayesian_ml_test.png')
    print("Saved prediction plot to 'bayesian_ml_test.png'")
    
    # Test online updating
    print("\nTesting online updates...")
    for i in range(5):
        # Get a small batch of data
        start_idx = train_size + i*5
        end_idx = start_idx + 5
        batch_X = features.iloc[start_idx:end_idx].drop('target', axis=1)
        batch_y = features.iloc[start_idx:end_idx]['target']
        
        # Update model
        updated = ml_engine.update_online(batch_X, batch_y)
        if updated:
            print(f"Batch {i+1}: Full model retrained")
        else:
            print(f"Batch {i+1}: Incremental update")
    
    return ml_engine

def test_online_trading_ml():
    """Test the full OnlineTradingML system"""
    print("\n=== Testing Online Trading ML System ===")
    
    # Generate sample data
    data = generate_sample_data(n_samples=300)
    
    # Save to CSV for testing
    data.to_csv('test_data.csv', index=False)
    print("Saved test data to 'test_data.csv'")
    
    # Initialize the trading ML system
    trading_ml = OnlineTradingML(model_dir="test_trading_models", confidence_threshold=0.6)
    
    # Load the data
    market_data = trading_ml.load_market_data('test_data.csv')
    
    if not market_data:
        print("Error loading market data")
        return
    
    # Get the first symbol
    symbol = list(market_data.keys())[0]
    
    # Initial training
    result = trading_ml.process_new_data(symbol, market_data[symbol])
    print(f"Initial training result: {result}")
    
    # Test prediction on new data
    # Generate some new data
    new_data = generate_sample_data(n_samples=10, symbol=symbol)
    new_data.to_csv('test_new_data.csv', index=False)
    
    # Load and process the new data
    new_market_data = trading_ml.load_market_data('test_new_data.csv')
    
    # Update the model with new data
    for i in range(5):
        # Process one bar at a time to simulate real-time updates
        single_bar = new_market_data[symbol].iloc[[i]]
        update_result = trading_ml.process_new_data(symbol, single_bar)
        print(f"Update {i+1} result: {update_result}")
    
    # Save state
    trading_ml.save_state()
    print("Saved trading ML state")
    
    return trading_ml

if __name__ == "__main__":
    # Create test directories
    os.makedirs("test_models", exist_ok=True)
    os.makedirs("test_trading_models", exist_ok=True)
    
    # Run tests
    test_feature_engineering()
    test_bayesian_ml_engine()
    test_online_trading_ml()
    
    print("\nAll tests completed!") 