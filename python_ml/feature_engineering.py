import pandas as pd
import numpy as np
from scipy import stats
import talib

class FeatureEngineer:
    """
    Feature engineering class for financial time series data.
    Extracts technical indicators, statistical features, and market microstructure metrics.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, df, symbol=None):
        """
        Extract features from OHLCV data
        
        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
            symbol: Optional symbol name for multi-asset features
            
        Returns:
            DataFrame with extracted features
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=data.index)
        
        # Price features
        self._add_price_features(data, features)
        
        # Return features
        self._add_return_features(data, features)
        
        # Volatility features
        self._add_volatility_features(data, features)
        
        # Volume features
        self._add_volume_features(data, features)
        
        # Technical indicators
        self._add_technical_indicators(data, features)
        
        # Statistical features
        self._add_statistical_features(data, features)
        
        # Time features
        self._add_time_features(data, features)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features
    
    def _add_price_features(self, data, features):
        """Add price-based features"""
        # Price levels
        features['close'] = data['Close']
        features['open'] = data['Open']
        features['high'] = data['High']
        features['low'] = data['Low']
        
        # Price ratios
        features['high_low_ratio'] = data['High'] / data['Low']
        features['close_open_ratio'] = data['Close'] / data['Open']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = data['Close'].rolling(window=window).mean()
            features[f'close_ma_{window}_ratio'] = data['Close'] / features[f'ma_{window}']
    
    def _add_return_features(self, data, features):
        """Add return-based features"""
        # Simple returns
        features['return_1d'] = data['Close'].pct_change()
        
        # Log returns
        features['log_return_1d'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Cumulative returns
        for window in [5, 10, 20]:
            features[f'cum_return_{window}d'] = (
                data['Close'] / data['Close'].shift(window)
            ) - 1
            
        # Return momentum
        for window in [5, 10, 20]:
            features[f'return_mom_{window}d'] = features['return_1d'].rolling(window).sum()
    
    def _add_volatility_features(self, data, features):
        """Add volatility-based features"""
        # Historical volatility (std of log returns)
        for window in [5, 10, 20]:
            features[f'volatility_{window}d'] = features['log_return_1d'].rolling(window).std()
        
        # Parkinson volatility estimator
        features['parkinson_vol_5d'] = (
            np.sqrt(1 / (4 * np.log(2)) * 
                   (np.log(data['High'] / data['Low']) ** 2).rolling(5).mean())
        )
        
        # Garman-Klass volatility estimator
        features['garman_klass_vol_5d'] = np.sqrt(
            (0.5 * np.log(data['High'] / data['Low']) ** 2) -
            ((2 * np.log(2) - 1) * np.log(data['Close'] / data['Open']) ** 2)
        ).rolling(5).mean()
    
    def _add_volume_features(self, data, features):
        """Add volume-based features"""
        # Raw volume
        features['volume'] = data['Volume']
        
        # Volume changes
        features['volume_change_1d'] = data['Volume'].pct_change()
        
        # Volume moving averages
        for window in [5, 10, 20]:
            features[f'volume_ma_{window}'] = data['Volume'].rolling(window).mean()
            features[f'volume_ma_{window}_ratio'] = data['Volume'] / features[f'volume_ma_{window}']
        
        # Price-volume correlations
        for window in [5, 10, 20]:
            features[f'price_volume_corr_{window}d'] = (
                data['Close'].rolling(window).corr(data['Volume'])
            )
    
    def _add_technical_indicators(self, data, features):
        """Add technical indicators using TA-Lib"""
        try:
            # RSI
            features['rsi_14'] = talib.RSI(data['Close'].values, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                data['Close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                data['Close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
            features['bb_width'] = (upper - lower) / middle
            features['bb_position'] = (data['Close'] - lower) / (upper - lower)
            
            # ADX
            features['adx_14'] = talib.ADX(
                data['High'].values,
                data['Low'].values,
                data['Close'].values,
                timeperiod=14
            )
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                data['High'].values,
                data['Low'].values,
                data['Close'].values,
                fastk_period=5,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            
            # OBV
            features['obv'] = talib.OBV(data['Close'].values, data['Volume'].values)
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            # If TA-Lib fails, add basic versions of some indicators
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))
    
    def _add_statistical_features(self, data, features):
        """Add statistical features"""
        # Skewness and kurtosis of returns
        for window in [10, 20]:
            features[f'return_skew_{window}d'] = features['log_return_1d'].rolling(window).skew()
            features[f'return_kurt_{window}d'] = features['log_return_1d'].rolling(window).kurt()
        
        # Z-score of price
        for window in [10, 20]:
            features[f'price_zscore_{window}d'] = (
                (data['Close'] - data['Close'].rolling(window).mean()) / 
                data['Close'].rolling(window).std()
            )
    
    def _add_time_features(self, data, features):
        """Add time-based features if index is datetime"""
        if isinstance(data.index, pd.DatetimeIndex):
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['is_month_start'] = data.index.is_month_start.astype(int)
            features['is_month_end'] = data.index.is_month_end.astype(int)
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names
    
    def clean_features(self, features_df):
        """Clean features by handling NaN values"""
        # Forward fill NaN values first (for time series continuity)
        features_df = features_df.fillna(method='ffill')
        
        # Then backfill any remaining NaNs (typically at the beginning)
        features_df = features_df.fillna(method='bfill')
        
        # If still have NaNs, fill with zeros
        features_df = features_df.fillna(0)
        
        # Replace infinities with large values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        return features_df

# Example usage
if __name__ == "__main__":
    # Create sample OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.normal(100, 5, 100),
        'High': np.random.normal(105, 5, 100),
        'Low': np.random.normal(95, 5, 100),
        'Close': np.random.normal(102, 5, 100),
        'Volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Fix High/Low values to be consistent
    sample_data['High'] = np.maximum(
        sample_data[['Open', 'Close']].max(axis=1),
        sample_data['High']
    )
    sample_data['Low'] = np.minimum(
        sample_data[['Open', 'Close']].min(axis=1),
        sample_data['Low']
    )
    
    # Extract features
    engineer = FeatureEngineer()
    features = engineer.extract_features(sample_data)
    features = engineer.clean_features(features)
    
    # Print features
    print(f"Generated {len(features.columns)} features:")
    print(features.head()) 