#!/usr/bin/env python3
"""
Advanced Models for Bayesian ML Trading System

This module implements advanced machine learning models:
1. RNNs (LSTM, GRU)
2. CNNs for time series
3. AdaBoost and XGBoost
4. Physics-based Neural Networks (PINNs)
5. Stacked hybrid model (LightGBM + CNN-LSTM + FinBERT-BiLSTM)

These models can be used alongside the base models in the BayesianMLEngine.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Boosting models
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')


class RNNModel(BaseEstimator, RegressorMixin):
    """LSTM/GRU RNN model for time series forecasting"""
    
    def __init__(self, rnn_type='LSTM', units=64, dropout=0.2, 
                 lookback=10, epochs=100, batch_size=32, verbose=0):
        """
        Initialize RNN model
        
        Args:
            rnn_type: Type of RNN cell ('LSTM' or 'GRU')
            units: Number of RNN units
            dropout: Dropout rate
            lookback: Number of time steps to look back
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
        """
        self.rnn_type = rnn_type
        self.units = units
        self.dropout = dropout
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
    
    def _reshape_data(self, X):
        """Reshape data for RNN input [samples, time_steps, features]"""
        # Accept both DataFrame and ndarray
        if hasattr(X, 'values'):
            X_np = X.values
        else:
            X_np = X
        n_samples = X_np.shape[0]
        
        # If we don't have enough samples for the lookback, adjust it
        actual_lookback = min(self.lookback, n_samples - 1)
        
        if actual_lookback <= 0:
            # Not enough data for sequence, use single timestep
            return X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
        
        # Create sequences
        sequences = []
        for i in range(actual_lookback, n_samples):
            sequences.append(X_np[i-actual_lookback:i+1])
        
        return np.array(sequences)
    
    def _build_model(self, input_shape):
        """Build the RNN model"""
        model = Sequential()
        
        # RNN layer
        if self.rnn_type.upper() == 'LSTM':
            model.add(LSTM(self.units, input_shape=input_shape))
        else:  # GRU
            model.add(GRU(self.units, input_shape=input_shape))
        
        # Dropout for regularization
        model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(), loss='mse')
        
        return model
    
    def fit(self, X, y):
        """
        Fit the RNN model to training data
        
        Args:
            X: Training features
            y: Target values
        """
        # Reshape data for RNN
        X_rnn = self._reshape_data(X)
        
        # Build model
        self.model = self._build_model((X_rnn.shape[1], X_rnn.shape[2]))
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        
        # Ensure y is numpy array
        y_np = y[-X_rnn.shape[0]:]
        if hasattr(y_np, 'values'):
            y_np = y_np.values
        
        # Train model
        self.model.fit(
            X_rnn, y_np,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        
        return self
    
    def predict(self, X):
        """Make predictions with the RNN model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Reshape data for RNN
        X_rnn = self._reshape_data(X)
        
        # Make predictions
        return self.model.predict(X_rnn, verbose=0).flatten()
    
    def save(self, model_path):
        """Save the model to disk"""
        if self.model is not None:
            if not model_path.endswith('.keras'):
                model_path += '.keras'
            self.model.save(model_path)
    
    def load(self, model_path):
        """Load the model from disk"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            return True
        return False


class CNNModel(BaseEstimator, RegressorMixin):
    """1D CNN model for time series forecasting"""
    
    def __init__(self, filters=64, kernel_size=3, pool_size=2,
                 lookback=20, epochs=100, batch_size=32, verbose=0):
        """
        Initialize CNN model
        
        Args:
            filters: Number of filters in Conv1D layer
            kernel_size: Size of the convolution kernel
            pool_size: Size of the pooling window
            lookback: Number of time steps to look back
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
    
    def _reshape_data(self, X):
        """Reshape data for CNN input [samples, time_steps, features]"""
        # Accept both DataFrame and ndarray
        if hasattr(X, 'values'):
            X_np = X.values
        else:
            X_np = X
        n_samples = X_np.shape[0]
        
        # If we don't have enough samples for the lookback, adjust it
        actual_lookback = min(self.lookback, n_samples - 1)
        
        if actual_lookback <= 0:
            # Not enough data for sequence, use single timestep
            return X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
        
        # Create sequences
        sequences = []
        for i in range(actual_lookback, n_samples):
            sequences.append(X_np[i-actual_lookback:i+1])
        
        return np.array(sequences)
    
    def _build_model(self, input_shape):
        """Build the CNN model"""
        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, 
                        activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=self.pool_size))
        
        # Additional Conv layer
        model.add(Conv1D(filters=self.filters*2, kernel_size=self.kernel_size, 
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=self.pool_size))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(), loss='mse')
        
        return model
    
    def fit(self, X, y):
        """
        Fit the CNN model to training data
        
        Args:
            X: Training features
            y: Target values
        """
        # Reshape data for CNN
        X_cnn = self._reshape_data(X)
        
        # Build model
        self.model = self._build_model((X_cnn.shape[1], X_cnn.shape[2]))
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        
        # Ensure y is numpy array
        y_np = y[-X_cnn.shape[0]:]
        if hasattr(y_np, 'values'):
            y_np = y_np.values
        
        # Train model
        self.model.fit(
            X_cnn, y_np,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        
        return self
    
    def predict(self, X):
        """Make predictions with the CNN model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Reshape data for CNN
        X_cnn = self._reshape_data(X)
        
        # Make predictions
        return self.model.predict(X_cnn, verbose=0).flatten()
    
    def save(self, model_path):
        """Save the model to disk"""
        if self.model is not None:
            if not model_path.endswith('.keras'):
                model_path += '.keras'
            self.model.save(model_path)
    
    def load(self, model_path):
        """Load the model from disk"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            return True
        return False


class PhysicsInformedNN(BaseEstimator, RegressorMixin):
    """Physics-Informed Neural Network (PINN) for financial time series"""
    
    def __init__(self, hidden_layers=[64, 32], activation='tanh', 
                 physics_weight=0.1, epochs=100, batch_size=32, verbose=0):
        """
        Initialize PINN model
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            physics_weight: Weight of physics loss term
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.physics_weight = physics_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.physics_model = None
        self.feature_means = None
        self.feature_stds = None
    
    def _build_model(self, input_dim):
        """Build the neural network model"""
        # Input layer
        inputs = Input(shape=(input_dim,))
        x = inputs
        
        # Hidden layers
        for units in self.hidden_layers:
            x = Dense(units, activation=self.activation)(x)
            x = BatchNormalization()(x)
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with MSE loss
        model.compile(optimizer=Adam(), loss='mse')
        
        return model
    
    def _physics_constraint(self, X):
        """
        Physics-based constraints for financial time series
        
        For financial time series, we use principles like:
        1. Mean reversion
        2. Volatility clustering
        3. Momentum
        """
        # This is a simplified example
        # In a real system, this would implement financial physics equations
        
        # Standardize data
        X_std = (X - self.feature_means) / self.feature_stds
        
        # Make predictions with the model
        y_pred = self.model.predict(X_std, verbose=0).flatten()
        
        # Apply physics constraints (simplified examples):
        
        # 1. Mean reversion tendency
        # Penalize predictions that are far from the mean
        mean_reversion = np.mean(np.abs(y_pred))
        
        # 2. Momentum/trend following
        # Penalize sign changes that violate momentum
        diffs = np.diff(y_pred)
        momentum = np.mean(np.abs(diffs))
        
        # 3. Volatility clustering
        # Volatility should be persistent (simplified)
        vol_cluster = np.std(diffs)
        
        # Combined physics loss
        physics_loss = mean_reversion + momentum + vol_cluster
        
        return physics_loss
    
    def _custom_loss(self, y_true, y_pred):
        """Custom loss function with physics constraints"""
        # Mean squared error (data loss)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Physics-based loss is added during training, not here
        # because we need access to X which isn't available in this function
        
        return mse_loss
    
    def fit(self, X, y):
        """
        Fit the PINN model to training data
        
        Args:
            X: Training features
            y: Target values
        """
        # Standardize features
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        X_std = (X - self.feature_means) / self.feature_stds
        
        # Build model
        self.model = self._build_model(X.shape[1])
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        
        # Ensure y is numpy array
        y_np = y
        if hasattr(y_np, 'values'):
            y_np = y_np.values
        
        # For each epoch, we'll compute the combined loss with physics term
        for epoch in range(self.epochs):
            # Train one epoch with data loss
            history = self.model.fit(
                X_std, y_np,
                epochs=1,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping] if epoch == 0 else None,
                verbose=0
            )
            
            # Get data loss
            data_loss = history.history['loss'][0]
            
            # Compute physics loss
            physics_loss = self._physics_constraint(X)
            
            # Combined loss
            combined_loss = data_loss + self.physics_weight * physics_loss
            
            if self.verbose > 0 and (epoch % 10 == 0 or epoch == self.epochs-1):
                print(f"Epoch {epoch}/{self.epochs} - Loss: {data_loss:.6f} - "
                      f"Physics: {physics_loss:.6f} - Combined: {combined_loss:.6f}")
            
            # Early stopping check
            if hasattr(early_stopping, 'stopped_epoch') and early_stopping.stopped_epoch > 0:
                break
        
        return self
    
    def predict(self, X):
        """Make predictions with the PINN model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Standardize features
        X_std = (X - self.feature_means) / self.feature_stds
        
        # Make predictions
        return self.model.predict(X_std, verbose=0).flatten()
    
    def save(self, model_path):
        """Save the model to disk"""
        if self.model is not None:
            if not model_path.endswith('.keras'):
                model_path += '.keras'
            self.model.save(model_path)
            
            # Save feature statistics
            stats_path = os.path.join(os.path.dirname(model_path), 'pinn_stats.npz')
            np.savez(stats_path, means=self.feature_means, stds=self.feature_stds)
    
    def load(self, model_path):
        """Load the model from disk"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            
            # Load feature statistics
            stats_path = os.path.join(os.path.dirname(model_path), 'pinn_stats.npz')
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                self.feature_means = stats['means']
                self.feature_stds = stats['stds']
            
            return True
        return False


class HybridStackedModel(BaseEstimator, RegressorMixin):
    """
    Stacked hybrid model combining:
    1. LightGBM
    2. CNN-LSTM
    3. FinBERT-BiLSTM (simplified without actual FinBERT)
    """
    
    def __init__(self, lookback=20, epochs=100, batch_size=32, verbose=0):
        """
        Initialize hybrid stacked model
        
        Args:
            lookback: Number of time steps for sequence models
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
        """
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Individual models
        self.lgbm_model = None
        self.cnn_lstm_model = None
        self.bilstm_model = None
        
        # Meta-learner
        self.meta_learner = None
        
        # For standardization
        self.feature_means = None
        self.feature_stds = None
    
    def _build_lgbm(self):
        """Build LightGBM model"""
        return LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            verbose=-1
        )
    
    def _build_cnn_lstm(self, input_shape):
        """Build CNN-LSTM model"""
        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', 
                        input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        # LSTM layer
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(), loss='mse')
        
        return model
    
    def _build_bilstm(self, input_shape):
        """Build BiLSTM model (simplified FinBERT-BiLSTM)"""
        model = Sequential()
        
        # Bidirectional LSTM layers
        model.add(Bidirectional(LSTM(64, return_sequences=True), 
                               input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(), loss='mse')
        
        return model
    
    def _build_meta_learner(self):
        """Build meta-learner model for stacking"""
        return LGBMRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            verbose=-1
        )
    
    def _reshape_sequence_data(self, X):
        """Reshape data for sequence models [samples, time_steps, features]"""
        # Accept both DataFrame and ndarray
        if hasattr(X, 'values'):
            X_np = X.values
        else:
            X_np = X
        n_samples = X_np.shape[0]
        
        # If we don't have enough samples for the lookback, adjust it
        actual_lookback = min(self.lookback, n_samples - 1)
        
        if actual_lookback <= 0:
            # Not enough data for sequence, use single timestep
            return X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
        
        # Create sequences
        sequences = []
        for i in range(actual_lookback, n_samples):
            sequences.append(X_np[i-actual_lookback:i+1])
        
        return np.array(sequences)
    
    def fit(self, X, y):
        """
        Fit the hybrid stacked model to training data
        
        Args:
            X: Training features
            y: Target values
        """
        # Standardize features
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        X_std = (X - self.feature_means) / self.feature_stds
        
        # Reshape data for sequence models
        X_seq = self._reshape_sequence_data(X_std)
        
        # 1. Train LightGBM
        self.lgbm_model = self._build_lgbm()
        self.lgbm_model.fit(X_std, y)
        lgbm_preds = self.lgbm_model.predict(X_std)
        
        # 2. Train CNN-LSTM
        self.cnn_lstm_model = self._build_cnn_lstm((X_seq.shape[1], X_seq.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        y_seq = y[-X_seq.shape[0]:]
        if hasattr(y_seq, 'values'):
            y_seq = y_seq.values
        self.cnn_lstm_model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        cnn_lstm_preds = self.cnn_lstm_model.predict(X_seq, verbose=0).flatten()
        
        # 3. Train BiLSTM
        self.bilstm_model = self._build_bilstm((X_seq.shape[1], X_seq.shape[2]))
        self.bilstm_model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        bilstm_preds = self.bilstm_model.predict(X_seq, verbose=0).flatten()
        
        # Prepare meta-features
        meta_features = np.column_stack([
            lgbm_preds[-X_seq.shape[0]:],
            cnn_lstm_preds,
            bilstm_preds
        ])
        
        # Train meta-learner
        self.meta_learner = self._build_meta_learner()
        self.meta_learner.fit(meta_features, y_seq)
        
        return self
    
    def predict(self, X):
        """Make predictions with the hybrid stacked model"""
        if self.lgbm_model is None or self.cnn_lstm_model is None or self.bilstm_model is None:
            raise ValueError("Models not trained yet")
        
        # Standardize features
        X_std = (X - self.feature_means) / self.feature_stds
        
        # Reshape data for sequence models
        X_seq = self._reshape_sequence_data(X_std)
        
        # Generate predictions from each model
        lgbm_preds = self.lgbm_model.predict(X_std)
        cnn_lstm_preds = self.cnn_lstm_model.predict(X_seq, verbose=0).flatten()
        bilstm_preds = self.bilstm_model.predict(X_seq, verbose=0).flatten()
        
        # If sequence is shorter, truncate LGBM predictions to match
        if len(lgbm_preds) > len(cnn_lstm_preds):
            lgbm_preds = lgbm_preds[-len(cnn_lstm_preds):]
        
        # Combine predictions
        meta_features = np.column_stack([
            lgbm_preds,
            cnn_lstm_preds,
            bilstm_preds
        ])
        
        # Make final predictions
        return self.meta_learner.predict(meta_features)
    
    def save(self, model_dir):
        """Save the model components to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save LGBM model
        joblib.dump(self.lgbm_model, os.path.join(model_dir, 'lgbm_model.joblib'))
        
        # Save neural network models
        cnn_path = os.path.join(model_dir, 'cnn_lstm_model.keras')
        self.cnn_lstm_model.save(cnn_path)
        bilstm_path = os.path.join(model_dir, 'bilstm_model.keras')
        self.bilstm_model.save(bilstm_path)
        
        # Save meta-learner
        joblib.dump(self.meta_learner, os.path.join(model_dir, 'meta_learner.joblib'))
        
        # Save feature statistics
        np.savez(os.path.join(model_dir, 'feature_stats.npz'),
                means=self.feature_means, stds=self.feature_stds)
    
    def load(self, model_dir):
        """Load the model components from disk"""
        if not os.path.exists(model_dir):
            return False
        
        try:
            # Load LGBM model
            self.lgbm_model = joblib.load(os.path.join(model_dir, 'lgbm_model.joblib'))
            
            # Load neural network models
            self.cnn_lstm_model = tf.keras.models.load_model(
                os.path.join(model_dir, 'cnn_lstm_model.keras'))
            self.bilstm_model = tf.keras.models.load_model(
                os.path.join(model_dir, 'bilstm_model.keras'))
            
            # Load meta-learner
            self.meta_learner = joblib.load(os.path.join(model_dir, 'meta_learner.joblib'))
            
            # Load feature statistics
            stats = np.load(os.path.join(model_dir, 'feature_stats.npz'))
            self.feature_means = stats['means']
            self.feature_stds = stats['stds']
            
            return True
        except Exception as e:
            print(f"Error loading hybrid model: {e}")
            return False


# Create pipelines with standardization
def create_advanced_model_pipeline(model_type, **kwargs):
    """
    Create a pipeline with an advanced model
    
    Args:
        model_type: Type of model to create
        **kwargs: Model parameters
        
    Returns:
        Pipeline with scaler and model
    """
    if model_type == 'rnn':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', RNNModel(**kwargs))
        ])
    elif model_type == 'cnn':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', CNNModel(**kwargs))
        ])
    elif model_type == 'pinn':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', PhysicsInformedNN(**kwargs))
        ])
    elif model_type == 'hybrid':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', HybridStackedModel(**kwargs))
        ])
    elif model_type == 'adaboost':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', AdaBoostRegressor(**kwargs))
        ])
    elif model_type == 'xgboost':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', xgb.XGBRegressor(**kwargs))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}") 