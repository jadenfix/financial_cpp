#!/bin/bash
# Setup and Run Script for Advanced ML Trading System
# This script helps set up the environment, install dependencies, and run the ML system

echo "==== Advanced ML Trading System Setup and Run ===="
echo

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if Python 3 is installed
if ! command_exists python3; then
  echo "Error: Python 3 is required but not installed."
  echo "Please install Python 3 and try again."
  exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"

# Function to install dependencies
install_dependencies() {
  echo "Installing required Python packages..."
  
  # Basic dependencies
  echo "Installing basic dependencies..."
  python3 -m pip install numpy pandas scikit-learn matplotlib
  
  # ML dependencies
  echo "Installing ML dependencies..."
  python3 -m pip install xgboost joblib
  
  # Ask about installing TensorFlow (for neural networks)
  echo
  read -p "Install TensorFlow for neural network models? (y/n): " install_tf
  if [[ $install_tf == "y" || $install_tf == "Y" ]]; then
    echo "Installing TensorFlow..."
    python3 -m pip install tensorflow
  else
    echo "Skipping TensorFlow installation. Neural network models will not be available."
  fi
  
  # Ask about installing LightGBM
  echo
  read -p "Install LightGBM for hybrid models? (y/n): " install_lgbm
  if [[ $install_lgbm == "y" || $install_lgbm == "Y" ]]; then
    echo "Installing LightGBM..."
    python3 -m pip install lightgbm
  else
    echo "Skipping LightGBM installation. Hybrid stacked models will have limited functionality."
  fi
  
  echo "Dependencies installed successfully."
  echo
}

# Function to check data files
check_data_files() {
  echo "Checking for data files..."
  
  data_dir="data/ml_data"
  if [ ! -d "$data_dir" ]; then
    echo "Data directory not found. Creating: $data_dir"
    mkdir -p "$data_dir"
  fi
  
  # Check for asset data files
  assets=("btc" "eth" "ada" "solana")
  data_found=false
  
  for asset in "${assets[@]}"; do
    data_file="$data_dir/processed_data_$asset.csv"
    if [ -f "$data_file" ]; then
      echo "Found data file: $data_file"
      data_found=true
    fi
  done
  
  if [ "$data_found" = false ]; then
    echo "No data files found in $data_dir."
    echo "Please place your preprocessed data files in this directory."
    echo "Expected format: processed_data_btc.csv, processed_data_eth.csv, etc."
    echo
  fi
}

# Function to train models
train_models() {
  echo "==== Training ML Models ===="
  
  # Create directories
  mkdir -p python_ml/models
  
  # Ask for assets to train
  read -p "Enter assets to train (space-separated, e.g., btc eth): " assets_input
  
  if [ -z "$assets_input" ]; then
    assets_input="btc eth"
  fi
  
  # Convert to array
  read -ra assets <<< "$assets_input"
  
  # Ask for model types
  echo "Available model types:"
  echo "1. ensemble - Ensemble of multiple models (default)"
  echo "2. adaboost - AdaBoost regressor"
  echo "3. xgboost - XGBoost regressor"
  echo "4. rnn - Recurrent Neural Networks (requires TensorFlow)"
  echo "5. cnn - Convolutional Neural Networks (requires TensorFlow)"
  echo "6. pinn - Physics-Informed Neural Networks (requires TensorFlow)"
  echo "7. hybrid_stacked - Hybrid stacked model (requires TensorFlow and LightGBM)"
  echo "8. all - Train all available models"
  
  read -p "Enter model types to train (space-separated, e.g., ensemble xgboost): " model_types_input
  
  if [ -z "$model_types_input" ]; then
    model_types_input="ensemble"
  fi
  
  if [ "$model_types_input" = "all" ]; then
    model_types_arg=""
  else
    # Convert to array
    read -ra model_types <<< "$model_types_input"
    model_types_arg="--model-types ${model_types[*]}"
  fi
  
  # Ask about hyperparameter tuning
  read -p "Perform hyperparameter tuning? This will take longer. (y/n): " do_tune
  
  if [[ $do_tune == "n" || $do_tune == "N" ]]; then
    tune_arg="--no-tune"
  else
    tune_arg=""
  fi
  
  # Run training
  echo "Training models for assets: ${assets[*]}"
  python3 python_ml/train_all_models.py --assets "${assets[@]}" $model_types_arg $tune_arg
}

# Function to run predictions
run_predictions() {
  echo "==== Making Predictions with ML Models ===="
  
  # Ask for asset
  read -p "Enter an asset to predict (e.g., btc): " asset_input
  
  if [ -z "$asset_input" ]; then
    asset_input="btc"
  fi
  
  data_file="data/ml_data/processed_data_$asset_input.csv"
  
  if [ ! -f "$data_file" ]; then
    echo "Error: Data file not found: $data_file"
    return 1
  fi
  
  # Ask for model type
  echo "Available model types:"
  echo "1. ensemble - Ensemble of multiple models (default)"
  echo "2. adaboost - AdaBoost regressor"
  echo "3. xgboost - XGBoost regressor"
  echo "4. rnn - Recurrent Neural Networks"
  echo "5. cnn - Convolutional Neural Networks"
  echo "6. pinn - Physics-Informed Neural Networks"
  echo "7. hybrid_stacked - Hybrid stacked model"
  
  read -p "Enter model type to use for prediction: " model_type_input
  
  if [ -z "$model_type_input" ]; then
    model_type_input="ensemble"
  fi
  
  # Run prediction
  echo "Making predictions for $asset_input using $model_type_input model..."
  python3 python_ml/real_time_predictor.py --data-file "$data_file" --model-type "$model_type_input" --assets "$asset_input"
}

# Function to test the C++ integration
test_cpp_integration() {
  echo "==== Testing C++ Integration ===="
  
  # Ask for asset
  read -p "Enter an asset to use (e.g., btc): " asset_input
  
  if [ -z "$asset_input" ]; then
    asset_input="btc"
  fi
  
  data_file="data/ml_data/processed_data_$asset_input.csv"
  
  if [ ! -f "$data_file" ]; then
    echo "Error: Data file not found: $data_file"
    return 1
  fi
  
  # Ask for model type
  echo "Available model types:"
  echo "1. ensemble - Ensemble of multiple models (default)"
  echo "2. adaboost - AdaBoost regressor"
  echo "3. xgboost - XGBoost regressor"
  echo "4. rnn - Recurrent Neural Networks"
  echo "5. cnn - Convolutional Neural Networks"
  echo "6. pinn - Physics-Informed Neural Networks"
  echo "7. hybrid_stacked - Hybrid stacked model"
  
  read -p "Enter model type to use: " model_type_input
  
  if [ -z "$model_type_input" ]; then
    model_type_input="ensemble"
  fi
  
  # Output file
  output_file="cpp_integration_test_output.csv"
  
  # Run integration test
  echo "Testing C++ integration for $asset_input using $model_type_input model..."
  python3 python_ml/cpp_integration.py --mode predict --input "$data_file" --output "$output_file" --model-type "$model_type_input" --assets "$asset_input"
  
  # Show output
  if [ -f "$output_file" ]; then
    echo "Integration test successful. Results saved to $output_file"
    echo "Results preview:"
    head -n 10 "$output_file"
  else
    echo "Error: Integration test failed to produce output file."
  fi
}

# Main menu
show_menu() {
  echo
  echo "==== Advanced ML Trading System Menu ===="
  echo "1. Install dependencies"
  echo "2. Check data files"
  echo "3. Train ML models"
  echo "4. Run predictions"
  echo "5. Test C++ integration"
  echo "6. Run comprehensive tests"
  echo "7. Exit"
  echo
  read -p "Enter your choice (1-7): " choice
  
  case $choice in
    1) install_dependencies ;;
    2) check_data_files ;;
    3) train_models ;;
    4) run_predictions ;;
    5) test_cpp_integration ;;
    6) 
      echo "Running comprehensive tests..."
      python3 python_ml/test_advanced_models.py
      ;;
    7) 
      echo "Exiting."
      exit 0
      ;;
    *) 
      echo "Invalid option. Please try again."
      ;;
  esac
  
  # Return to menu
  show_menu
}

# Start the menu
show_menu 