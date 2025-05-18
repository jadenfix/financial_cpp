#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <array>
#include <map>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <regex>

/**
 * @brief ML Bridge for calling Python ML models from C++
 * 
 * This class provides an interface for the C++ trading engine to call
 * the Python ML predictor for making predictions with trained models.
 */
class MLBridge {
public:
    /**
     * @brief Construct a new MLBridge object
     * 
     * @param pythonPath Path to Python executable
     * @param scriptPath Path to ML predictor script
     * @param modelDir Directory containing trained models
     * @param asset Asset to predict (e.g., "btc", "eth")
     * @param useMock Whether to use mock model instead of real model
     * @param loadFeatureCols Whether to load feature_cols.txt (True for production)
     */
    MLBridge(
        const std::string& pythonPath = "python3", 
        const std::string& scriptPath = "/Users/jadenfix/financial_cpp-1/python_ml/ml_predictor.py",
        const std::string& modelDir = "/Users/jadenfix/financial_cpp-1/python_ml/models_test",
        const std::string& asset = "btc",
        bool useMock = true,
        bool loadFeatureCols = true
    ) : pythonPath_(pythonPath), 
        scriptPath_(scriptPath),
        modelDir_(modelDir),
        asset_(asset),
        useMock_(useMock) {
            // Print configuration
            std::cout << "ML Bridge Configuration:" << std::endl;
            std::cout << "  Python Path: " << pythonPath_ << std::endl;
            std::cout << "  Script Path: " << scriptPath_ << std::endl;
            std::cout << "  Model Directory: " << modelDir_ << std::endl;
            std::cout << "  Asset: " << asset_ << std::endl;
            std::cout << "  Using Mock Model: " << (useMock_ ? "Yes" : "No") << std::endl;
            
            // Load feature columns if requested
            if (loadFeatureCols) {
                loadFeatureColumns();
            }
        }
    
    /**
     * @brief Load feature column names from feature_cols.txt
     * 
     * @return bool True if successfully loaded, false otherwise
     */
    bool loadFeatureColumns() {
        // Path to feature columns file
        std::string featureColsPath = modelDir_ + "/" + asset_ + "/feature_cols.txt";
        
        // Open feature columns file
        std::ifstream featureFile(featureColsPath);
        if (!featureFile.is_open()) {
            std::cerr << "Warning: Could not open feature columns file: " << featureColsPath << std::endl;
            return false;
        }
        
        // Read feature columns
        featureColumns_.clear();
        std::string line;
        while (std::getline(featureFile, line)) {
            if (!line.empty()) {
                // Trim whitespace
                line.erase(0, line.find_first_not_of(" \n\r\t"));
                line.erase(line.find_last_not_of(" \n\r\t") + 1);
                
                if (!line.empty()) {
                    featureColumns_.push_back(line);
                }
            }
        }
        
        std::cout << "Loaded " << featureColumns_.size() << " feature columns" << std::endl;
        if (!featureColumns_.empty()) {
            std::cout << "  First 5 features: ";
            for (size_t i = 0; i < std::min(featureColumns_.size(), size_t(5)); ++i) {
                std::cout << featureColumns_[i] << " ";
            }
            std::cout << "..." << std::endl;
        }
        
        return !featureColumns_.empty();
    }
    
    /**
     * @brief Get prediction for market data
     * 
     * @param marketData Map of feature name to value
     * @return std::pair<double, double> Prediction and uncertainty
     */
    std::pair<double, double> getPrediction(const std::map<std::string, double>& marketData) {
        // Create a complete feature map with all required features
        std::map<std::string, double> completeFeatures;
        
        // If we have feature columns, use them to ensure all required features are present
        if (!featureColumns_.empty()) {
            // Initialize all features to 0.0
            for (const auto& col : featureColumns_) {
                completeFeatures[col] = 0.0;
            }
            
            // Copy provided values
            for (const auto& [key, value] : marketData) {
                completeFeatures[key] = value;
            }
        } else {
            // No feature columns loaded, just use the provided data
            completeFeatures = marketData;
        }
        
        // Format market data as JSON
        std::string jsonData = "{";
        bool first = true;
        for (const auto& [key, value] : completeFeatures) {
            if (!first) jsonData += ",";
            jsonData += "\"" + key + "\"" + ":" + std::to_string(value);
            first = false;
        }
        jsonData += "}";
        
        // Write JSON to temporary file to avoid stdin issues
        std::string tempFilePath = "/tmp/ml_bridge_input.json";
        std::ofstream tempFile(tempFilePath);
        if (!tempFile.is_open()) {
            std::cerr << "Error creating temporary file: " << tempFilePath << std::endl;
            throw std::runtime_error("Failed to create temporary file");
        }
        tempFile << jsonData;
        tempFile.close();
        
        // Create command with absolute paths and input file
        std::string command = pythonPath_ + " " + 
                             scriptPath_ + " " +
                             "--model-dir " + modelDir_ + " " +
                             "--asset " + asset_ + " " +
                             "--json-input " +
                             "--input-file " + tempFilePath;
                             
        // Add use-mock flag if needed
        if (useMock_) {
            command += " --use-mock";
        }
        
        std::cout << "Running command: " << command << std::endl;
        std::cout << "With JSON data (showing sample): {";
        int count = 0;
        for (const auto& [key, value] : completeFeatures) {
            if (count++ < 5) {
                std::cout << "\"" << key << "\":" << value << ",";
            } else {
                break;
            }
        }
        std::cout << "...} (" << completeFeatures.size() << " features)" << std::endl;
        
        // Call Python script
        std::array<char, 256> buffer;
        std::string result;
        
        // Open pipe to read output from process
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            std::cerr << "Error opening pipe: " << strerror(errno) << std::endl;
            throw std::runtime_error("popen() failed!");
        }
        
        // Read output from pipe
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        
        // Close pipe
        int status = pclose(pipe);
        if (status != 0) {
            std::cerr << "Python process failed with code " << status << std::endl;
            throw std::runtime_error("Python process failed with code " + std::to_string(status));
        }
        
        // Parse result
        double prediction = 0.0;
        double uncertainty = 0.0;
        
        std::cout << "Raw result: '" << result << "'" << std::endl;
        
        // Extract just the last line which contains our prediction values
        std::string lastLine;
        std::istringstream resultStream(result);
        std::string line;
        while (std::getline(resultStream, line)) {
            if (!line.empty()) {
                lastLine = line;
            }
        }
        
        // Parse the prediction and uncertainty
        if (!lastLine.empty()) {
            // Use regex to extract numbers from the last line - handles scientific notation
            std::regex numberPattern("([+-]?\\d+\\.?\\d*(?:[eE][+-]?\\d+)?)\\s*,\\s*([+-]?\\d+\\.?\\d*(?:[eE][+-]?\\d+)?)");
            std::smatch matches;
            
            if (std::regex_search(lastLine, matches, numberPattern) && matches.size() == 3) {
                try {
                    prediction = std::stod(matches[1].str());
                    uncertainty = std::stod(matches[2].str());
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing prediction values: " << e.what() << std::endl;
                    std::cerr << "Line: '" << lastLine << "'" << std::endl;
                    throw std::runtime_error("Failed to parse prediction values");
                }
            } else {
                std::cerr << "Could not find prediction pattern in: '" << lastLine << "'" << std::endl;
                throw std::runtime_error("Invalid prediction format");
            }
        } else {
            throw std::runtime_error("Empty prediction result");
        }
        
        return {prediction, uncertainty};
    }
    
    /**
     * @brief Get prediction using OHLCV data
     * 
     * @param open Open price
     * @param high High price
     * @param low Low price
     * @param close Close price
     * @param volume Volume
     * @return std::pair<double, double> Prediction and uncertainty
     */
    std::pair<double, double> getPredictionOHLCV(
        double open, double high, double low, double close, double volume
    ) {
        std::map<std::string, double> marketData = {
            {asset_ + "_open", open},
            {asset_ + "_high", high},
            {asset_ + "_low", low},
            {asset_ + "_close", close},
            {asset_ + "_volume", volume}
        };
        
        return getPrediction(marketData);
    }
    
private:
    std::string pythonPath_;
    std::string scriptPath_;
    std::string modelDir_;
    std::string asset_;
    bool useMock_;
    std::vector<std::string> featureColumns_;
}; 