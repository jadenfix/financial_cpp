#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <array>
#include <map>
#include <cstdio>
#include <stdexcept>

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
     */
    MLBridge(
        const std::string& pythonPath = "python3", 
        const std::string& scriptPath = "python_ml/ml_predictor.py",
        const std::string& modelDir = "python_ml/models",
        const std::string& asset = "btc"
    ) : pythonPath_(pythonPath), 
        scriptPath_(scriptPath),
        modelDir_(modelDir),
        asset_(asset) {}
    
    /**
     * @brief Get prediction for market data
     * 
     * @param marketData Map of feature name to value
     * @return std::pair<double, double> Prediction and uncertainty
     */
    std::pair<double, double> getPrediction(const std::map<std::string, double>& marketData) {
        // Convert market data to CSV
        std::stringstream ss;
        for (const auto& [key, value] : marketData) {
            ss << key << ",";
        }
        ss << "\n";
        for (const auto& [key, value] : marketData) {
            ss << value << ",";
        }
        ss << "\n";
        
        std::string csvData = ss.str();
        
        // Create command
        std::string command = pythonPath_ + " " + scriptPath_ + 
                              " --model-dir " + modelDir_ +
                              " --asset " + asset_;
        
        // Call Python script
        std::array<char, 128> buffer;
        std::string result;
        
        // Open pipe to process
        FILE* pipe = popen(command.c_str(), "w+");
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        
        // Write CSV data to pipe
        fwrite(csvData.c_str(), sizeof(char), csvData.size(), pipe);
        fflush(pipe);
        
        // Read output from pipe
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        
        // Close pipe
        int status = pclose(pipe);
        if (status != 0) {
            throw std::runtime_error("Python process failed with code " + std::to_string(status));
        }
        
        // Parse result
        double prediction = 0.0;
        double uncertainty = 0.0;
        
        std::stringstream resultStream(result);
        std::string line;
        if (std::getline(resultStream, line)) {
            std::stringstream lineStream(line);
            std::string predStr, uncertStr;
            
            if (std::getline(lineStream, predStr, ',') && 
                std::getline(lineStream, uncertStr, ',')) {
                prediction = std::stod(predStr);
                uncertainty = std::stod(uncertStr);
            }
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
}; 