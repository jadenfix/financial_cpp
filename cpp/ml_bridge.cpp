#include "ml_bridge.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <map>

/**
 * @brief Simple test program for the ML Bridge
 * 
 * This program tests the ML Bridge by:
 * 1. Making a prediction with sample OHLCV data
 * 2. Making a prediction with a custom feature map
 * 3. Simulating a series of predictions in a trading loop
 */
int main(int argc, char* argv[]) {
    std::cout << "ML Bridge Test" << std::endl;
    std::cout << "==============" << std::endl;
    
    // Create ML Bridge
    MLBridge mlBridge;
    
    try {
        // Test 1: Simple OHLCV prediction
        std::cout << "\nTest 1: OHLCV Prediction" << std::endl;
        double open = 40000.0;
        double high = 41000.0;
        double low = 39500.0;
        double close = 40500.0;
        double volume = 1000.0;
        
        auto [pred1, uncert1] = mlBridge.getPredictionOHLCV(open, high, low, close, volume);
        
        std::cout << "Input: OHLCV = [" << open << ", " << high << ", " 
                  << low << ", " << close << ", " << volume << "]" << std::endl;
        std::cout << "Prediction: " << pred1 << std::endl;
        std::cout << "Uncertainty: " << uncert1 << std::endl;
        
        // Test 2: Custom feature map
        std::cout << "\nTest 2: Custom Feature Map" << std::endl;
        std::map<std::string, double> features = {
            {"btc_open", 40000.0},
            {"btc_high", 41000.0},
            {"btc_low", 39500.0},
            {"btc_close", 40500.0},
            {"btc_volume", 1000.0},
            {"btc_rsi_14", 55.5},
            {"btc_ma_20", 39800.0}
        };
        
        auto [pred2, uncert2] = mlBridge.getPrediction(features);
        
        std::cout << "Input: Custom features with " << features.size() << " elements" << std::endl;
        std::cout << "Prediction: " << pred2 << std::endl;
        std::cout << "Uncertainty: " << uncert2 << std::endl;
        
        // Test 3: Simulated trading loop
        std::cout << "\nTest 3: Simulated Trading Loop" << std::endl;
        std::cout << "Running 5 predictions with changing price..." << std::endl;
        
        // Start with initial price
        double currentPrice = 40000.0;
        
        for (int i = 0; i < 5; i++) {
            // Simulate price movement
            double priceChange = (std::rand() % 2000) - 1000;  // Random -1000 to +1000
            double newPrice = currentPrice + priceChange;
            
            // Create OHLCV data (simplified)
            double sim_open = currentPrice;
            double sim_high = std::max(currentPrice, newPrice) + (std::rand() % 200);
            double sim_low = std::min(currentPrice, newPrice) - (std::rand() % 200);
            double sim_close = newPrice;
            double sim_volume = 500 + (std::rand() % 1000);
            
            // Get prediction
            auto [pred, uncert] = mlBridge.getPredictionOHLCV(
                sim_open, sim_high, sim_low, sim_close, sim_volume);
            
            // Print results
            std::cout << "Iteration " << (i+1) << ":" << std::endl;
            std::cout << "  Price: " << currentPrice << " -> " << newPrice 
                      << " (Change: " << priceChange << ")" << std::endl;
            std::cout << "  Prediction: " << pred << " ± " << uncert << std::endl;
            
            // Simulate trading decision
            std::string decision = "HOLD";
            if (pred > 0.0002 && uncert < 0.0005) {
                decision = "BUY";
            } else if (pred < -0.0002 && uncert < 0.0005) {
                decision = "SELL";
            }
            std::cout << "  Decision: " << decision << std::endl;
            
            // Update current price for next iteration
            currentPrice = newPrice;
            
            // Pause to simulate real-time
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 