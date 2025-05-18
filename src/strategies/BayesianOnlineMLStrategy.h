#pragma once
#include "Strategy.h"
#include "../core/Event.h"
#include "../core/EventQueue.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib> // For std::system
#include <deque>
#include <map>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

/**
 * BayesianOnlineMLStrategy: Advanced ML strategy with Bayesian updating
 * 
 * This strategy:
 * 1. Collects market data in real-time
 * 2. Extracts features using the Python feature engineering module
 * 3. Updates models online with each new data point
 * 4. Generates trade signals with uncertainty estimates
 * 5. Adjusts position sizes based on prediction confidence
 */
class BayesianOnlineMLStrategy : public Strategy {
public:
    BayesianOnlineMLStrategy(
        const std::string& symbol,
        size_t feature_window = 50,
        size_t update_interval = 1,  // Update every minute
        double confidence_threshold = 0.6,
        double position_size = 100.0
    ) : 
        symbol_(symbol),
        feature_window_(feature_window),
        update_interval_(update_interval),
        confidence_threshold_(confidence_threshold),
        position_size_(position_size),
        bar_count_(0),
        last_signal_(SignalDirection::FLAT),
        last_confidence_(0.0)
    {
        // Create directories if they don't exist
        std::system("mkdir -p python_ml/models");
    }

    void handle_market_event(const MarketEvent& event, EventQueue& queue) override {
        auto it = event.data.find(symbol_);
        if (it == event.data.end()) return;
        
        const PriceBar& bar = it->second;
        
        // Store market data
        market_data_.push_back(bar);
        if (market_data_.size() > feature_window_) {
            market_data_.pop_front();
        }
        
        // Need enough data to extract features
        if (market_data_.size() < feature_window_) return;
        
        // Update counter
        ++bar_count_;
        
        // Only process at specified intervals
        if (bar_count_ % update_interval_ != 0) return;
        
        // Export market data to CSV for Python processing
        export_market_data("market_data.csv");
        
        // Call Python script for online learning and prediction
        // Use 'update' mode to incrementally update the model
        std::string cmd = "python3 python_ml/online_ml_trading.py --mode update "
                         "--input market_data.csv --output predictions.csv "
                         "--symbol " + symbol_ + " --confidence " + 
                         std::to_string(confidence_threshold_);
        
        std::system(cmd.c_str());
        
        // Import predictions and generate signals
        import_predictions("predictions.csv", queue, event.timestamp);
    }
    
    void handle_fill_event(const FillEvent& event) override {
        // Track fills to manage positions
        if (event.symbol == symbol_) {
            std::cout << "Fill received for " << symbol_ 
                      << " Direction: " << (event.direction == OrderDirection::BUY ? "BUY" : "SELL")
                      << " Quantity: " << event.quantity
                      << " Price: " << event.fill_price << std::endl;
        }
    }

private:
    std::string symbol_;
    size_t feature_window_;
    size_t update_interval_;
    double confidence_threshold_;
    double position_size_;
    size_t bar_count_;
    std::deque<PriceBar> market_data_;
    SignalDirection last_signal_;
    double last_confidence_;
    
    // Export market data to CSV for Python processing
    void export_market_data(const std::string& filename) {
        std::ofstream ofs(filename);
        
        // Write header
        ofs << "timestamp,symbol,open,high,low,close,volume" << std::endl;
        
        // Write data
        for (const auto& bar : market_data_) {
            // Convert timestamp to ISO string
            auto time_t = std::chrono::system_clock::to_time_t(bar.timestamp);
            std::tm tm = *std::localtime(&time_t);
            char time_str[100];
            std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm);
            
            ofs << time_str << ","
                << symbol_ << ","
                << bar.Open << ","
                << bar.High << ","
                << bar.Low << ","
                << bar.Close << ","
                << bar.Volume << std::endl;
        }
    }
    
    // Import predictions and generate signals
    void import_predictions(const std::string& filename, EventQueue& queue, 
                           std::chrono::system_clock::time_point ts) {
        std::ifstream ifs(filename);
        std::string line;
        
        // Skip header
        std::getline(ifs, line);
        
        // Read predictions
        while (std::getline(ifs, line)) {
            // Parse CSV line
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string token;
            
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }
            
            // Check if we have enough tokens
            if (tokens.size() < 6) continue;
            
            // Extract prediction data
            std::string pred_symbol = tokens[1];
            double prediction = std::stod(tokens[2]);
            double uncertainty = std::stod(tokens[3]);
            double confidence = std::stod(tokens[4]);
            std::string signal_str = tokens[5];
            
            // Only process predictions for our symbol
            if (pred_symbol != symbol_) continue;
            
            // Convert signal string to enum
            SignalDirection signal = SignalDirection::FLAT;
            if (signal_str == "LONG") {
                signal = SignalDirection::LONG;
            } else if (signal_str == "SHORT") {
                signal = SignalDirection::SHORT;
            }
            
            // Only generate new signal if it's different or confidence changed significantly
            if (signal != last_signal_ || std::abs(confidence - last_confidence_) > 0.1) {
                // Generate signal
                send_event(std::make_shared<SignalEvent>(ts, symbol_, signal), queue);
                
                // Calculate position size based on confidence
                double adjusted_size = position_size_ * confidence;
                
                // Generate order based on signal
                if (signal == SignalDirection::LONG) {
                    send_event(std::make_shared<OrderEvent>(
                        ts, symbol_, OrderType::MARKET, OrderDirection::BUY, adjusted_size
                    ), queue);
                } else if (signal == SignalDirection::SHORT) {
                    send_event(std::make_shared<OrderEvent>(
                        ts, symbol_, OrderType::MARKET, OrderDirection::SELL, adjusted_size
                    ), queue);
                } else if (signal == SignalDirection::FLAT && last_signal_ != SignalDirection::FLAT) {
                    // Close any existing position
                    if (last_signal_ == SignalDirection::LONG) {
                        send_event(std::make_shared<OrderEvent>(
                            ts, symbol_, OrderType::MARKET, OrderDirection::SELL, position_size_
                        ), queue);
                    } else if (last_signal_ == SignalDirection::SHORT) {
                        send_event(std::make_shared<OrderEvent>(
                            ts, symbol_, OrderType::MARKET, OrderDirection::BUY, position_size_
                        ), queue);
                    }
                }
                
                // Update last signal and confidence
                last_signal_ = signal;
                last_confidence_ = confidence;
                
                // Log prediction details
                std::cout << "ML Prediction for " << symbol_ 
                          << " - Value: " << prediction
                          << " Uncertainty: " << uncertainty
                          << " Confidence: " << confidence
                          << " Signal: " << signal_str << std::endl;
            }
        }
    }
}; 