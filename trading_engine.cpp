#include "ml_bridge.h"
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <thread>
#include <iomanip>
#include <fstream>
#include <ctime>

// Simple position class to track trades
class Position {
public:
    enum Type { LONG, SHORT, NONE };
    
    Position() : type(NONE), entryPrice(0.0), size(0.0), entryTime(0) {}
    
    void open(Type t, double price, double sz, time_t time) {
        type = t;
        entryPrice = price;
        size = sz;
        entryTime = time;
    }
    
    double close(double exitPrice, time_t exitTime) {
        double pnl = 0.0;
        
        if (type == LONG) {
            pnl = (exitPrice - entryPrice) * size;
        } else if (type == SHORT) {
            pnl = (entryPrice - exitPrice) * size;
        }
        
        // Reset position
        Type oldType = type;
        type = NONE;
        
        return pnl;
    }
    
    bool isOpen() const {
        return type != NONE;
    }
    
    Type getType() const {
        return type;
    }
    
    std::string getTypeString() const {
        switch(type) {
            case LONG: return "LONG";
            case SHORT: return "SHORT";
            default: return "NONE";
        }
    }
    
    double getEntryPrice() const {
        return entryPrice;
    }
    
    double getSize() const {
        return size;
    }
    
private:
    Type type;
    double entryPrice;
    double size;
    time_t entryTime;
};

// Trading strategy class
class MLTradingStrategy {
public:
    MLTradingStrategy(
        const std::string& asset = "btc",
        double initialBalance = 10000.0,
        double positionSizePercent = 0.1,
        double stopLossPercent = 0.02,
        double takeProfitPercent = 0.05
    ) : asset_(asset),
        balance_(initialBalance),
        positionSizePercent_(positionSizePercent),
        stopLossPercent_(stopLossPercent),
        takeProfitPercent_(takeProfitPercent),
        totalTrades_(0),
        winningTrades_(0),
        totalPnl_(0.0) {
        
        // Initialize ML Bridge
        mlBridge_ = std::make_unique<MLBridge>(
            "python3",
            "/Users/jadenfix/financial_cpp-1/python_ml/ml_predictor.py",
            "/Users/jadenfix/financial_cpp-1/python_ml/models_test",
            asset_,
            true  // Use mock model
        );
    }
    
    void processBar(double open, double high, double low, double close, double volume, time_t timestamp) {
        // Current price is the close price
        double currentPrice = close;
        
        // Check for stop loss or take profit if we have an open position
        if (position_.isOpen()) {
            if (checkStopLoss(currentPrice) || checkTakeProfit(currentPrice)) {
                return;  // Position was closed
            }
        }
        
        // Get prediction from ML model
        try {
            auto [prediction, uncertainty] = mlBridge_->getPredictionOHLCV(
                open, high, low, close, volume);
            
            std::cout << "ML Prediction: " << prediction << " ± " << uncertainty << std::endl;
            
            // Determine position size based on prediction strength and uncertainty
            double confidenceAdjustment = 1.0;
            if (uncertainty > 0.0) {
                confidenceAdjustment = std::min(1.0, 0.005 / uncertainty);
            }
            
            double positionSize = balance_ * positionSizePercent_ * confidenceAdjustment;
            
            // Trading logic - enter positions based on prediction
            if (!position_.isOpen()) {
                // Strong positive prediction - go LONG
                if (prediction > 0.002) {
                    enterPosition(Position::LONG, currentPrice, positionSize, timestamp);
                    std::cout << "SIGNAL: LONG - prediction: " << prediction 
                              << ", uncertainty: " << uncertainty << std::endl;
                }
                // Strong negative prediction - go SHORT
                else if (prediction < -0.002) {
                    enterPosition(Position::SHORT, currentPrice, positionSize, timestamp);
                    std::cout << "SIGNAL: SHORT - prediction: " << prediction 
                              << ", uncertainty: " << uncertainty << std::endl;
                } else {
                    std::cout << "SIGNAL: HOLD - prediction not strong enough" << std::endl;
                }
            }
            // Exit logic if we already have a position
            else {
                // If prediction changes significantly, consider exiting
                if ((position_.getType() == Position::LONG && prediction < -0.001) ||
                    (position_.getType() == Position::SHORT && prediction > 0.001)) {
                    double pnl = position_.close(currentPrice, timestamp);
                    totalPnl_ += pnl;
                    totalTrades_++;
                    if (pnl > 0) winningTrades_++;
                    
                    std::cout << "SIGNAL: EXIT " << position_.getTypeString() 
                              << " - contrary prediction: " << prediction << std::endl;
                    std::cout << "Trade PnL: " << pnl << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error getting prediction: " << e.what() << std::endl;
        }
    }
    
    // Print current strategy status
    void printStatus() {
        std::cout << "\n=== Strategy Status ===\n";
        std::cout << "Balance: $" << balance_ << std::endl;
        std::cout << "Current Position: " << position_.getTypeString();
        if (position_.isOpen()) {
            std::cout << " (Entry: " << position_.getEntryPrice() 
                      << ", Size: " << position_.getSize() << ")";
        }
        std::cout << std::endl;
        std::cout << "Total Trades: " << totalTrades_ << std::endl;
        std::cout << "Win Rate: " << (totalTrades_ > 0 ? (winningTrades_ * 100.0 / totalTrades_) : 0.0) << "%" << std::endl;
        std::cout << "Total PnL: $" << totalPnl_ << std::endl;
        std::cout << "=====================\n" << std::endl;
    }
    
private:
    // Enter a new position
    void enterPosition(Position::Type type, double price, double size, time_t timestamp) {
        if (position_.isOpen()) {
            // Close existing position first
            double pnl = position_.close(price, timestamp);
            totalPnl_ += pnl;
            totalTrades_++;
            if (pnl > 0) winningTrades_++;
            
            std::cout << "Closed existing position with PnL: " << pnl << std::endl;
        }
        
        // Open new position
        position_.open(type, price, size, timestamp);
        std::cout << "Opened " << position_.getTypeString() << " position at " << price 
                  << " with size " << size << std::endl;
    }
    
    // Check for stop loss
    bool checkStopLoss(double currentPrice) {
        if (!position_.isOpen()) return false;
        
        double stopLossLevel = 0.0;
        bool triggered = false;
        
        if (position_.getType() == Position::LONG) {
            stopLossLevel = position_.getEntryPrice() * (1.0 - stopLossPercent_);
            triggered = currentPrice < stopLossLevel;
        } else if (position_.getType() == Position::SHORT) {
            stopLossLevel = position_.getEntryPrice() * (1.0 + stopLossPercent_);
            triggered = currentPrice > stopLossLevel;
        }
        
        if (triggered) {
            double pnl = position_.close(currentPrice, std::time(nullptr));
            totalPnl_ += pnl;
            totalTrades_++;
            if (pnl > 0) winningTrades_++;  // Unlikely but possible with slippage
            
            std::cout << "STOP LOSS triggered at " << currentPrice << std::endl;
            std::cout << "Trade PnL: " << pnl << std::endl;
        }
        
        return triggered;
    }
    
    // Check for take profit
    bool checkTakeProfit(double currentPrice) {
        if (!position_.isOpen()) return false;
        
        double takeProfitLevel = 0.0;
        bool triggered = false;
        
        if (position_.getType() == Position::LONG) {
            takeProfitLevel = position_.getEntryPrice() * (1.0 + takeProfitPercent_);
            triggered = currentPrice > takeProfitLevel;
        } else if (position_.getType() == Position::SHORT) {
            takeProfitLevel = position_.getEntryPrice() * (1.0 - takeProfitPercent_);
            triggered = currentPrice < takeProfitLevel;
        }
        
        if (triggered) {
            double pnl = position_.close(currentPrice, std::time(nullptr));
            totalPnl_ += pnl;
            totalTrades_++;
            winningTrades_++;  // Take profit is always a winning trade
            
            std::cout << "TAKE PROFIT triggered at " << currentPrice << std::endl;
            std::cout << "Trade PnL: " << pnl << std::endl;
        }
        
        return triggered;
    }
    
private:
    std::string asset_;
    std::unique_ptr<MLBridge> mlBridge_;
    Position position_;
    double balance_;
    double positionSizePercent_;
    double stopLossPercent_;
    double takeProfitPercent_;
    int totalTrades_;
    int winningTrades_;
    double totalPnl_;
};

// Main function to run the backtest with simulated price data
int main() {
    std::cout << "ML Trading Engine - Simulation" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Create the trading strategy
    MLTradingStrategy strategy("btc", 10000.0, 0.1, 0.02, 0.05);
    
    // Start with BTC price around $40000
    double currentPrice = 40000.0;
    
    // Run simulation for 20 bars
    for (int i = 0; i < 20; i++) {
        std::cout << "\n--- Bar " << (i+1) << " ---" << std::endl;
        
        // Generate random price movement (more realistic than pure random)
        double trend = (i % 7 < 4) ? 0.001 : -0.001;  // Slight trend bias
        double randomMove = (std::rand() % 2000 - 1000) / 10.0;  // -100 to +100
        double volatility = (40000.0 * 0.02);  // 2% of starting price
        
        double priceChange = (trend * currentPrice) + ((randomMove / 1000.0) * volatility);
        double newPrice = currentPrice + priceChange;
        
        // Generate OHLCV data
        double open = currentPrice;
        double close = newPrice;
        double high = std::max(open, close) + (std::rand() % 100);
        double low = std::min(open, close) - (std::rand() % 100);
        double volume = 500 + (std::rand() % 1000);
        
        // Get current time
        std::time_t timestamp = std::time(nullptr);
        
        // Print current price
        std::cout << "Time: " << std::put_time(std::localtime(&timestamp), "%Y-%m-%d %H:%M:%S") << std::endl;
        std::cout << "Price: " << currentPrice << " -> " << newPrice 
                  << " (Change: " << priceChange << ")" << std::endl;
        
        // Process this bar
        strategy.processBar(open, high, low, close, volume, timestamp);
        
        // Update current price
        currentPrice = newPrice;
        
        // Print strategy status
        strategy.printStatus();
        
        // Pause between updates
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    std::cout << "\nSimulation complete!" << std::endl;
    
    return 0;
} 