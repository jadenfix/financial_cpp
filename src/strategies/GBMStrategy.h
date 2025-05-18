#pragma once
#include "Strategy.h"
#include "../core/Event.h"
#include "../core/EventQueue.h"
#include <vector>
#include <string>
#include <random>
#include <cmath>

// Geometric Brownian Motion (GBM) Strategy
class GBMStrategy : public Strategy {
public:
    GBMStrategy(const std::string& symbol, double drift = 0.0, double volatility = 0.0)
        : symbol_(symbol), drift_(drift), volatility_(volatility) {}

    void handle_market_event(const MarketEvent& event, EventQueue& queue) override {
        // Find the symbol's price in the event data
        auto it = event.data.find(symbol_);
        if (it == event.data.end()) return;
        double price = it->second.Close;
        if (!price_history_.empty()) {
            double last_price = price_history_.back();
            double log_return = std::log(price / last_price);
            log_returns_.push_back(log_return);
        }
        price_history_.push_back(price);
        // Estimate parameters after enough data
        if (log_returns_.size() > 20) {
            double mean = 0.0, var = 0.0;
            for (double r : log_returns_) mean += r;
            mean /= log_returns_.size();
            for (double r : log_returns_) var += (r - mean) * (r - mean);
            var /= log_returns_.size();
            drift_ = mean;
            volatility_ = std::sqrt(var);
            // Simple signal: buy if drift > 0, sell if drift < 0
            if (drift_ > 0.0) {
                send_event(std::make_shared<SignalEvent>(event.timestamp, symbol_, SignalDirection::LONG), queue);
            } else if (drift_ < 0.0) {
                send_event(std::make_shared<SignalEvent>(event.timestamp, symbol_, SignalDirection::SHORT), queue);
            }
        }
    }

    // Optionally override handle_fill_event if needed

private:
    std::string symbol_;
    double drift_;
    double volatility_;
    std::vector<double> price_history_;
    std::vector<double> log_returns_;
    // Add more state as needed
}; 